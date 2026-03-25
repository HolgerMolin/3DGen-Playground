"""
Training script for DiT on 3DGS data (class-conditional).
3DGS data (16384 points x 59 features) on 128x128 grid is the latent space directly — no VAE needed.

Single-GPU:  python dit/train.py --obj_list ... --gs_path ...
Multi-GPU:   accelerate launch [--num_processes N] dit/train.py --obj_list ... --gs_path ...
"""

import argparse
import json
import logging
import math
import os
import random
import sys
import tarfile
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image

from accelerate import Accelerator
from accelerate.utils import set_seed

# Add repo root to path for imports
REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Make gaussian-splatting submodule importable.
GS_ROOT = os.path.join(REPO_ROOT, "submodules", "gaussian-splatting")
if GS_ROOT not in sys.path:
    sys.path.insert(0, GS_ROOT)

from dataloaders.standard_3dgen_loader import Standard3DGenDataset
from dataloaders.class_3dgen_loader import (
    Class3DGenDataset, DC_ONLY_FEATURE_INDICES, FULL_3DGS_FEATURE_DIM,
    load_sphere2plane, plane_to_point_cloud,
)
from dit.models import DiT_3DGS_models
from dit.diffusion import create_diffusion


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RENDER_OPACITY_RAW_MIN = -12.0
RENDER_OPACITY_RAW_MAX = 12.0
RENDER_SCALE_RAW_MIN = -12.0
RENDER_SCALE_RAW_MAX = 8.0
RENDER_QUAT_EPS = 1e-8


#################################################################################
#                          Rendering Loss Helpers                               #
#################################################################################

def _tensor_debug_summary(tensor: torch.Tensor) -> dict[str, Any]:
    """Summarize a tensor for non-finite debugging without dumping full contents."""
    t = tensor.detach()
    finite_mask = torch.isfinite(t)
    finite_count = int(finite_mask.sum().item())
    summary: dict[str, Any] = {
        "shape": tuple(t.shape),
        "dtype": str(t.dtype),
        "device": str(t.device),
        "numel": int(t.numel()),
        "nonfinite": int(t.numel() - finite_count),
    }
    if finite_count > 0:
        finite_vals = t[finite_mask].float()
        summary.update({
            "min": float(finite_vals.min().item()),
            "max": float(finite_vals.max().item()),
            "mean": float(finite_vals.mean().item()),
        })
    return summary


def _debug_nonfinite_mse(
    *,
    args,
    diffusion,
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    x_full: Optional[torch.Tensor],
    t: torch.Tensor,
    noise: torch.Tensor,
    sample_losses: torch.Tensor,
    step: int,
    epoch: int,
    hash_keys: list[str],
    is_main: bool,
) -> None:
    """Fail fast with enough context to localize the first non-finite MSE."""
    bad_positions = (~torch.isfinite(sample_losses)).nonzero(as_tuple=False).flatten().tolist()
    if not bad_positions:
        bad_positions = list(range(min(1, x.shape[0])))

    with torch.no_grad():
        x_t_debug = diffusion.q_sample(x, t, noise=noise)
        model_out_debug = model(x_t_debug, t, y)
        target_debug = x if args.predict_xstart else noise

    bad_param_summaries = []
    total_bad_param_tensors = 0
    total_bad_param_values = 0
    for name, param in model.named_parameters():
        bad_count = int((~torch.isfinite(param)).sum().item())
        if bad_count == 0:
            continue
        total_bad_param_tensors += 1
        total_bad_param_values += bad_count
        bad_param_summaries.append({
            "name": name,
            "shape": tuple(param.shape),
            "nonfinite": bad_count,
        })
        if len(bad_param_summaries) >= 16:
            break

    per_sample_debug = []
    for pos in bad_positions[:4]:
        per_sample_debug.append({
            "batch_pos": int(pos),
            "hash_key": hash_keys[pos],
            "label": int(y[pos].detach().cpu().item()),
            "loss": float(sample_losses[pos].detach().float().cpu().item()),
            "x": _tensor_debug_summary(x[pos]),
            "x_t": _tensor_debug_summary(x_t_debug[pos]),
            "model_output": _tensor_debug_summary(model_out_debug[pos]),
            "target": _tensor_debug_summary(target_debug[pos]),
        })

    debug_payload = {
        "step": int(step),
        "epoch": int(epoch),
        "predict_xstart": bool(args.predict_xstart),
        "bad_positions": bad_positions,
        "hash_keys": [hash_keys[pos] for pos in bad_positions[:16]],
        "labels": [int(v) for v in y.detach().cpu().tolist()],
        "timesteps": [int(v) for v in t.detach().cpu().tolist()],
        "sample_loss_isfinite": torch.isfinite(sample_losses).detach().cpu(),
        "sample_losses": sample_losses.detach().cpu(),
        "bad_param_tensors": total_bad_param_tensors,
        "bad_param_values": total_bad_param_values,
        "bad_param_summaries": bad_param_summaries,
        "per_sample_debug": per_sample_debug,
        "x_bad_samples": x[bad_positions[:4]].detach().cpu(),
        "x_t_bad_samples": x_t_debug[bad_positions[:4]].detach().cpu(),
        "x_full_bad_samples": None if x_full is None else x_full[bad_positions[:4]].detach().cpu(),
        "model_output_bad_samples": model_out_debug[bad_positions[:4]].detach().cpu(),
        "noise_bad_samples": noise[bad_positions[:4]].detach().cpu(),
    }
    debug_path = os.path.join(args.results_dir, f"nonfinite_step_{step:07d}.pt")
    if is_main:
        torch.save(debug_payload, debug_path)
        logger.error(
            "[nonfinite] step=%d epoch=%d bad_positions=%s bad_hash_keys=%s debug_dump=%s",
            step,
            epoch,
            bad_positions,
            [hash_keys[pos] for pos in bad_positions[:16]],
            debug_path,
        )
        logger.error(
            "[nonfinite] bad_param_tensors=%d bad_param_values=%d bad_params=%s",
            total_bad_param_tensors,
            total_bad_param_values,
            bad_param_summaries,
        )
        for sample_info in per_sample_debug:
            logger.error("[nonfinite] sample_debug=%s", sample_info)
    raise FloatingPointError(f"Non-finite diffusion MSE detected at step {step}; debug dump saved to {debug_path}")


def _try_import_renderer():
    try:
        import gsplat
        return gsplat
    except Exception as e:
        return e


def _try_import_lpips():
    try:
        import lpips
        return lpips
    except Exception as e:
        return e


class _PipeConfig:
    convert_SHs_python = False
    compute_cov3D_python = False
    debug = False
    antialiasing = False


def _fov2focal(fov: float, pixels: int) -> float:
    return pixels / (2.0 * math.tan(fov / 2.0))


def _load_reference_cameras(ref_camera_tar: str) -> list:
    cams = []
    with tarfile.open(ref_camera_tar, "r:gz") as tar:
        json_members = [m for m in tar.getmembers() if m.name.endswith(".json")]
        json_members.sort(key=lambda m: m.name)
        for m in json_members:
            meta = json.loads(tar.extractfile(m).read().decode("utf-8"))
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, 0] = np.array(meta["x"], dtype=np.float32)
            c2w[:3, 1] = np.array(meta["y"], dtype=np.float32)
            c2w[:3, 2] = np.array(meta["z"], dtype=np.float32)
            c2w[:3, 3] = np.array(meta["origin"], dtype=np.float32)
            w2c = np.linalg.inv(c2w).astype(np.float32)
            fovx = float(meta["x_fov"])
            fovy = float(meta["y_fov"])
            width = int(meta.get("width", 512))
            height = int(meta.get("height", 512))
            cams.append({
                "R": w2c[:3, :3],
                "T": w2c[:3, 3],
                "fovx": fovx,
                "fovy": fovy,
                "width": width,
                "height": height,
            })
    if not cams:
        raise ValueError(f"No camera json found in {ref_camera_tar}")
    return cams


def _camera_viewmat_from_ref(ref_cam: dict) -> torch.Tensor:
    """Build a world-to-camera matrix matching the legacy renderer contract."""
    viewmat = np.zeros((4, 4), dtype=np.float32)
    viewmat[:3, :3] = np.asarray(ref_cam["R"], dtype=np.float32).transpose()
    viewmat[:3, 3] = np.asarray(ref_cam["T"], dtype=np.float32)
    viewmat[3, 3] = 1.0
    return torch.from_numpy(viewmat)


def _camera_intrinsics_from_ref(ref_cam: dict) -> torch.Tensor:
    width = int(ref_cam["width"])
    height = int(ref_cam["height"])
    fx = _fov2focal(float(ref_cam["fovx"]), width)
    fy = _fov2focal(float(ref_cam["fovy"]), height)
    K = torch.tensor(
        [
            [fx, 0.0, width * 0.5],
            [0.0, fy, height * 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    return K


def _prepare_train_cameras(ref_cameras: list, train_render_size: int, device: torch.device) -> dict[str, Any]:
    viewmats = []
    Ks = []
    for rc in ref_cameras:
        rc_small = dict(rc)
        rc_small["width"] = int(train_render_size)
        rc_small["height"] = int(train_render_size)
        viewmats.append(_camera_viewmat_from_ref(rc_small))
        Ks.append(_camera_intrinsics_from_ref(rc_small))
    return {
        "viewmats": torch.stack(viewmats, dim=0).to(device=device),
        "Ks": torch.stack(Ks, dim=0).to(device=device),
        "width": int(train_render_size),
        "height": int(train_render_size),
    }


def _plane_to_point_cloud_batch(planes: torch.Tensor, plane_to_sphere: torch.Tensor) -> torch.Tensor:
    """Convert plane grids (B, D, H, W) to sphere-ordered point clouds (B, N, D)."""
    if planes.ndim == 3:
        return plane_to_point_cloud(planes, plane_to_sphere).unsqueeze(0)
    if planes.ndim != 4:
        raise ValueError(f"Expected 3D or 4D plane tensor, got shape {tuple(planes.shape)}")

    batch, channels, height, width = planes.shape
    num_points = height * width
    flat = planes.permute(0, 2, 3, 1).reshape(batch, num_points, channels)
    perm = plane_to_sphere.to(device=planes.device)
    sphere_to_plane = torch.empty_like(perm)
    sphere_to_plane[perm] = torch.arange(num_points, device=planes.device, dtype=perm.dtype)
    return flat.index_select(1, sphere_to_plane)


def _normalize_quaternions_with_identity_fallback(rotations_raw: torch.Tensor) -> torch.Tensor:
    quat_norms = rotations_raw.norm(dim=-1, keepdim=True)
    quats = rotations_raw / quat_norms.clamp_min(RENDER_QUAT_EPS)
    identity_quat = torch.zeros_like(quats)
    identity_quat[..., 0] = 1.0
    return torch.where(quat_norms > RENDER_QUAT_EPS, quats, identity_quat)


def _point_clouds_to_gsplat_inputs(
    point_clouds: torch.Tensor,
    *,
    dc_only: bool = False,
    detach_input: bool = False,
    semantic_values: bool = False,
) -> dict[str, torch.Tensor | int]:
    """Convert batched point clouds to gsplat-native tensors."""
    if detach_input:
        point_clouds = point_clouds.detach()
    if point_clouds.ndim == 2:
        point_clouds = point_clouds.unsqueeze(0)
    if point_clouds.ndim != 3:
        raise ValueError(f"Expected point clouds with shape (B, N, D), got {tuple(point_clouds.shape)}")

    feature_dim = point_clouds.shape[-1]
    expected_dim = 14 if dc_only else 59
    if dc_only and feature_dim != expected_dim:
        raise ValueError(f"Expected {expected_dim} DC-only features, got {feature_dim}")
    if not dc_only and feature_dim < expected_dim:
        raise ValueError(f"Expected at least {expected_dim} full features, got {feature_dim}")

    pc = point_clouds[..., :expected_dim].to(dtype=torch.float32).contiguous()
    means = pc[..., 0:3]

    if dc_only:
        colors = pc[..., 4:7].unsqueeze(-2).contiguous()  # (B, N, 1, 3)
        scales_raw = pc[..., 7:10]
        rotations_raw = pc[..., 10:14]
        sh_degree = 0
    else:
        features = pc[..., 4:52]
        sh_coeffs = features.reshape(*pc.shape[:-1], 3, 16)
        features_dc = sh_coeffs[..., 0].unsqueeze(-2)
        features_rest = sh_coeffs[..., 1:].transpose(-1, -2)
        colors = torch.cat((features_dc, features_rest), dim=-2).contiguous()  # (B, N, 16, 3)
        scales_raw = pc[..., 52:55]
        rotations_raw = pc[..., 55:59]
        sh_degree = 3

    if semantic_values:
        opacities = pc[..., 3].clamp(0.0, 1.0)
        scales = scales_raw.clamp_min(torch.finfo(pc.dtype).tiny)
    else:
        opacities = torch.sigmoid(pc[..., 3].clamp(RENDER_OPACITY_RAW_MIN, RENDER_OPACITY_RAW_MAX))
        scales = torch.exp(scales_raw.clamp(RENDER_SCALE_RAW_MIN, RENDER_SCALE_RAW_MAX))
    quats = _normalize_quaternions_with_identity_fallback(rotations_raw)

    return {
        "means": means,
        "quats": quats,
        "scales": scales,
        "opacities": opacities,
        "colors": colors,
        "sh_degree": sh_degree,
    }


def _render_gsplat_batch(
    renderer_module,
    gaussian_inputs: dict[str, torch.Tensor | int],
    camera_bundle: dict[str, Any],
    cam_indices: list[int],
    device: torch.device,
) -> torch.Tensor:
    """Render a batch of Gaussian sets across a batch of cameras with gsplat."""
    batch_size = gaussian_inputs["means"].shape[0]
    cam_idx = torch.tensor(cam_indices, device=device, dtype=torch.long)
    viewmats = camera_bundle["viewmats"].index_select(0, cam_idx)
    Ks = camera_bundle["Ks"].index_select(0, cam_idx)
    num_cam = int(viewmats.shape[0])

    viewmats = viewmats.unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous()
    Ks = Ks.unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous()
    backgrounds = torch.zeros((batch_size, num_cam, 3), dtype=torch.float32, device=device)

    renders, _, _ = renderer_module.rasterization(
        means=gaussian_inputs["means"],
        quats=gaussian_inputs["quats"],
        scales=gaussian_inputs["scales"],
        opacities=gaussian_inputs["opacities"],
        colors=gaussian_inputs["colors"],
        viewmats=viewmats,
        Ks=Ks,
        width=int(camera_bundle["width"]),
        height=int(camera_bundle["height"]),
        sh_degree=gaussian_inputs["sh_degree"],
        backgrounds=backgrounds,
        packed=False,
        render_mode="RGB",
    )
    return renders.permute(0, 1, 4, 2, 3).clamp(0.0, 1.0).contiguous()


def _denormalize_point_cloud(
    point_cloud: torch.Tensor,
    mean: Optional[torch.Tensor],
    std: Optional[torch.Tensor],
) -> torch.Tensor:
    if mean is None or std is None:
        return point_cloud
    mean_t = mean.to(device=point_cloud.device, dtype=point_cloud.dtype)
    std_t = std.to(device=point_cloud.device, dtype=point_cloud.dtype)
    expand_shape = (1,) * (point_cloud.ndim - 1) + (point_cloud.shape[-1],)
    return point_cloud * (std_t.view(expand_shape) + 1e-8) + mean_t.view(expand_shape)


def _constrain_denormalized_point_cloud_for_render(
    point_cloud: torch.Tensor,
    *,
    dc_only: bool = False,
) -> torch.Tensor:
    """Project denormalized predictions into canonical 3DGS parameter space for rendering."""
    if dc_only:
        scale_slice = slice(7, 10)
        rotation_slice = slice(10, 14)
    else:
        scale_slice = slice(52, 55)
        rotation_slice = slice(55, 59)

    # Keep this path fully out-of-place so autograd does not see view-based slices
    # being mutated after they have already participated in the computation graph.
    safe_point_cloud = torch.nan_to_num(point_cloud, nan=0.0, posinf=0.0, neginf=0.0)
    xyz = safe_point_cloud[..., :3]
    opacity = torch.sigmoid(safe_point_cloud[..., 3:4])
    middle = safe_point_cloud[..., 4:scale_slice.start]
    scales = torch.exp(
        safe_point_cloud[..., scale_slice].clamp(RENDER_SCALE_RAW_MIN, RENDER_SCALE_RAW_MAX)
    )
    rotations = _normalize_quaternions_with_identity_fallback(
        safe_point_cloud[..., rotation_slice]
    )
    tail = safe_point_cloud[..., rotation_slice.stop:]

    return torch.cat((xyz, opacity, middle, scales, rotations, tail), dim=-1)


def _compute_render_loss_for_batch(
    x0_pred: torch.Tensor,
    x_gt_full: torch.Tensor,
    plane_to_sphere: torch.Tensor,
    norm_mean_pred: Optional[torch.Tensor],
    norm_std_pred: Optional[torch.Tensor],
    norm_mean_full: Optional[torch.Tensor],
    norm_std_full: Optional[torch.Tensor],
    train_cameras: list,
    renderer_tuple: tuple,
    lpips_fn: Optional[nn.Module],
    num_cam: int,
    device: torch.device,
    dc_only: bool = False,
) -> tuple:
    """Compute differentiable 2D render losses over the full batch with gsplat."""
    batch_size = x_gt_full.shape[0]
    view_count = max(1, int(num_cam))
    total_cams = int(train_cameras["viewmats"].shape[0])
    cam_indices = random.sample(range(total_cams), min(view_count, total_cams))

    gt_pc_norm = _plane_to_point_cloud_batch(x_gt_full.float(), plane_to_sphere)
    gt_pc_raw = _denormalize_point_cloud(gt_pc_norm, norm_mean_full, norm_std_full)
    gt_gaussians = _point_clouds_to_gsplat_inputs(gt_pc_raw.to(device), dc_only=False, detach_input=True)

    pred_pc_norm = _plane_to_point_cloud_batch(x0_pred.float(), plane_to_sphere)
    pred_pc_raw = _denormalize_point_cloud(pred_pc_norm, norm_mean_pred, norm_std_pred)
    pred_pc_raw = _constrain_denormalized_point_cloud_for_render(
        pred_pc_raw,
        dc_only=dc_only,
    )
    pred_gaussians = _point_clouds_to_gsplat_inputs(
        pred_pc_raw.to(device),
        dc_only=dc_only,
        detach_input=False,
        semantic_values=True,
    )

    with torch.no_grad():
        target = _render_gsplat_batch(renderer_tuple, gt_gaussians, train_cameras, cam_indices, device)
    pred = _render_gsplat_batch(renderer_tuple, pred_gaussians, train_cameras, cam_indices, device)

    l1_loss = torch.mean(torch.abs(pred - target))
    lpips_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
    if lpips_fn is not None:
        pred_n = (pred * 2.0 - 1.0).reshape(batch_size * len(cam_indices), 3, pred.shape[-2], pred.shape[-1])
        target_n = (target * 2.0 - 1.0).reshape(batch_size * len(cam_indices), 3, target.shape[-2], target.shape[-1])
        lpips_loss = lpips_fn(pred_n, target_n).mean()

    return l1_loss, lpips_loss


def _save_training_render_preview(
    x0_pred: torch.Tensor,
    x_gt_full: torch.Tensor,
    plane_to_sphere: torch.Tensor,
    norm_mean_pred: Optional[torch.Tensor],
    norm_std_pred: Optional[torch.Tensor],
    norm_mean_full: Optional[torch.Tensor],
    norm_std_full: Optional[torch.Tensor],
    train_cameras: list,
    renderer_tuple: tuple,
    output_dir: str,
    epoch: int,
    step: int,
    timesteps: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    num_cam: int,
    dc_only: bool = False,
) -> None:
    """Save side-by-side GT/pred train-time renders for quick visual inspection."""
    bsz = x_gt_full.shape[0]
    sample_idx = random.randrange(max(1, bsz))
    view_count = max(1, int(num_cam))
    total_cams = int(train_cameras["viewmats"].shape[0])
    cam_indices = random.sample(range(total_cams), min(view_count, total_cams))
    rows = []

    with torch.no_grad():
        gt_pc_norm = _plane_to_point_cloud_batch(x_gt_full[sample_idx : sample_idx + 1].float(), plane_to_sphere)
        gt_pc_raw = _denormalize_point_cloud(gt_pc_norm, norm_mean_full, norm_std_full)
        gt_gaussians = _point_clouds_to_gsplat_inputs(gt_pc_raw.to(device), dc_only=False, detach_input=True)

        pred_pc_norm = _plane_to_point_cloud_batch(x0_pred[sample_idx : sample_idx + 1].float(), plane_to_sphere)
        pred_pc_raw = _denormalize_point_cloud(pred_pc_norm, norm_mean_pred, norm_std_pred)
        pred_pc_raw = _constrain_denormalized_point_cloud_for_render(
            pred_pc_raw,
            dc_only=dc_only,
        )
        pred_gaussians = _point_clouds_to_gsplat_inputs(
            pred_pc_raw.to(device),
            dc_only=dc_only,
            detach_input=True,
            semantic_values=True,
        )

        target_views = _render_gsplat_batch(renderer_tuple, gt_gaussians, train_cameras, cam_indices, device)[0]
        pred_views = _render_gsplat_batch(renderer_tuple, pred_gaussians, train_cameras, cam_indices, device)[0]

        for target, pred in zip(target_views, pred_views):
            
            target_np = (target.permute(1, 2, 0).clamp(0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8)
            pred_np = (pred.permute(1, 2, 0).clamp(0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8)
            column_separator = np.full((target_np.shape[0], 4, 3), 255, dtype=np.uint8)
            rows.append(np.concatenate([target_np, column_separator, pred_np], axis=1))

    if not rows:
        return
    if len(rows) == 1:
        preview = rows[0]
    else:
        row_separator = np.full((4, rows[0].shape[1], 3), 255, dtype=np.uint8)
        preview = np.concatenate(
            [piece for row in rows[:-1] for piece in (row, row_separator)] + [rows[-1]],
            axis=0,
        )

    preview_dir = os.path.join(output_dir, "dit_train_renders")
    os.makedirs(preview_dir, exist_ok=True)
    y_label = int(labels[sample_idx].item())
    timestep = int(timesteps[sample_idx].item())
    out_path = os.path.join(
        preview_dir,
        f"epoch_{epoch:03d}_step_{step:07d}_class{y_label:03d}_t{timestep:04d}.png",
    )
    latest_path = os.path.join(preview_dir, "latest.png")
    Image.fromarray(preview).save(out_path)
    Image.fromarray(preview).save(latest_path)
    logger.info(
        f"[train-render] saved: {out_path} (left=gt, right=pred, class={y_label}, t={timestep})"
    )


def _run_validation_render(
    model: nn.Module,
    plane_to_sphere: torch.Tensor,
    norm_mean: Optional[torch.Tensor],
    norm_std: Optional[torch.Tensor],
    train_cameras: list,
    renderer_tuple: tuple,
    output_dir: str,
    epoch: int,
    step: int,
    device: torch.device,
    in_channels: int,
    num_classes: int,
    dc_only: bool = False,
    predict_xstart: bool = False,
    val_sampling_steps: int = 250,
) -> None:
    """Generate a sample from pure noise via full DDPM sampling, render and save as PNG."""
    # Create a respaced diffusion for faster validation sampling
    val_diffusion = create_diffusion(
        timestep_respacing=str(val_sampling_steps),
        learn_sigma=False,
        predict_xstart=predict_xstart,
    )

    # Randomly sample a class label
    y_label = random.randrange(num_classes)
    y = torch.tensor([y_label], dtype=torch.long, device=device)

    # Full sampling from pure Gaussian noise
    shape = (1, in_channels, 128, 128)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        sample = val_diffusion.p_sample_loop(
            model,
            shape,
            clip_denoised=False,
            model_kwargs=dict(y=y),
            device=device,
        )
    if was_training:
        model.train()

    # Build GS inputs from generated sample.
    pred_pc = _plane_to_point_cloud_batch(sample.float(), plane_to_sphere)
    pred_pc_raw = _denormalize_point_cloud(pred_pc, norm_mean, norm_std)
    pred_pc_raw = _constrain_denormalized_point_cloud_for_render(
        pred_pc_raw,
        dc_only=dc_only,
    )
    pred_gaussians = _point_clouds_to_gsplat_inputs(
        pred_pc_raw.to(device),
        dc_only=dc_only,
        detach_input=True,
        semantic_values=True,
    )

    cam_indices = [random.randrange(int(train_cameras["viewmats"].shape[0]))]
    with torch.no_grad():
        pred_img = _render_gsplat_batch(renderer_tuple, pred_gaussians, train_cameras, cam_indices, device)[0, 0]

    # Convert to numpy HWC and save
    pred_np = pred_img.permute(1, 2, 0).clamp(0.0, 1.0).cpu().numpy()
    img_uint8 = (pred_np * 255.0).astype(np.uint8)

    val_dir = os.path.join(output_dir, "dit_validation")
    os.makedirs(val_dir, exist_ok=True)
    out_path = os.path.join(val_dir, f"epoch_{epoch:03d}_step_{step:07d}_class{y_label:03d}.png")
    Image.fromarray(img_uint8).save(out_path)
    logger.info(f"[validation] saved: {out_path} (class={y_label}, steps={val_sampling_steps})")



#################################################################################
#                             EMA Utilities                                     #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """Update EMA model parameters. `model` should be the unwrapped model."""
    ema_params = dict(ema_model.named_parameters())
    model_params = dict(model.named_parameters())
    for key in ema_params:
        ema_params[key].mul_(decay).add_(model_params[key].data, alpha=1 - decay)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


#################################################################################
#                             Training Loop                                     #
#################################################################################

def main(args):
    # ── Accelerator ──────────────────────────────────────────────────────
    accelerator = Accelerator(
        mixed_precision="no" if args.mixed_precision == "none" else args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with=None,
    )
    device = accelerator.device
    is_main = accelerator.is_main_process

    if is_main:
        logger.info(f"Accelerator: num_processes={accelerator.num_processes}, "
                     f"mixed_precision={accelerator.mixed_precision}, device={device}")

    # Seed for reproducibility (accelerate handles per-process offset)
    set_seed(args.seed)

    # Create results directory (main process only to avoid race)
    if is_main:
        os.makedirs(args.results_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Load class map
    if is_main:
        logger.info(f"Loading class map from {args.class_map}")
    with open(args.class_map, 'r') as f:
        class_map = json.load(f)
    num_classes = max(v for v in class_map.values() if v >= 0) + 1
    if is_main:
        logger.info(f"Number of classes: {num_classes}")

    # Create base dataset
    if is_main:
        logger.info("Creating base dataset...")
    base_dataset = Standard3DGenDataset(
        obj_list=[args.obj_list],
        gs_path=args.gs_path,
        caption_path=None,
        mean_file=args.mean_file,
        std_file=args.std_file,
    )

    # Resolve feature indices for sh_degree0_only
    if args.sh_degree0_only:
        feature_indices = torch.tensor(DC_ONLY_FEATURE_INDICES, dtype=torch.long)
        in_channels = len(DC_ONLY_FEATURE_INDICES)
        if is_main:
            logger.info(f"sh_degree0_only: selecting {in_channels} features from {FULL_3DGS_FEATURE_DIM}")
    else:
        feature_indices = None
        in_channels = FULL_3DGS_FEATURE_DIM

    # Load sphere2plane permutation
    num_points = base_dataset[0]['point_cloud'].shape[0]
    plane_to_sphere = load_sphere2plane(args.sphere2plane_path, num_points)
    if is_main:
        logger.info(f"Loaded sphere2plane permutation: {num_points} points")

    # Render-related features
    render_loss_requested = args.render_loss_weight > 0.0 or args.lpips_loss_weight > 0.0
    use_render_loss = render_loss_requested and args.enable_render_loss_after >= 0
    enable_train_render_log = args.train_render_log_every > 0
    if render_loss_requested and not use_render_loss and is_main:
        logger.info("[render-loss] disabled because enable_render_loss_after < 0")

    # Wrap with class-conditional dataset
    dataset = Class3DGenDataset(
        base_dataset, class_map,
        plane_to_sphere=plane_to_sphere,
        feature_indices=feature_indices,
        return_full_for_render=((use_render_loss or enable_train_render_log) and feature_indices is not None),
        preload_to_cpu=args.preload_to_cpu,
        lazy_cache_to_cpu=args.lazy_cache_to_cpu,
        cache_dtype=(torch.bfloat16 if args.mixed_precision == 'bf16' else torch.float32),
        preload_max_samples=args.preload_max_samples,
        preload_workers=args.preload_workers,
    )

    # DataLoader — accelerate will inject DistributedSampler automatically
    loader_kwargs = dict(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = args.persistent_workers
        if args.prefetch_factor > 0:
            loader_kwargs["prefetch_factor"] = args.prefetch_factor
    loader = DataLoader(**loader_kwargs)
    if is_main:
        logger.info(f"Dataset size: {len(dataset)}, Per-GPU batch size: {args.batch_size}")

    # Create model
    if is_main:
        logger.info(f"Creating model: {args.model}")
    model = DiT_3DGS_models[args.model](
        input_size=128,
        in_channels=in_channels,
        num_classes=num_classes,
        learn_sigma=False,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    if is_main:
        logger.info(f"Model parameters: {total_params:,} ({total_params/1e6:.1f}M)")

    # Create EMA model (lives on device, not wrapped by accelerate)
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    ema.eval()

    # Create diffusion
    diffusion = create_diffusion(
        timestep_respacing="",  # use all 1000 timesteps for training
        learn_sigma=False,
        predict_xstart=args.predict_xstart,
    )
    if is_main:
        logger.info(f"Diffusion timesteps: {diffusion.num_timesteps}, "
                     f"predict={'x0' if args.predict_xstart else 'eps'}")

    # Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)

    # ── Let accelerate prepare model, optimizer, dataloader ──────────────
    model, opt, loader = accelerator.prepare(model, opt, loader)

    # Initialize EMA from the (now device-placed) unwrapped model
    update_ema(ema, accelerator.unwrap_model(model), decay=0)

    # Load normalization stats for render loss denormalization
    norm_mean = None
    norm_std = None
    norm_mean_full = None
    norm_std_full = None
    if args.mean_file and args.std_file:
        norm_mean_full = torch.load(args.mean_file, weights_only=True).float().cpu()
        norm_std_full = torch.load(args.std_file, weights_only=True).float().cpu()
        if feature_indices is not None:
            norm_mean = norm_mean_full[feature_indices]
            norm_std = norm_std_full[feature_indices]
        else:
            norm_mean = norm_mean_full
            norm_std = norm_std_full

    # Render / validation setup (per-process; gsplat renderer is local)
    renderer_for_train = None
    lpips_fn_for_train = None
    train_cameras = None
    enable_val = args.val_every > 0
    needs_renderer = use_render_loss or enable_val or enable_train_render_log
    if needs_renderer and device.type != "cuda":
        if is_main:
            logger.info("[renderer] disabled: CUDA required for gsplat")
        needs_renderer = False
        enable_val = False
    if needs_renderer:
        ref_cameras = _load_reference_cameras(args.ref_camera_tar)
        if is_main:
            logger.info(f"Loaded {len(ref_cameras)} reference cameras from {args.ref_camera_tar}")
        renderer_probe = _try_import_renderer()
        if not isinstance(renderer_probe, Exception):
            renderer_for_train = renderer_probe
            train_cameras = _prepare_train_cameras(ref_cameras, args.train_render_size, device)
            if use_render_loss and args.lpips_loss_weight > 0.0:
                lpips_probe = _try_import_lpips()
                if isinstance(lpips_probe, Exception):
                    if is_main:
                        logger.warning(f"[render-loss] LPIPS disabled: import failed: {lpips_probe}")
                else:
                    lpips_fn_for_train = lpips_probe.LPIPS(net=args.lpips_net).to(device).eval()
                    for p in lpips_fn_for_train.parameters():
                        p.requires_grad_(False)
        else:
            if is_main:
                logger.warning(f"[renderer] disabled: import failed: {renderer_probe}")
            needs_renderer = False
            enable_val = False

    # Resume from checkpoint if provided
    start_step = 0
    start_epoch = 0
    if args.resume:
        if is_main:
            logger.info(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        accelerator.unwrap_model(model).load_state_dict(ckpt['model'])
        ema.load_state_dict(ckpt['ema'])
        ema.eval()
        opt.load_state_dict(ckpt['opt'])
        start_step = ckpt['step']
        start_epoch = start_step // len(loader)
        if is_main:
            logger.info(f"Resumed at step {start_step}")

    # Training
    model.train()
    step = start_step
    log_loss = 0.0
    log_render_l1 = 0.0
    log_render_lpips = 0.0
    log_steps = 0
    start_time = time.time()

    if is_main:
        logger.info(f"Starting training from epoch {start_epoch}, step {start_step}...")

    dc_only = args.sh_degree0_only
    has_full_for_render = getattr(dataset, 'return_full_for_render', False)

    for epoch in range(start_epoch, args.epochs):
        for batch in loader:
            if has_full_for_render:
                x, y, x_full, hash_keys = batch
            else:
                x, y, hash_keys = batch
                x_full = None
            y = y.long()  # (B,)
            hash_keys = list(hash_keys)

            with accelerator.accumulate(model):
                # Sample random timesteps
                t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
                noise = torch.randn_like(x)

                # Forward pass (accelerate handles autocast)
                loss_dict = diffusion.training_losses(model, x, t, model_kwargs=dict(y=y), noise=noise)
                sample_losses = loss_dict["loss"]
                mse_loss = sample_losses.mean()
                if not torch.isfinite(mse_loss):
                    _debug_nonfinite_mse(
                        args=args,
                        diffusion=diffusion,
                        model=model,
                        x=x,
                        y=y,
                        x_full=x_full,
                        t=t,
                        noise=noise,
                        sample_losses=sample_losses,
                        step=step,
                        epoch=epoch,
                        hash_keys=hash_keys,
                        is_main=is_main,
                    )

                # Render loss (computed in fp32 outside autocast for GS renderer compatibility)
                render_l1_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
                render_lpips_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
                should_log_train_render = (
                    is_main
                    and args.train_render_log_every > 0
                    and renderer_for_train is not None
                    and train_cameras is not None
                    and step % args.train_render_log_every == 0
                )
                should_compute_render = (
                    renderer_for_train is not None
                    and train_cameras is not None
                    and use_render_loss
                    and step >= args.enable_render_loss_after
                )
                x0_pred = None
                x_gt_for_render = x_full if x_full is not None else x
                if should_compute_render:
                    noise_for_render = torch.randn_like(x)
                    x_t = diffusion.q_sample(x, t, noise=noise_for_render)
                    model_out = model(x_t, t, y)
                    if args.predict_xstart:
                        x0_pred = model_out.float()
                    else:
                        x0_pred = diffusion._predict_xstart_from_eps(x_t.float(), t, model_out.float())

                if use_render_loss and x0_pred is not None:
                    render_l1_loss, render_lpips_loss = _compute_render_loss_for_batch(
                        x0_pred=x0_pred,
                        x_gt_full=x_gt_for_render,
                        plane_to_sphere=plane_to_sphere,
                        norm_mean_pred=norm_mean,
                        norm_std_pred=norm_std,
                        norm_mean_full=norm_mean_full if x_full is not None else norm_mean,
                        norm_std_full=norm_std_full if x_full is not None else norm_std,
                        train_cameras=train_cameras,
                        renderer_tuple=renderer_for_train,
                        lpips_fn=lpips_fn_for_train,
                        num_cam=args.render_loss_num_cam,
                        device=device,
                        dc_only=dc_only,
                    )

                if should_log_train_render:
                    preview_idx = random.randrange(max(1, x.shape[0]))
                    preview_slice = slice(preview_idx, preview_idx + 1)
                    preview_x_gt = x_gt_for_render[preview_slice]
                    preview_t = t[preview_slice]
                    preview_y = y[preview_slice]

                    if x0_pred is not None:
                        preview_x0_pred = x0_pred.detach()[preview_slice]
                    else:
                        preview_x = x[preview_slice]
                        noise_for_preview = torch.randn_like(preview_x)
                        x_t_preview = diffusion.q_sample(preview_x, preview_t, noise=noise_for_preview)
                        with torch.no_grad():
                            model_out_preview = model(x_t_preview, preview_t, preview_y)
                            if args.predict_xstart:
                                preview_x0_pred = model_out_preview.float()
                            else:
                                preview_x0_pred = diffusion._predict_xstart_from_eps(
                                    x_t_preview.float(),
                                    preview_t,
                                    model_out_preview.float(),
                                )

                    _save_training_render_preview(
                        x0_pred=preview_x0_pred,
                        x_gt_full=preview_x_gt,
                        plane_to_sphere=plane_to_sphere,
                        norm_mean_pred=norm_mean,
                        norm_std_pred=norm_std,
                        norm_mean_full=norm_mean_full if x_full is not None else norm_mean,
                        norm_std_full=norm_std_full if x_full is not None else norm_std,
                        train_cameras=train_cameras,
                        renderer_tuple=renderer_for_train,
                        output_dir=args.results_dir,
                        epoch=epoch,
                        step=step,
                        timesteps=preview_t,
                        labels=preview_y,
                        device=device,
                        num_cam=args.train_render_log_num_cam,
                        dc_only=dc_only,
                    )

                total_loss = (
                    mse_loss
                    + float(args.render_loss_weight) * render_l1_loss
                    + float(args.lpips_loss_weight) * render_lpips_loss
                )

                # Backward pass (accelerate handles scaling + sync)
                accelerator.backward(total_loss)
                if args.max_grad_norm > 0.0:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                opt.step()
                opt.zero_grad()

            # Update EMA (on unwrapped model, every step regardless of accumulation)
            update_ema(ema, accelerator.unwrap_model(model), decay=args.ema_decay)

            # Logging
            log_loss += mse_loss.item()
            log_render_l1 += render_l1_loss.item()
            log_render_lpips += render_lpips_loss.item()
            log_steps += 1
            step += 1

            if step % args.log_every == 0 and is_main:
                avg_loss = log_loss / log_steps
                elapsed = time.time() - start_time
                steps_per_sec = log_steps / elapsed
                msg = (
                    f"Step {step:>7d} | Epoch {epoch:>3d} | "
                    f"MSE: {avg_loss:.4f} | "
                    f"Steps/sec: {steps_per_sec:.2f}"
                )
                if use_render_loss:
                    avg_rl1 = log_render_l1 / log_steps
                    avg_rlpips = log_render_lpips / log_steps
                    msg += f" | Render_L1: {avg_rl1:.4f} | Render_LPIPS: {avg_rlpips:.4f}"
                logger.info(msg)
                log_loss = 0.0
                log_render_l1 = 0.0
                log_render_lpips = 0.0
                log_steps = 0
                start_time = time.time()

            # Save checkpoint (main process only)
            if step % args.ckpt_every == 0 and is_main:
                ckpt_path = os.path.join(args.results_dir, f"{step:07d}.pt")
                torch.save({
                    'model': accelerator.unwrap_model(model).state_dict(),
                    'ema': ema.state_dict(),
                    'opt': opt.state_dict(),
                    'args': vars(args),
                    'step': step,
                }, ckpt_path)
                logger.info(f"Saved checkpoint to {ckpt_path}")

            # Validation render (main process only)
            if enable_val and step % args.val_every == 0 and is_main:
                _run_validation_render(
                    model=ema,
                    plane_to_sphere=plane_to_sphere,
                    norm_mean=norm_mean,
                    norm_std=norm_std,
                    train_cameras=train_cameras,
                    renderer_tuple=renderer_for_train,
                    output_dir=args.results_dir,
                    epoch=epoch,
                    step=step,
                    device=device,
                    in_channels=in_channels,
                    num_classes=num_classes,
                    dc_only=dc_only,
                    predict_xstart=args.predict_xstart,
                    val_sampling_steps=args.val_sampling_steps,
                )

    # Save final checkpoint
    if is_main:
        ckpt_path = os.path.join(args.results_dir, f"{step:07d}.pt")
        torch.save({
            'model': accelerator.unwrap_model(model).state_dict(),
            'ema': ema.state_dict(),
            'opt': opt.state_dict(),
            'args': vars(args),
            'step': step,
        }, ckpt_path)
        logger.info(f"Training complete. Final checkpoint: {ckpt_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DiT for 3DGS generation')

    # Model
    parser.add_argument('--model', type=str, default='DiT-B/8',
                        choices=list(DiT_3DGS_models.keys()))
    parser.add_argument('--predict_xstart', action=argparse.BooleanOptionalAction, default=False,
                        help='Predict x0 directly instead of epsilon')

    # Data
    parser.add_argument('--obj_list', type=str, required=True,
                        help='Path to obj_list JSON file')
    parser.add_argument('--gs_path', type=str, required=True,
                        help='Path to 3DGS data directory')
    parser.add_argument('--mean_file', type=str, default=None,
                        help='Path to normalization mean file')
    parser.add_argument('--std_file', type=str, default=None,
                        help='Path to normalization std file')
    parser.add_argument('--class_map', type=str, default='object_labels/object_to_class.json',
                        help='Path to object-to-class mapping JSON')
    parser.add_argument('--sphere2plane_path', type=str, default='data/sphere2plane.npy',
                        help='Path to sphere2plane.npy permutation file')
    parser.add_argument('--sh_degree0_only', action=argparse.BooleanOptionalAction, default=False,
                        help='Keep only SH degree-0 / DC coefficients, reducing from 59 to 14 channels')

    # Render loss
    parser.add_argument('--render_loss_weight', type=float, default=0.0,
                        help='Weight for render L1 photometric loss term')
    parser.add_argument('--lpips_loss_weight', type=float, default=0.0,
                        help='Weight for render LPIPS photometric loss term')
    parser.add_argument('--lpips_net', type=str, default='vgg', choices=('vgg', 'alex', 'squeeze'),
                        help='LPIPS backbone')
    parser.add_argument('--render_loss_num_cam', type=int, default=1,
                        help='Number of cameras to randomly sample per render loss step')
    parser.add_argument('--train_render_size', type=int, default=128,
                        help='Train-time rendering resolution for render loss')
    parser.add_argument('--ref_camera_tar', type=str, default='/home/tiangexiang/gen3d/ref_camera.tar.gz',
                        help='Path to reference camera tar.gz for render loss')
    parser.add_argument('--enable_render_loss_after', type=int, default=0,
                        help='Number of training steps before enabling render loss (-1 disables render loss)')
    parser.add_argument('--train_render_log_every', type=int, default=0,
                        help='Save side-by-side train-time 2D render previews every N steps (0 = disabled)')
    parser.add_argument('--train_render_log_num_cam', type=int, default=2,
                        help='Number of camera views per train-time render preview')

    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--ema_decay', type=float, default=0.9999)
    parser.add_argument('--mixed_precision', type=str, default='fp16',
                        choices=['fp16', 'bf16', 'none'])
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Number of gradient accumulation steps')
    parser.add_argument('--gradient_checkpointing', action=argparse.BooleanOptionalAction, default=True,
                        help='Enable gradient checkpointing to save memory (reduces speed). '
                             'Disable with --no-gradient_checkpointing to use more memory but train faster.')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Max gradient norm for clipping (0 = disabled)')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--preload_to_cpu', action=argparse.BooleanOptionalAction, default=False,
                        help='Preload the transformed class-conditioned training dataset into a shared RAM cache in /dev/shm at startup. '
                             'All local GPU processes attach to the same in-memory cache; this does not fall back to disk.')
    parser.add_argument('--lazy_cache_to_cpu', action=argparse.BooleanOptionalAction, default=False,
                        help='Cache samples into the shared /dev/shm CPU cache on first access so training speeds up progressively instead of paying the full preload cost up front.')
    parser.add_argument('--preload_max_samples', type=int, default=0,
                        help='Cap eager CPU preloading to the first N samples (0 = preload the full dataset). '
                             'When used with --preload_to_cpu, training is restricted to that cached subset.')
    parser.add_argument('--preload_workers', type=int, default=0,
                        help='Worker processes used to build the shared preload cache (0 = auto, uses all available CPU workers).')
    parser.add_argument('--persistent_workers', action=argparse.BooleanOptionalAction, default=True,
                        help='Keep DataLoader worker processes alive across epochs when num_workers > 0.')
    parser.add_argument('--prefetch_factor', type=int, default=2,
                        help='Number of batches each DataLoader worker prefetches ahead when num_workers > 0. '
                             'Set <= 0 to disable the explicit override.')
    parser.add_argument('--seed', type=int, default=0)

    # Logging / Checkpoints / Validation
    parser.add_argument('--log_every', type=int, default=100)
    parser.add_argument('--ckpt_every', type=int, default=10000)
    parser.add_argument('--val_every', type=int, default=0,
                        help='Steps between validation renders (0 = disabled)')
    parser.add_argument('--val_sampling_steps', type=int, default=250,
                        help='Number of diffusion steps for validation sampling')
    parser.add_argument('--results_dir', type=str, default='output/dit_results')

    # Resume
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    args = parser.parse_args()
    main(args)
