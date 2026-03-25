"""
Training script for a class-conditional diffusers UNet on GaussianVerse 3DGS plane tensors.

The overall training structure follows the existing DiT GaussianVerse trainer:
- Standard3DGenDataset + Class3DGenDataset
- OpenAI diffusion utilities from dit.diffusion
- accelerate for distributed / mixed precision training
- EMA checkpoints and optional render-based diagnostics
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import random
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

import diffusers
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

GS_ROOT = os.path.join(REPO_ROOT, "submodules", "gaussian-splatting")
if GS_ROOT not in sys.path:
    sys.path.insert(0, GS_ROOT)

from dataloaders.class_3dgen_loader import (  # noqa: E402
    Class3DGenDataset,
    DC_ONLY_FEATURE_INDICES,
    FULL_3DGS_FEATURE_DIM,
    load_sphere2plane,
)
from dataloaders.standard_3dgen_loader import Standard3DGenDataset  # noqa: E402
from dit.diffusion import create_diffusion  # noqa: E402
from unet.models import GAUSSIANVERSE_UNET_PRESETS, build_gaussianverse_unet  # noqa: E402
from unet.sampling import sample_with_dpm  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _tensor_debug_summary(tensor: torch.Tensor) -> dict[str, Any]:
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
        summary.update(
            {
                "min": float(finite_vals.min().item()),
                "max": float(finite_vals.max().item()),
                "mean": float(finite_vals.mean().item()),
            }
        )
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
        bad_param_summaries.append(
            {
                "name": name,
                "shape": tuple(param.shape),
                "nonfinite": bad_count,
            }
        )
        if len(bad_param_summaries) >= 16:
            break

    per_sample_debug = []
    for pos in bad_positions[:4]:
        per_sample_debug.append(
            {
                "batch_pos": int(pos),
                "hash_key": hash_keys[pos],
                "label": int(y[pos].detach().cpu().item()),
                "loss": float(sample_losses[pos].detach().float().cpu().item()),
                "x": _tensor_debug_summary(x[pos]),
                "x_t": _tensor_debug_summary(x_t_debug[pos]),
                "model_output": _tensor_debug_summary(model_out_debug[pos]),
                "target": _tensor_debug_summary(target_debug[pos]),
            }
        )

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
    raise FloatingPointError(f"Non-finite diffusion MSE detected at step {step}; debug dump saved to {debug_path}")


@torch.no_grad()
def update_ema(ema_model, model, decay: float = 0.9999) -> None:
    ema_params = dict(ema_model.named_parameters())
    model_params = dict(model.named_parameters())
    for key in ema_params:
        ema_params[key].mul_(decay).add_(model_params[key].data, alpha=1 - decay)


def requires_grad(model, flag: bool = True) -> None:
    for param in model.parameters():
        param.requires_grad_(flag)


def _load_render_utils():
    try:
        return importlib.import_module("dit.train_gsplat")
    except Exception as exc:  # pragma: no cover - runtime dependency probe
        return exc


class IterExponential:
    """Iteration-wise exponential LR scheduler with linear warmup (from Marigold)."""

    def __init__(self, total_iter_length: int, final_ratio: float, warmup_steps: int = 0) -> None:
        self.total_length = total_iter_length
        self.effective_length = total_iter_length - warmup_steps
        self.final_ratio = final_ratio
        self.warmup_steps = warmup_steps

    def __call__(self, n_iter: int) -> float:
        if n_iter < self.warmup_steps:
            return 1.0 * n_iter / self.warmup_steps
        if n_iter >= self.total_length:
            return self.final_ratio
        actual_iter = n_iter - self.warmup_steps
        return np.exp(actual_iter / self.effective_length * np.log(self.final_ratio))


def _run_validation_render(
    *,
    render_utils,
    model: torch.nn.Module,
    plane_to_sphere: torch.Tensor,
    norm_mean: Optional[torch.Tensor],
    norm_std: Optional[torch.Tensor],
    train_cameras: dict,
    renderer_tuple,
    output_dir: str,
    epoch: int,
    step: int,
    device: torch.device,
    in_channels: int,
    num_classes: int,
    dc_only: bool = False,
    predict_xstart: bool = False,
    val_sampling_steps: int = 250,
    val_sampler: str = "dpm",
    dpm_solver_order: int = 2,
    dpm_algorithm_type: str = "dpmsolver++",
    dpm_solver_type: str = "midpoint",
    dpm_timestep_spacing: str = "trailing",
    dpm_use_karras_sigmas: bool = False,
) -> None:
    y_label = random.randrange(num_classes)
    y = torch.tensor([y_label], dtype=torch.long, device=device)

    shape = (1, in_channels, 128, 128)
    if val_sampler == "dpm":
        sample = sample_with_dpm(
            model=model,
            shape=shape,
            class_labels=y,
            num_inference_steps=val_sampling_steps,
            device=device,
            predict_xstart=predict_xstart,
            solver_order=dpm_solver_order,
            algorithm_type=dpm_algorithm_type,
            solver_type=dpm_solver_type,
            timestep_spacing=dpm_timestep_spacing,
            use_karras_sigmas=dpm_use_karras_sigmas,
        )
    else:
        val_diffusion = create_diffusion(
            timestep_respacing=str(val_sampling_steps),
            learn_sigma=False,
            predict_xstart=predict_xstart,
        )
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

    pred_pc = render_utils._plane_to_point_cloud_batch(sample.float(), plane_to_sphere)
    pred_pc_raw = render_utils._denormalize_point_cloud(pred_pc, norm_mean, norm_std)
    pred_pc_raw = render_utils._constrain_denormalized_point_cloud_for_render(
        pred_pc_raw,
        dc_only=dc_only,
    )
    pred_gaussians = render_utils._point_clouds_to_gsplat_inputs(
        pred_pc_raw.to(device),
        dc_only=dc_only,
        detach_input=True,
        semantic_values=True,
    )

    cam_indices = [random.randrange(int(train_cameras["viewmats"].shape[0]))]
    with torch.no_grad():
        pred_img = render_utils._render_gsplat_batch(
            renderer_tuple,
            pred_gaussians,
            train_cameras,
            cam_indices,
            device,
        )[0, 0]

    pred_np = pred_img.permute(1, 2, 0).clamp(0.0, 1.0).cpu().numpy()
    img_uint8 = (pred_np * 255.0).astype("uint8")

    val_dir = os.path.join(output_dir, "unet_validation")
    os.makedirs(val_dir, exist_ok=True)
    out_path = os.path.join(val_dir, f"epoch_{epoch:03d}_step_{step:07d}_class{y_label:03d}.png")
    render_utils.Image.fromarray(img_uint8).save(out_path)
    logger.info("[validation] saved: %s (class=%d, steps=%d)", out_path, y_label, val_sampling_steps)


def main(args) -> None:
    accelerator = Accelerator(
        mixed_precision="no" if args.mixed_precision == "none" else args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with=None,
    )
    device = accelerator.device
    is_main = accelerator.is_main_process

    if is_main:
        logger.info(
            "Accelerator: num_processes=%d, mixed_precision=%s, device=%s",
            accelerator.num_processes,
            accelerator.mixed_precision,
            device,
        )
        logger.info("Diffusers version: %s", diffusers.__version__)
        logger.info("Validation sampler: %s", args.val_sampler)

    set_seed(args.seed)

    if is_main:
        os.makedirs(args.results_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    if is_main:
        logger.info("Loading class map from %s", args.class_map)
    with open(args.class_map, "r", encoding="utf-8") as handle:
        class_map = json.load(handle)
    num_classes = max(value for value in class_map.values() if value >= 0) + 1
    if is_main:
        logger.info("Number of classes: %d", num_classes)

    if is_main:
        logger.info("Creating base dataset...")
    base_dataset = Standard3DGenDataset(
        obj_list=[args.obj_list],
        gs_path=args.gs_path,
        caption_path=None,
        mean_file=args.mean_file,
        std_file=args.std_file,
    )

    if args.sh_degree0_only:
        feature_indices = torch.tensor(DC_ONLY_FEATURE_INDICES, dtype=torch.long)
        in_channels = len(DC_ONLY_FEATURE_INDICES)
        if is_main:
            logger.info(
                "sh_degree0_only: selecting %d features from %d",
                in_channels,
                FULL_3DGS_FEATURE_DIM,
            )
    else:
        feature_indices = None
        in_channels = FULL_3DGS_FEATURE_DIM

    num_points = base_dataset[0]["point_cloud"].shape[0]
    plane_to_sphere = load_sphere2plane(args.sphere2plane_path, num_points)
    if is_main:
        logger.info("Loaded sphere2plane permutation: %d points", num_points)

    render_loss_requested = args.render_loss_weight > 0.0 or args.lpips_loss_weight > 0.0
    use_render_loss = render_loss_requested and args.enable_render_loss_after >= 0
    enable_train_render_log = args.train_render_log_every > 0
    if render_loss_requested and not use_render_loss and is_main:
        logger.info("[render-loss] disabled because enable_render_loss_after < 0")

    dataset = Class3DGenDataset(
        base_dataset,
        class_map,
        plane_to_sphere=plane_to_sphere,
        feature_indices=feature_indices,
        return_full_for_render=((use_render_loss or enable_train_render_log) and feature_indices is not None),
        preload_to_cpu=args.preload_to_cpu,
        lazy_cache_to_cpu=args.lazy_cache_to_cpu,
        cache_dtype=(torch.bfloat16 if args.mixed_precision == "bf16" else torch.float32),
        preload_max_samples=args.preload_max_samples,
        preload_workers=args.preload_workers,
    )

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
        logger.info("Dataset size: %d, Per-GPU batch size: %d", len(dataset), args.batch_size)

    if is_main:
        logger.info("Creating model: %s", args.model)
    model = build_gaussianverse_unet(
        args.model,
        sample_size=128,
        in_channels=in_channels,
        out_channels=in_channels,
        num_classes=num_classes,
        class_embedding_dim=args.class_embed_dim,
        norm_num_groups=args.norm_num_groups,
        dropout=args.dropout,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    total_params = sum(parameter.numel() for parameter in model.parameters())
    if is_main:
        logger.info("Model parameters: %s (%.1fM)", f"{total_params:,}", total_params / 1e6)

    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    ema.eval()

    diffusion = create_diffusion(
        timestep_respacing="",
        noise_schedule=args.noise_schedule,
        learn_sigma=False,
        predict_xstart=args.predict_xstart,
    )
    if is_main:
        logger.info(
            "Diffusion timesteps: %d, predict=%s, schedule=%s, snr_gamma=%s",
            diffusion.num_timesteps,
            "x0" if args.predict_xstart else "eps",
            args.noise_schedule,
            args.snr_gamma,
        )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_func = IterExponential(
        total_iter_length=args.lr_total_steps * accelerator.num_processes,
        final_ratio=args.lr_final_ratio,
        warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
    )
    lr_scheduler = LambdaLR(optimizer=opt, lr_lambda=lr_func)
    model, opt, loader, lr_scheduler = accelerator.prepare(model, opt, loader, lr_scheduler)
    update_ema(ema, accelerator.unwrap_model(model), decay=0.0)

    norm_mean = None
    norm_std = None
    norm_mean_full = None
    norm_std_full = None
    if args.mean_file and args.std_file:
        norm_mean_full = torch.load(args.mean_file, map_location="cpu", weights_only=False).float()
        norm_std_full = torch.load(args.std_file, map_location="cpu", weights_only=False).float()
        if feature_indices is not None:
            norm_mean = norm_mean_full[feature_indices]
            norm_std = norm_std_full[feature_indices]
        else:
            norm_mean = norm_mean_full
            norm_std = norm_std_full

    render_utils = None
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
        render_probe = _load_render_utils()
        if isinstance(render_probe, Exception):
            if is_main:
                logger.warning("[renderer] disabled: helper import failed: %s", render_probe)
            needs_renderer = False
            enable_val = False
        else:
            render_utils = render_probe
            ref_cameras = render_utils._load_reference_cameras(args.ref_camera_tar)
            if is_main:
                logger.info("Loaded %d reference cameras from %s", len(ref_cameras), args.ref_camera_tar)
            renderer_probe = render_utils._try_import_renderer()
            if not isinstance(renderer_probe, Exception):
                renderer_for_train = renderer_probe
                train_cameras = render_utils._prepare_train_cameras(
                    ref_cameras,
                    args.train_render_size,
                    device,
                )
                if use_render_loss and args.lpips_loss_weight > 0.0:
                    lpips_probe = render_utils._try_import_lpips()
                    if isinstance(lpips_probe, Exception):
                        if is_main:
                            logger.warning("[render-loss] LPIPS disabled: import failed: %s", lpips_probe)
                    else:
                        lpips_fn_for_train = lpips_probe.LPIPS(net=args.lpips_net).to(device).eval()
                        for parameter in lpips_fn_for_train.parameters():
                            parameter.requires_grad_(False)
            else:
                if is_main:
                    logger.warning("[renderer] disabled: import failed: %s", renderer_probe)
                needs_renderer = False
                enable_val = False

    start_step = 0
    start_epoch = 0
    if args.resume:
        if is_main:
            logger.info("Resuming from checkpoint: %s", args.resume)
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        accelerator.unwrap_model(model).load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        ema.eval()
        opt.load_state_dict(ckpt["opt"])
        if "lr_scheduler" in ckpt:
            lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        start_step = ckpt["step"]
        start_epoch = start_step // len(loader)
        if is_main:
            logger.info("Resumed at step %d", start_step)

    model.train()
    step = start_step
    log_loss = 0.0
    log_render_l1 = 0.0
    log_render_lpips = 0.0
    log_steps = 0
    start_time = time.time()

    if is_main:
        logger.info("Starting training from epoch %d, step %d...", start_epoch, start_step)

    dc_only = args.sh_degree0_only
    has_full_for_render = getattr(dataset, "return_full_for_render", False)

    for epoch in range(start_epoch, args.epochs):
        for batch in loader:
            if has_full_for_render:
                x, y, x_full, hash_keys = batch
            else:
                x, y, hash_keys = batch
                x_full = None
            y = y.long()
            hash_keys = list(hash_keys)

            with accelerator.accumulate(model):
                t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
                noise = torch.randn_like(x)

                loss_dict = diffusion.training_losses(model, x, t, model_kwargs=dict(y=y), noise=noise)
                sample_losses = loss_dict["loss"].float()
                if args.snr_gamma is not None:
                    alphas_cumprod = torch.tensor(diffusion.alphas_cumprod, device=device, dtype=torch.float32)
                    snr = alphas_cumprod[t] / (1.0 - alphas_cumprod[t])
                    # min-SNR weighting (arXiv:2303.09556, Section 3.4)
                    snr_weight = torch.minimum(snr, torch.full_like(snr, args.snr_gamma))
                    if args.predict_xstart:
                        # x0 prediction: natural weight = 1, apply min(SNR, gamma) directly
                        pass
                    else:
                        # epsilon prediction: natural weight = SNR, divide out
                        snr_weight = snr_weight / snr
                    # Scale by 4.0 to compensate for reduced average magnitude from SNR weighting
                    mse_loss = (sample_losses * snr_weight).mean() * 4.0
                else:
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

                render_l1_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
                render_lpips_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
                should_log_train_render = (
                    is_main
                    and args.train_render_log_every > 0
                    and render_utils is not None
                    and renderer_for_train is not None
                    and train_cameras is not None
                    and step % args.train_render_log_every == 0
                )
                should_compute_render = (
                    render_utils is not None
                    and renderer_for_train is not None
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

                if use_render_loss and x0_pred is not None and render_utils is not None:
                    render_l1_loss, render_lpips_loss = render_utils._compute_render_loss_for_batch(
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

                if should_log_train_render and render_utils is not None:
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

                    render_utils._save_training_render_preview(
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

                accelerator.backward(total_loss)
                if args.max_grad_norm > 0.0:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                opt.step()
                lr_scheduler.step()
                opt.zero_grad()

            update_ema(ema, accelerator.unwrap_model(model), decay=args.ema_decay)

            log_loss += mse_loss.item()
            log_render_l1 += render_l1_loss.item()
            log_render_lpips += render_lpips_loss.item()
            log_steps += 1
            step += 1

            if step % args.log_every == 0 and is_main:
                avg_loss = log_loss / log_steps
                elapsed = time.time() - start_time
                steps_per_sec = log_steps / elapsed
                current_lr = lr_scheduler.get_last_lr()[0]
                message = (
                    f"Step {step:>7d} | Epoch {epoch:>3d} | "
                    f"MSE: {avg_loss:.4f} | "
                    f"LR: {current_lr:.2e} | "
                    f"Steps/sec: {steps_per_sec:.2f}"
                )
                if use_render_loss:
                    avg_rl1 = log_render_l1 / log_steps
                    avg_rlpips = log_render_lpips / log_steps
                    message += f" | Render_L1: {avg_rl1:.4f} | Render_LPIPS: {avg_rlpips:.4f}"
                logger.info(message)
                log_loss = 0.0
                log_render_l1 = 0.0
                log_render_lpips = 0.0
                log_steps = 0
                start_time = time.time()

            if step % args.ckpt_every == 0 and is_main:
                ckpt_path = os.path.join(args.results_dir, f"{step:07d}.pt")
                torch.save(
                    {
                        "model": accelerator.unwrap_model(model).state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "args": vars(args),
                        "step": step,
                    },
                    ckpt_path,
                )
                logger.info("Saved checkpoint to %s", ckpt_path)

            if (
                enable_val
                and render_utils is not None
                and renderer_for_train is not None
                and train_cameras is not None
                and step % args.val_every == 0
                and is_main
            ):
                _run_validation_render(
                    render_utils=render_utils,
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
                    val_sampler=args.val_sampler,
                    dpm_solver_order=args.dpm_solver_order,
                    dpm_algorithm_type=args.dpm_algorithm_type,
                    dpm_solver_type=args.dpm_solver_type,
                    dpm_timestep_spacing=args.dpm_timestep_spacing,
                    dpm_use_karras_sigmas=args.dpm_use_karras_sigmas,
                )

    if is_main:
        ckpt_path = os.path.join(args.results_dir, f"{step:07d}.pt")
        torch.save(
            {
                "model": accelerator.unwrap_model(model).state_dict(),
                "ema": ema.state_dict(),
                "opt": opt.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "args": vars(args),
                "step": step,
            },
            ckpt_path,
        )
        logger.info("Training complete. Final checkpoint: %s", ckpt_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a diffusers UNet for GaussianVerse generation")

    parser.add_argument("--model", type=str, default="UNet-B", choices=sorted(GAUSSIANVERSE_UNET_PRESETS.keys()))
    parser.add_argument("--class_embed_dim", type=int, default=768, help="Class embedding dimension for UNet conditioning")
    parser.add_argument("--norm_num_groups", type=int, default=32, help="GroupNorm group count inside the UNet")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout used inside the diffusers UNet")
    parser.add_argument(
        "--predict_xstart",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Predict x0 directly instead of epsilon",
    )
    parser.add_argument(
        "--noise_schedule",
        type=str,
        default="squaredcos_cap_v2",
        choices=["linear", "squaredcos_cap_v2"],
        help="Beta schedule for diffusion noise",
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=5.0,
        help="Min-SNR gamma for loss reweighting (None to disable). Recommended: 5.0",
    )

    parser.add_argument("--obj_list", type=str, required=True, help="Path to obj_list JSON file")
    parser.add_argument("--gs_path", type=str, required=True, help="Path to 3DGS data directory")
    parser.add_argument("--mean_file", type=str, default=None, help="Path to normalization mean file")
    parser.add_argument("--std_file", type=str, default=None, help="Path to normalization std file")
    parser.add_argument(
        "--class_map",
        type=str,
        default="object_labels/object_to_class.json",
        help="Path to object-to-class mapping JSON",
    )
    parser.add_argument(
        "--sphere2plane_path",
        type=str,
        default="data/sphere2plane.npy",
        help="Path to sphere2plane.npy permutation file",
    )
    parser.add_argument(
        "--sh_degree0_only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep only SH degree-0 / DC coefficients, reducing from 59 to 14 channels",
    )

    parser.add_argument("--render_loss_weight", type=float, default=0.0, help="Weight for render L1 photometric loss term")
    parser.add_argument("--lpips_loss_weight", type=float, default=0.0, help="Weight for render LPIPS photometric loss term")
    parser.add_argument("--lpips_net", type=str, default="vgg", choices=("vgg", "alex", "squeeze"), help="LPIPS backbone")
    parser.add_argument("--render_loss_num_cam", type=int, default=1, help="Number of cameras to randomly sample per render loss step")
    parser.add_argument("--train_render_size", type=int, default=128, help="Train-time rendering resolution for render loss")
    parser.add_argument(
        "--ref_camera_tar",
        type=str,
        default="/home/tiangexiang/gen3d/ref_camera.tar.gz",
        help="Path to reference camera tar.gz for render loss",
    )
    parser.add_argument(
        "--enable_render_loss_after",
        type=int,
        default=0,
        help="Number of training steps before enabling render loss (-1 disables render loss)",
    )
    parser.add_argument(
        "--train_render_log_every",
        type=int,
        default=0,
        help="Save side-by-side train-time 2D render previews every N steps (0 = disabled)",
    )
    parser.add_argument("--train_render_log_num_cam", type=int, default=2, help="Number of camera views per train-time render preview")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="AdamW weight decay")
    parser.add_argument("--lr_warmup_steps", type=int, default=100, help="Linear LR warmup steps")
    parser.add_argument("--lr_total_steps", type=int, default=20000, help="Total steps for LR schedule (decay ends here)")
    parser.add_argument("--lr_final_ratio", type=float, default=0.01, help="Final LR as a fraction of peak LR")
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["fp16", "bf16", "none"])
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps")
    parser.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable diffusers UNet gradient checkpointing to reduce memory use",
    )
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping (0 = disabled)")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--preload_to_cpu",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Preload the transformed class-conditioned training dataset into a shared RAM cache in /dev/shm at startup",
    )
    parser.add_argument(
        "--lazy_cache_to_cpu",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Cache samples into the shared /dev/shm CPU cache on first access",
    )
    parser.add_argument(
        "--preload_max_samples",
        type=int,
        default=0,
        help="Cap eager CPU preloading to the first N samples (0 = preload the full dataset)",
    )
    parser.add_argument(
        "--preload_workers",
        type=int,
        default=0,
        help="Worker processes used to build the shared preload cache (0 = auto)",
    )
    parser.add_argument(
        "--persistent_workers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep DataLoader worker processes alive across epochs when num_workers > 0",
    )
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=2,
        help="Number of batches each DataLoader worker prefetches ahead when num_workers > 0",
    )
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--ckpt_every", type=int, default=10000)
    parser.add_argument("--val_every", type=int, default=0, help="Steps between validation renders (0 = disabled)")
    parser.add_argument(
        "--val_sampling_steps",
        type=int,
        default=40,
        help="Number of inference steps for validation sampling",
    )
    parser.add_argument(
        "--val_sampler",
        type=str,
        default="dpm",
        choices=["dpm", "ddpm"],
        help="Sampler used for validation generation",
    )
    parser.add_argument(
        "--dpm_solver_order",
        type=int,
        default=2,
        choices=[1, 2, 3],
        help="Diffusers DPM solver order",
    )
    parser.add_argument(
        "--dpm_algorithm_type",
        type=str,
        default="dpmsolver++",
        choices=["dpmsolver", "dpmsolver++", "sde-dpmsolver", "sde-dpmsolver++"],
        help="Diffusers DPM algorithm variant",
    )
    parser.add_argument(
        "--dpm_solver_type",
        type=str,
        default="midpoint",
        choices=["midpoint", "heun"],
        help="Diffusers DPM solver type",
    )
    parser.add_argument(
        "--dpm_timestep_spacing",
        type=str,
        default="trailing",
        choices=["linspace", "leading", "trailing"],
        help="Diffusers timestep spacing for DPM sampling",
    )
    parser.add_argument(
        "--dpm_use_karras_sigmas",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Karras sigmas in the diffusers DPM scheduler",
    )
    parser.add_argument("--results_dir", type=str, default="output/unet_results_gsplat")

    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
