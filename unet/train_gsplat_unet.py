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
from accelerate.utils import TorchDynamoPlugin, set_seed
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
)
from dataloaders.standard_3dgen_loader import Standard3DGenDataset  # noqa: E402
from dit.diffusion import create_diffusion  # noqa: E402
from unet.models import GAUSSIANVERSE_UNET_PRESETS, build_gaussianverse_unet  # noqa: E402
from unet.sampling import resolve_sampling_shape, sample_with_dpm  # noqa: E402


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
    ema_params = _normalize_compile_wrapped_keys(dict(ema_model.named_parameters()))
    model_params = _normalize_compile_wrapped_keys(dict(model.named_parameters()))
    missing_keys = sorted(set(ema_params) - set(model_params))
    if missing_keys:
        raise KeyError(
            "EMA/model parameter mismatch after compile-key normalization. "
            f"Missing keys in source model: {missing_keys[:8]}"
        )
    for key in ema_params:
        ema_params[key].mul_(decay).add_(model_params[key].data, alpha=1 - decay)


def requires_grad(model, flag: bool = True) -> None:
    for param in model.parameters():
        param.requires_grad_(flag)


def _load_render_utils():
    try:
        return importlib.import_module("utils.gsplat_render_util")
    except Exception as exc:  # pragma: no cover - runtime dependency probe
        return exc


def _normalize_compile_key(key: str) -> str:
    return key.removeprefix("_orig_mod.").replace("._orig_mod.", ".")


def _normalize_compile_wrapped_keys(state: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in state.items():
        normalized_key = _normalize_compile_key(key)
        if normalized_key in normalized and normalized[normalized_key] is not value:
            raise KeyError(
                "Compile-key normalization produced duplicate parameter/state names: "
                f"{normalized_key!r} from {key!r}"
            )
        normalized[normalized_key] = value
    return normalized


def _unwrap_training_model(accelerator: Accelerator, model: torch.nn.Module) -> torch.nn.Module:
    return accelerator.unwrap_model(model, keep_torch_compile=False)


def _model_state_dict_for_checkpoint(model: torch.nn.Module) -> dict[str, Any]:
    return _normalize_compile_wrapped_keys(model.state_dict())


def _capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _coerce_rng_byte_tensor(value: Any, *, name: str) -> torch.ByteTensor:
    if isinstance(value, (bytes, bytearray)):
        tensor = torch.tensor(list(value), dtype=torch.uint8)
    elif torch.is_tensor(value):
        tensor = value.detach()
    else:
        try:
            tensor = torch.as_tensor(value)
        except Exception as exc:
            raise TypeError(f"{name} must be convertible to a torch.ByteTensor, got {type(value)!r}") from exc

    tensor = tensor.to(device="cpu", dtype=torch.uint8).contiguous().view(-1)
    return tensor


def _restore_rng_state(state: Optional[dict[str, Any]], *, is_main: bool) -> None:
    if not state:
        return

    python_state = state.get("python")
    if python_state is not None:
        random.setstate(python_state)

    numpy_state = state.get("numpy")
    if numpy_state is not None:
        np.random.set_state(numpy_state)

    torch_cpu_state = state.get("torch_cpu")
    if torch_cpu_state is not None:
        torch.set_rng_state(_coerce_rng_byte_tensor(torch_cpu_state, name="torch_cpu"))

    torch_cuda_state = state.get("torch_cuda")
    if torch_cuda_state is not None:
        if torch.cuda.is_available():
            if torch.is_tensor(torch_cuda_state) or isinstance(torch_cuda_state, (bytes, bytearray)):
                cuda_states = [_coerce_rng_byte_tensor(torch_cuda_state, name="torch_cuda")]
            else:
                cuda_states = [
                    _coerce_rng_byte_tensor(item, name=f"torch_cuda[{index}]")
                    for index, item in enumerate(torch_cuda_state)
                ]
            torch.cuda.set_rng_state_all(cuda_states)
        elif is_main:
            logger.warning("Checkpoint contains CUDA RNG state but CUDA is unavailable; skipping CUDA RNG restore")


def _seed_dataloader_for_epoch(
    loader,
    *,
    epoch: int,
    base_seed: int,
    loader_generator: Optional[torch.Generator],
) -> None:
    epoch_seed = int(base_seed) + int(epoch)
    if loader_generator is not None:
        loader_generator.manual_seed(epoch_seed)

    candidates = [
        getattr(loader, "generator", None),
        getattr(loader, "sampler", None),
        getattr(loader, "batch_sampler", None),
    ]
    batch_sampler = getattr(loader, "batch_sampler", None)
    if batch_sampler is not None:
        candidates.append(getattr(batch_sampler, "sampler", None))

    seen: set[int] = set()
    for candidate in candidates:
        if candidate is None:
            continue
        candidate_id = id(candidate)
        if candidate_id in seen:
            continue
        seen.add(candidate_id)

        if isinstance(candidate, torch.Generator):
            candidate.manual_seed(epoch_seed)
            continue

        set_epoch = getattr(candidate, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(epoch)

        candidate_generator = getattr(candidate, "generator", None)
        if isinstance(candidate_generator, torch.Generator):
            candidate_generator.manual_seed(epoch_seed)


def _save_checkpoint(
    *,
    ckpt_path: str,
    raw_model: torch.nn.Module,
    ema: torch.nn.Module,
    opt: torch.optim.Optimizer,
    lr_scheduler,
    args,
    step: int,
    steps_per_epoch: int,
) -> None:
    torch.save(
        {
            "model": _model_state_dict_for_checkpoint(raw_model),
            "ema": ema.state_dict(),
            "opt": opt.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "args": vars(args),
            "step": step,
            "epoch": step // steps_per_epoch,
            "step_in_epoch": step % steps_per_epoch,
            "rng_state": _capture_rng_state(),
        },
        ckpt_path,
    )


def _build_dynamo_plugin(args) -> Optional[TorchDynamoPlugin]:
    compile_overrides_requested = any(
        value is not None
        for value in (
            args.compile,
            args.compile_backend,
            args.compile_mode,
            args.compile_fullgraph,
            args.compile_dynamic,
            args.compile_regional,
        )
    )
    if not compile_overrides_requested:
        return None

    if args.compile is False:
        conflicting_flags = []
        if args.compile_backend not in (None, "no"):
            conflicting_flags.append("--compile_backend")
        if args.compile_mode is not None:
            conflicting_flags.append("--compile_mode")
        if args.compile_fullgraph is not None:
            conflicting_flags.append("--compile_fullgraph/--no-compile_fullgraph")
        if args.compile_dynamic is not None:
            conflicting_flags.append("--compile_dynamic/--no-compile_dynamic")
        if args.compile_regional is not None:
            conflicting_flags.append("--compile_regional/--no-compile_regional")
        if conflicting_flags:
            raise ValueError(
                "--no-compile cannot be combined with compile configuration flags: "
                + ", ".join(conflicting_flags)
            )
        return TorchDynamoPlugin(backend="no")

    backend = args.compile_backend
    if backend is None and (
        args.compile is True
        or args.compile_mode is not None
        or args.compile_fullgraph is not None
        or args.compile_dynamic is not None
        or args.compile_regional is not None
    ):
        backend = os.environ.get("ACCELERATE_DYNAMO_BACKEND", "inductor")
        if backend.lower() == "no":
            backend = "inductor"

    plugin_kwargs: dict[str, Any] = {}
    if backend is not None:
        plugin_kwargs["backend"] = backend
    if args.compile_mode is not None:
        plugin_kwargs["mode"] = args.compile_mode
    if args.compile_fullgraph is not None:
        plugin_kwargs["fullgraph"] = args.compile_fullgraph
    if args.compile_dynamic is not None:
        plugin_kwargs["dynamic"] = args.compile_dynamic
    if args.compile_regional is not None:
        plugin_kwargs["use_regional_compilation"] = args.compile_regional
    return TorchDynamoPlugin(**plugin_kwargs)


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
    noise_schedule: str = "linear",
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

    shape = resolve_sampling_shape(model=model, batch_size=1, in_channels=in_channels)
    if val_sampler == "dpm":
        sample = sample_with_dpm(
            model=model,
            shape=shape,
            class_labels=y,
            num_inference_steps=val_sampling_steps,
            device=device,
            predict_xstart=predict_xstart,
            noise_schedule=noise_schedule,
            solver_order=dpm_solver_order,
            algorithm_type=dpm_algorithm_type,
            solver_type=dpm_solver_type,
            timestep_spacing=dpm_timestep_spacing,
            use_karras_sigmas=dpm_use_karras_sigmas,
        )
    else:
        val_diffusion = create_diffusion(
            timestep_respacing=str(val_sampling_steps),
            noise_schedule=noise_schedule,
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

    pred_pc = render_utils._plane_to_point_cloud_batch(sample.float())
    pred_pc_raw = render_utils._denormalize_point_cloud(pred_pc, norm_mean, norm_std)
    pred_gaussians = render_utils._point_clouds_to_gsplat_inputs(
        pred_pc_raw.to(device),
        dc_only=dc_only,
        detach_input=True,
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


def _parse_fixed_check_timesteps(spec: str, *, diffusion_steps: int) -> tuple[int, ...]:
    raw_items = [item.strip() for item in spec.split(",")]
    timesteps: list[int] = []
    for item in raw_items:
        if not item:
            continue
        timestep = int(item)
        if timestep < 0 or timestep >= diffusion_steps:
            raise ValueError(
                f"fixed_check_timesteps contains out-of-range timestep {timestep}; "
                f"expected values in [0, {diffusion_steps - 1}]"
            )
        if timestep not in timesteps:
            timesteps.append(timestep)
    if not timesteps:
        raise ValueError("fixed_check_timesteps must contain at least one timestep when fixed checks are enabled")
    return tuple(timesteps)


def _prepare_fixed_check_samples(
    dataset,
    *,
    num_samples: int,
) -> list[dict[str, Any]]:
    if num_samples <= 0:
        return []

    first_index_for_label: dict[int, int] = {}
    for dataset_idx, label in enumerate(dataset.valid_labels):
        label_int = int(label)
        if label_int not in first_index_for_label:
            first_index_for_label[label_int] = dataset_idx

    prepared = []
    for label in sorted(first_index_for_label)[:num_samples]:
        dataset_idx = first_index_for_label[label]
        sample = dataset[dataset_idx]
        if len(sample) == 4:
            x, y, x_full, hash_key = sample
        else:
            x, y, hash_key = sample
            x_full = None
        prepared.append(
            {
                "dataset_idx": int(dataset_idx),
                "label": int(y),
                "hash_key": str(hash_key),
                "x": x.detach().cpu().clone(),
                "x_full": None if x_full is None else x_full.detach().cpu().clone(),
            }
        )
    return prepared


def _fixed_check_camera_indices(train_cameras: Optional[dict[str, Any]], num_cam: int) -> list[int]:
    if train_cameras is None or num_cam <= 0:
        return []
    total_cams = int(train_cameras["viewmats"].shape[0])
    if total_cams <= 0:
        return []
    if total_cams <= num_cam:
        return list(range(total_cams))
    return torch.linspace(0, total_cams - 1, steps=num_cam).round().to(dtype=torch.long).tolist()


def _summarize_predicted_gaussians(
    *,
    render_utils,
    sample: torch.Tensor,
    norm_mean: Optional[torch.Tensor],
    norm_std: Optional[torch.Tensor],
    dc_only: bool,
) -> dict[str, float]:
    pred_pc = render_utils._plane_to_point_cloud_batch(sample.float())
    pred_pc_raw = render_utils._denormalize_point_cloud(pred_pc, norm_mean, norm_std)
    pred_gaussians = render_utils._point_clouds_to_gsplat_inputs(
        pred_pc_raw,
        dc_only=dc_only,
        detach_input=True,
    )
    return {
        "xyz_mean_abs": float(pred_gaussians["means"].abs().mean().item()),
        "xyz_max_abs": float(pred_gaussians["means"].abs().max().item()),
        "opacity_mean": float(pred_gaussians["opacities"].mean().item()),
        "opacity_min": float(pred_gaussians["opacities"].min().item()),
        "opacity_max": float(pred_gaussians["opacities"].max().item()),
        "scale_mean": float(pred_gaussians["scales"].mean().item()),
        "scale_min": float(pred_gaussians["scales"].min().item()),
        "scale_max": float(pred_gaussians["scales"].max().item()),
    }


def _save_fixed_check_render_preview(
    *,
    render_utils,
    renderer_tuple,
    train_cameras: dict[str, Any],
    cam_indices: list[int],
    x_pred: torch.Tensor,
    x_gt: torch.Tensor,
    norm_mean_pred: Optional[torch.Tensor],
    norm_std_pred: Optional[torch.Tensor],
    norm_mean_gt: Optional[torch.Tensor],
    norm_std_gt: Optional[torch.Tensor],
    dc_only: bool,
    gt_dc_only: bool,
    output_path: str,
    device: torch.device,
) -> None:
    pred_pc = render_utils._plane_to_point_cloud_batch(x_pred.float())
    pred_pc_raw = render_utils._denormalize_point_cloud(pred_pc, norm_mean_pred, norm_std_pred)
    pred_gaussians = render_utils._point_clouds_to_gsplat_inputs(
        pred_pc_raw.to(device),
        dc_only=dc_only,
        detach_input=True,
    )

    gt_pc = render_utils._plane_to_point_cloud_batch(x_gt.float())
    gt_pc_raw = render_utils._denormalize_point_cloud(gt_pc, norm_mean_gt, norm_std_gt)
    gt_gaussians = render_utils._point_clouds_to_gsplat_inputs(
        gt_pc_raw.to(device),
        dc_only=gt_dc_only,
        detach_input=True,
    )

    target_views = render_utils._render_gsplat_batch(
        renderer_tuple,
        gt_gaussians,
        train_cameras,
        cam_indices,
        device,
    )[0]
    pred_views = render_utils._render_gsplat_batch(
        renderer_tuple,
        pred_gaussians,
        train_cameras,
        cam_indices,
        device,
    )[0]

    rows = []
    for target, pred in zip(target_views, pred_views):
        target_np = (
            target.permute(1, 2, 0).clamp(0.0, 1.0).cpu().numpy() * 255.0
        ).astype(np.uint8)
        pred_np = (
            pred.permute(1, 2, 0).clamp(0.0, 1.0).cpu().numpy() * 255.0
        ).astype(np.uint8)
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
    render_utils.Image.fromarray(preview).save(output_path)


def _run_fixed_denoise_checks(
    *,
    model: torch.nn.Module,
    diffusion,
    fixed_check_samples: list[dict[str, Any]],
    fixed_check_timesteps: tuple[int, ...],
    fixed_check_seed: int,
    fixed_check_num_cam: int,
    render_utils,
    norm_mean: Optional[torch.Tensor],
    norm_std: Optional[torch.Tensor],
    norm_mean_full: Optional[torch.Tensor],
    norm_std_full: Optional[torch.Tensor],
    train_cameras: Optional[dict[str, Any]],
    renderer_tuple,
    output_dir: str,
    epoch: int,
    step: int,
    device: torch.device,
    dc_only: bool,
) -> None:
    if not fixed_check_samples or not fixed_check_timesteps:
        return

    checks_dir = os.path.join(output_dir, "fixed_denoise_checks")
    os.makedirs(checks_dir, exist_ok=True)
    step_dir = os.path.join(checks_dir, f"step_{step:07d}")
    os.makedirs(step_dir, exist_ok=True)

    cam_indices = _fixed_check_camera_indices(train_cameras, fixed_check_num_cam)
    can_render = (
        render_utils is not None
        and renderer_tuple is not None
        and train_cameras is not None
        and len(cam_indices) > 0
    )

    records = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for sample_pos, sample_info in enumerate(fixed_check_samples):
            x = sample_info["x"].unsqueeze(0).to(device=device)
            x_gt_render = sample_info["x_full"] if sample_info["x_full"] is not None else sample_info["x"]
            x_gt_render = x_gt_render.unsqueeze(0).to(device=device)
            gt_dc_only = sample_info["x_full"] is None and dc_only
            y = torch.tensor([sample_info["label"]], dtype=torch.long, device=device)

            for timestep in fixed_check_timesteps:
                t_batch = torch.tensor([timestep], dtype=torch.long, device=device)
                generator = torch.Generator(device=device)
                generator.manual_seed(int(fixed_check_seed + sample_pos * 100000 + timestep))
                noise = torch.randn(x.shape, device=device, dtype=x.dtype, generator=generator)
                x_t = diffusion.q_sample(x, t_batch, noise=noise)
                pred = model(x_t, t_batch, y)

                record = {
                    "epoch": int(epoch),
                    "step": int(step),
                    "dataset_idx": int(sample_info["dataset_idx"]),
                    "label": int(sample_info["label"]),
                    "hash_key": sample_info["hash_key"],
                    "timestep": int(timestep),
                    "mse": float((pred.float() - x.float()).pow(2).mean().item()),
                    "mae": float((pred.float() - x.float()).abs().mean().item()),
                    "pred_mean": float(pred.float().mean().item()),
                    "pred_std": float(pred.float().std().item()),
                    "target_mean": float(x.float().mean().item()),
                    "target_std": float(x.float().std().item()),
                }

                if render_utils is not None:
                    try:
                        record.update(
                            _summarize_predicted_gaussians(
                                render_utils=render_utils,
                                sample=pred.detach().cpu(),
                                norm_mean=norm_mean,
                                norm_std=norm_std,
                                dc_only=dc_only,
                            )
                        )
                    except Exception as exc:  # pragma: no cover - diagnostic path
                        record["gaussian_summary_error"] = str(exc)

                if can_render:
                    preview_path = os.path.join(
                        step_dir,
                        f"sample_{sample_pos:02d}_class{sample_info['label']:03d}_t{timestep:04d}.png",
                    )
                    try:
                        _save_fixed_check_render_preview(
                            render_utils=render_utils,
                            renderer_tuple=renderer_tuple,
                            train_cameras=train_cameras,
                            cam_indices=cam_indices,
                            x_pred=pred,
                            x_gt=x_gt_render,
                            norm_mean_pred=norm_mean,
                            norm_std_pred=norm_std,
                            norm_mean_gt=norm_mean_full if sample_info["x_full"] is not None else norm_mean,
                            norm_std_gt=norm_std_full if sample_info["x_full"] is not None else norm_std,
                            dc_only=dc_only,
                            gt_dc_only=gt_dc_only,
                            output_path=preview_path,
                            device=device,
                        )
                        record["render_preview"] = os.path.basename(preview_path)
                    except Exception as exc:  # pragma: no cover - diagnostic path
                        record["render_error"] = str(exc)

                records.append(record)
                logger.info(
                    "[fixed-check] step=%d class=%d t=%d mse=%.5f mae=%.5f%s",
                    step,
                    sample_info["label"],
                    timestep,
                    record["mse"],
                    record["mae"],
                    "" if "scale_max" not in record else f" scale_max={record['scale_max']:.5f}",
                )

    if was_training:
        model.train()

    metrics_path = os.path.join(step_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)
    latest_path = os.path.join(checks_dir, "latest_metrics.json")
    with open(latest_path, "w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)
    logger.info("[fixed-check] saved: %s (%d records)", metrics_path, len(records))


def main(args) -> None:
    dynamo_plugin = _build_dynamo_plugin(args)
    accelerator_kwargs: dict[str, Any] = dict(
        mixed_precision="no" if args.mixed_precision == "none" else args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with=None,
    )
    if dynamo_plugin is not None:
        accelerator_kwargs["dynamo_plugin"] = dynamo_plugin
    accelerator = Accelerator(**accelerator_kwargs)
    device = accelerator.device
    is_main = accelerator.is_main_process

    if is_main:
        dynamo_state = accelerator.state.dynamo_plugin
        logger.info(
            "Accelerator: num_processes=%d, mixed_precision=%s, device=%s",
            accelerator.num_processes,
            accelerator.mixed_precision,
            device,
        )
        logger.info(
            "Compilation: backend=%s, mode=%s, fullgraph=%s, dynamic=%s, regional=%s",
            dynamo_state.backend.value.lower(),
            dynamo_state.mode,
            dynamo_state.fullgraph,
            dynamo_state.dynamic,
            dynamo_state.use_regional_compilation,
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
        sphere2plane_path=args.sphere2plane_path,
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

    render_loss_requested = (
        args.render_loss_weight > 0.0
        or args.alpha_mask_loss_weight > 0.0
        or args.lpips_loss_weight > 0.0
    )
    use_render_loss = render_loss_requested and args.enable_render_loss_after >= 0
    enable_train_render_log = args.train_render_log_every > 0
    if render_loss_requested and not use_render_loss and is_main:
        logger.info("[render-loss] disabled because enable_render_loss_after < 0")

    dataset = Class3DGenDataset(
        base_dataset,
        class_map,
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
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)
    loader_kwargs["generator"] = loader_generator
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = args.persistent_workers
        if args.prefetch_factor > 0:
            loader_kwargs["prefetch_factor"] = args.prefetch_factor
    loader = DataLoader(**loader_kwargs)
    if is_main:
        logger.info("Dataset size: %d, Per-GPU batch size: %d", len(dataset), args.batch_size)

    if is_main:
        logger.info("Creating model: %s (spatial_fold_factor=%d)", args.model, args.spatial_fold_factor)
    model = build_gaussianverse_unet(
        args.model,
        sample_size=128,
        in_channels=in_channels,
        out_channels=in_channels,
        num_classes=num_classes,
        class_embedding_dim=args.class_embed_dim,
        norm_num_groups=args.norm_num_groups,
        dropout=args.dropout,
        spatial_fold_factor=args.spatial_fold_factor,
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
    use_min_snr_weighting = (
        args.snr_gamma is not None and (not args.predict_xstart or args.allow_x0_min_snr_weighting)
    )
    if is_main:
        logger.info(
            "Diffusion timesteps: %d, predict=%s, schedule=%s, snr_gamma=%s",
            diffusion.num_timesteps,
            "x0" if args.predict_xstart else "eps",
            args.noise_schedule,
            args.snr_gamma,
        )
        if args.predict_xstart and args.snr_gamma is not None and not args.allow_x0_min_snr_weighting:
            logger.info(
                "Min-SNR weighting is disabled for predict_xstart=True; "
                "using unweighted x0 MSE to preserve supervision at high-noise timesteps."
            )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_func = IterExponential(
        total_iter_length=args.lr_total_steps * accelerator.num_processes,
        final_ratio=args.lr_final_ratio,
        warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
    )
    lr_scheduler = LambdaLR(optimizer=opt, lr_lambda=lr_func)
    model, opt, loader, lr_scheduler = accelerator.prepare(model, opt, loader, lr_scheduler)
    steps_per_epoch = len(loader)
    if steps_per_epoch <= 0:
        raise ValueError("Training dataloader is empty after batching; reduce batch_size or provide more samples")
    raw_model = _unwrap_training_model(accelerator, model)
    update_ema(ema, raw_model, decay=0.0)

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
    fixed_check_utils = None
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
            fixed_check_utils = render_utils
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

    if args.fixed_check_every > 0 and fixed_check_utils is None:
        fixed_check_probe = _load_render_utils()
        if isinstance(fixed_check_probe, Exception):
            if is_main:
                logger.warning("[fixed-check] gaussian summary disabled: helper import failed: %s", fixed_check_probe)
        else:
            fixed_check_utils = fixed_check_probe

    fixed_check_timesteps: tuple[int, ...] = ()
    fixed_check_samples: list[dict[str, Any]] = []
    if args.fixed_check_every > 0:
        fixed_check_timesteps = _parse_fixed_check_timesteps(
            args.fixed_check_timesteps,
            diffusion_steps=diffusion.num_timesteps,
        )
        if is_main:
            fixed_check_samples = _prepare_fixed_check_samples(
                dataset,
                num_samples=args.fixed_check_num_samples,
            )
            if fixed_check_samples:
                logger.info(
                    "Fixed denoise checks: every=%d, timesteps=%s, samples=%d, seed=%d",
                    args.fixed_check_every,
                    list(fixed_check_timesteps),
                    len(fixed_check_samples),
                    args.fixed_check_seed,
                )
                for sample_info in fixed_check_samples:
                    logger.info(
                        "[fixed-check] sample dataset_idx=%d class=%d hash_key=%s",
                        sample_info["dataset_idx"],
                        sample_info["label"],
                        sample_info["hash_key"],
                    )
            else:
                logger.warning("[fixed-check] enabled but no samples were prepared")

    start_step = 0
    start_epoch = 0
    resume_step_in_epoch = 0
    resume_rng_state = None
    if args.resume:
        if is_main:
            logger.info("Resuming from checkpoint: %s", args.resume)
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        raw_model.load_state_dict(_normalize_compile_wrapped_keys(ckpt["model"]))
        ema.load_state_dict(ckpt["ema"])
        ema.eval()
        opt.load_state_dict(ckpt["opt"])
        if "lr_scheduler" in ckpt:
            lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        start_step = int(ckpt["step"])
        start_epoch = start_step // steps_per_epoch
        resume_step_in_epoch = start_step % steps_per_epoch
        resume_rng_state = ckpt.get("rng_state")
        if is_main:
            logger.info(
                "Resumed at step %d (epoch=%d, step_in_epoch=%d)",
                start_step,
                start_epoch,
                resume_step_in_epoch,
            )
            if resume_rng_state is None:
                logger.warning(
                    "Checkpoint does not contain RNG state; optimizer/model progress is restored, "
                    "but stochastic ops will not resume bitwise-identically."
                )

    if resume_rng_state is not None:
        _restore_rng_state(resume_rng_state, is_main=is_main)

    model.train()
    step = start_step
    log_loss = 0.0
    log_render_l1 = 0.0
    log_render_alpha_l1 = 0.0
    log_render_lpips = 0.0
    log_steps = 0
    start_time = time.time()

    if is_main:
        logger.info("Starting training from epoch %d, step %d...", start_epoch, start_step)

    dc_only = args.sh_degree0_only
    has_full_for_render = getattr(dataset, "return_full_for_render", False)

    for epoch in range(start_epoch, args.epochs):
        _seed_dataloader_for_epoch(
            loader,
            epoch=epoch,
            base_seed=args.seed,
            loader_generator=loader_generator,
        )
        epoch_step_offset = resume_step_in_epoch if epoch == start_epoch else 0
        if epoch_step_offset > 0 and is_main:
            logger.info("Skipping %d already-processed batches in epoch %d", epoch_step_offset, epoch)

        for batch_idx, batch in enumerate(loader):
            if batch_idx < epoch_step_offset:
                continue
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
                if use_min_snr_weighting:
                    alphas_cumprod = torch.tensor(diffusion.alphas_cumprod, device=device, dtype=torch.float32)
                    snr = alphas_cumprod[t] / (1.0 - alphas_cumprod[t])
                    # min-SNR weighting (arXiv:2303.09556, Section 3.4)
                    snr_weight = torch.minimum(snr, torch.full_like(snr, args.snr_gamma))
                    if args.predict_xstart:
                        # x0 prediction: natural weight = 1, apply min(SNR, gamma) directly.
                        # This branch is only active when explicitly re-enabled via args.
                        pass
                    else:
                        # epsilon prediction: natural weight = SNR, divide out
                        snr_weight = snr_weight / snr
                    mse_loss = (sample_losses * snr_weight).mean()
                    if not args.predict_xstart:
                        # Scale by 4.0 only for epsilon prediction to compensate for
                        # reduced average magnitude from SNR division
                        mse_loss = mse_loss * 4.0
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
                render_alpha_l1_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
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

                x0_pred = loss_dict.get("pred_xstart")
                if x0_pred is not None:
                    x0_pred = x0_pred.float()
                x_gt_for_render = x_full if x_full is not None else x
                if should_compute_render and x0_pred is None:
                    noise_for_render = torch.randn_like(x)
                    x_t = diffusion.q_sample(x, t, noise=noise_for_render)
                    model_out = model(x_t, t, y)
                    if args.predict_xstart:
                        x0_pred = model_out.float()
                    else:
                        x0_pred = diffusion._predict_xstart_from_eps(x_t.float(), t, model_out.float())

                if use_render_loss and x0_pred is not None and render_utils is not None:
                    render_l1_loss, render_alpha_l1_loss, render_lpips_loss = render_utils._compute_render_loss_for_batch(
                        x0_pred=x0_pred,
                        x_gt_full=x_gt_for_render,
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
                    + float(args.alpha_mask_loss_weight) * render_alpha_l1_loss
                    + float(args.lpips_loss_weight) * render_lpips_loss
                )

                accelerator.backward(total_loss)
                if args.max_grad_norm > 0.0:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                opt.step()
                lr_scheduler.step()
                opt.zero_grad()

            update_ema(ema, raw_model, decay=args.ema_decay)

            log_loss += mse_loss.item()
            log_render_l1 += render_l1_loss.item()
            log_render_alpha_l1 += render_alpha_l1_loss.item()
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
                    avg_alpha_rl1 = log_render_alpha_l1 / log_steps
                    avg_rlpips = log_render_lpips / log_steps
                    message += (
                        f" | Render_L1: {avg_rl1:.4f}"
                        f" | Alpha_L1: {avg_alpha_rl1:.4f}"
                        f" | Render_LPIPS: {avg_rlpips:.4f}"
                    )
                logger.info(message)
                log_loss = 0.0
                log_render_l1 = 0.0
                log_render_alpha_l1 = 0.0
                log_render_lpips = 0.0
                log_steps = 0
                start_time = time.time()

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
                    noise_schedule=args.noise_schedule,
                    val_sampling_steps=args.val_sampling_steps,
                    val_sampler=args.val_sampler,
                    dpm_solver_order=args.dpm_solver_order,
                    dpm_algorithm_type=args.dpm_algorithm_type,
                    dpm_solver_type=args.dpm_solver_type,
                    dpm_timestep_spacing=args.dpm_timestep_spacing,
                    dpm_use_karras_sigmas=args.dpm_use_karras_sigmas,
                )
            if (
                args.fixed_check_every > 0
                and fixed_check_samples
                and step % args.fixed_check_every == 0
                and is_main
            ):
                try:
                    _run_fixed_denoise_checks(
                        model=ema,
                        diffusion=diffusion,
                        fixed_check_samples=fixed_check_samples,
                        fixed_check_timesteps=fixed_check_timesteps,
                        fixed_check_seed=args.fixed_check_seed,
                        fixed_check_num_cam=args.fixed_check_num_cam,
                        render_utils=fixed_check_utils,
                        norm_mean=norm_mean,
                        norm_std=norm_std,
                        norm_mean_full=norm_mean_full,
                        norm_std_full=norm_std_full,
                        train_cameras=train_cameras,
                        renderer_tuple=renderer_for_train,
                        output_dir=args.results_dir,
                        epoch=epoch,
                        step=step,
                        device=device,
                        dc_only=dc_only,
                    )
                except Exception as exc:  # pragma: no cover - diagnostics should not abort training
                    logger.warning("[fixed-check] failed at step %d: %s", step, exc)

            if step % args.ckpt_every == 0 and is_main:
                ckpt_path = os.path.join(args.results_dir, f"{step:07d}.pt")
                _save_checkpoint(
                    ckpt_path=ckpt_path,
                    raw_model=raw_model,
                    ema=ema,
                    opt=opt,
                    lr_scheduler=lr_scheduler,
                    args=args,
                    step=step,
                    steps_per_epoch=steps_per_epoch,
                )
                logger.info("Saved checkpoint to %s", ckpt_path)

    if is_main:
        ckpt_path = os.path.join(args.results_dir, f"{step:07d}.pt")
        _save_checkpoint(
            ckpt_path=ckpt_path,
            raw_model=raw_model,
            ema=ema,
            opt=opt,
            lr_scheduler=lr_scheduler,
            args=args,
            step=step,
            steps_per_epoch=steps_per_epoch,
        )
        logger.info("Training complete. Final checkpoint: %s", ckpt_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a diffusers UNet for GaussianVerse generation")

    parser.add_argument("--model", type=str, default="UNet-B", choices=sorted(GAUSSIANVERSE_UNET_PRESETS.keys()))
    parser.add_argument("--class_embed_dim", type=int, default=768, help="Class embedding dimension for UNet conditioning")
    parser.add_argument("--norm_num_groups", type=int, default=32, help="GroupNorm group count inside the UNet")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout used inside the diffusers UNet")
    parser.add_argument(
        "--spatial_fold_factor",
        type=int,
        default=1,
        help="Pixel-unshuffle factor applied before the UNet. 2 packs each 2x2 local patch into channels.",
    )
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
    parser.add_argument(
        "--allow_x0_min_snr_weighting",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Keep min-SNR loss weighting active when predict_xstart=True. "
            "Disabled by default because it suppresses high-noise x0 supervision."
        ),
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
    parser.add_argument("--alpha_mask_loss_weight", type=float, default=0.0, help="Weight for render alpha-mask L1 loss term")
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
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable torch.compile through Accelerate. Defaults to the launcher/environment configuration when unset.",
    )
    parser.add_argument(
        "--compile_backend",
        type=str,
        default=None,
        help="TorchDynamo backend passed to Accelerate, for example: inductor, eager, aot_eager, no",
    )
    parser.add_argument(
        "--compile_mode",
        type=str,
        default=None,
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode used by Accelerate when compilation is enabled",
    )
    parser.add_argument(
        "--compile_fullgraph",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Require torch.compile to capture the full model graph",
    )
    parser.add_argument(
        "--compile_dynamic",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable dynamic-shape tracing for torch.compile",
    )
    parser.add_argument(
        "--compile_regional",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use Accelerate regional compilation to reduce cold-start compile time",
    )
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
        "--fixed_check_every",
        type=int,
        default=0,
        help="Run deterministic denoise checks every N steps (0 = disabled)",
    )
    parser.add_argument(
        "--fixed_check_timesteps",
        type=str,
        default="900,975,999",
        help="Comma-separated diffusion timesteps used for deterministic denoise checks",
    )
    parser.add_argument(
        "--fixed_check_num_samples",
        type=int,
        default=2,
        help="Number of fixed class samples to monitor during deterministic denoise checks",
    )
    parser.add_argument(
        "--fixed_check_num_cam",
        type=int,
        default=1,
        help="Number of camera views to render per deterministic denoise check when the renderer is available",
    )
    parser.add_argument(
        "--fixed_check_seed",
        type=int,
        default=1234,
        help="Base RNG seed used to generate deterministic noise for fixed denoise checks",
    )
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
