from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image

REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

GS_ROOT = os.path.join(REPO_ROOT, "submodules", "gaussian-splatting")
if GS_ROOT not in sys.path:
    sys.path.insert(0, GS_ROOT)

from dataloaders.class_3dgen_loader import (  # noqa: E402
    DC_ONLY_FEATURE_INDICES,
    FULL_3DGS_FEATURE_DIM,
)
from unet.models import GAUSSIANVERSE_UNET_PRESETS, build_gaussianverse_unet  # noqa: E402
from unet.sampling import resolve_sampling_shape, sample_with_dpm  # noqa: E402
from utils.plane_utils import load_sphere2plane  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _load_render_utils():
    try:
        return importlib.import_module("utils.gsplat_render_util")
    except Exception as exc:  # pragma: no cover - runtime dependency probe
        return exc


def _looks_like_state_dict(obj: Any) -> bool:
    return isinstance(obj, dict) and bool(obj) and all(
        isinstance(key, str) and torch.is_tensor(value) for key, value in obj.items()
    )


def _select_state_dict(checkpoint: Any, state_key: str) -> tuple[dict[str, torch.Tensor], str]:
    if _looks_like_state_dict(checkpoint):
        return checkpoint, "root"
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")

    if state_key == "root":
        raise KeyError("Checkpoint root is not a raw state dict; choose one of 'auto', 'ema', 'model', or 'state_dict'.")

    if state_key == "auto":
        for key in ("ema", "model", "state_dict"):
            value = checkpoint.get(key)
            if _looks_like_state_dict(value):
                return value, key
        if _looks_like_state_dict(checkpoint):
            return checkpoint, "root"
        raise KeyError("Checkpoint does not contain an 'ema', 'model', or 'state_dict' entry.")

    value = checkpoint.get(state_key)
    if not _looks_like_state_dict(value):
        raise KeyError(f"Checkpoint entry {state_key!r} is missing or is not a state dict.")
    return value, state_key


def _format_value_summary(values: dict[str, Any], keys: tuple[str, ...]) -> str:
    return ", ".join(f"{key}={values[key]!r}" for key in keys if key in values)


def _runtime_config_as_dict(config: SimpleNamespace) -> dict[str, Any]:
    return {
        "model": config.model,
        "class_embed_dim": config.class_embed_dim,
        "norm_num_groups": config.norm_num_groups,
        "dropout": config.dropout,
        "spatial_fold_factor": config.spatial_fold_factor,
        "predict_xstart": config.predict_xstart,
        "noise_schedule": config.noise_schedule,
        "sh_degree0_only": config.sh_degree0_only,
        "class_map": config.class_map,
        "mean_file": config.mean_file,
        "std_file": config.std_file,
        "sphere2plane_path": config.sphere2plane_path,
        "ref_camera_tar": config.ref_camera_tar,
        "render_size": config.render_size,
    }


def _values_match(runtime_value: Any, saved_value: Any) -> bool:
    if isinstance(runtime_value, float) or isinstance(saved_value, float):
        return float(runtime_value) == float(saved_value)
    return runtime_value == saved_value


def _validate_checkpoint_runtime_config(config: SimpleNamespace, saved_args: dict[str, Any]) -> None:
    if not saved_args:
        logger.warning("Checkpoint does not contain saved args; skipping runtime compatibility validation.")
        return

    runtime_values = _runtime_config_as_dict(config)
    mismatch_keys = (
        "model",
        "class_embed_dim",
        "norm_num_groups",
        "dropout",
        "spatial_fold_factor",
        "predict_xstart",
        "noise_schedule",
        "sh_degree0_only",
    )
    mismatches = []
    for key in mismatch_keys:
        if key not in saved_args:
            continue
        runtime_value = runtime_values[key]
        saved_value = saved_args[key]
        if not _values_match(runtime_value, saved_value):
            mismatches.append(f"{key}: runtime={runtime_value!r}, checkpoint={saved_value!r}")

    if mismatches:
        raise ValueError(
            "Runtime UNet config does not match the checkpoint's saved args.\n"
            "Update the inference overrides or use the checkpoint defaults instead.\n"
            + "\n".join(f"  - {item}" for item in mismatches)
        )


def _log_checkpoint_path_overrides(config: SimpleNamespace, saved_args: dict[str, Any]) -> None:
    if not saved_args:
        return

    runtime_values = _runtime_config_as_dict(config)
    override_keys = ("class_map", "mean_file", "std_file", "sphere2plane_path", "ref_camera_tar")
    overrides = []
    for key in override_keys:
        if key not in saved_args:
            continue
        runtime_path = _resolve_repo_path(runtime_values[key])
        saved_path = _resolve_repo_path(saved_args[key])
        if runtime_path != saved_path:
            overrides.append(f"{key}: runtime={runtime_path!r}, checkpoint={saved_args[key]!r}")

    if overrides:
        logger.info("Using runtime asset-path overrides instead of checkpoint paths:\n%s", "\n".join(f"  - {item}" for item in overrides))


def _strip_known_state_dict_prefixes(state_dict: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], str]:
    stripped = state_dict
    removed_prefixes: list[str] = []
    while stripped:
        matched_prefix = None
        for prefix in ("module.", "model."):
            if all(key.startswith(prefix) for key in stripped):
                matched_prefix = prefix
                break
        if matched_prefix is None:
            break
        stripped = {key[len(matched_prefix) :]: value for key, value in stripped.items()}
        removed_prefixes.append(matched_prefix)

    if not removed_prefixes:
        return state_dict, "original"
    return stripped, f"stripped {' + '.join(removed_prefixes)}"


def _analyze_state_dict_match(
    model: torch.nn.Module, state_dict: dict[str, torch.Tensor]
) -> SimpleNamespace:
    model_state = model.state_dict()
    missing_keys = [key for key in model_state.keys() if key not in state_dict]
    unexpected_keys = [key for key in state_dict.keys() if key not in model_state]
    shape_mismatches = []

    for key, value in state_dict.items():
        if key not in model_state:
            continue
        model_value = model_state[key]
        if tuple(value.shape) != tuple(model_value.shape):
            shape_mismatches.append((key, tuple(value.shape), tuple(model_value.shape)))

    return SimpleNamespace(
        missing_keys=missing_keys,
        unexpected_keys=unexpected_keys,
        shape_mismatches=shape_mismatches,
        total_issues=len(missing_keys) + len(unexpected_keys) + len(shape_mismatches),
    )


def _format_examples(items: list[str], max_items: int = 8) -> str:
    if not items:
        return "none"
    shown = items[:max_items]
    suffix = "" if len(items) <= max_items else f", ... (+{len(items) - max_items} more)"
    return ", ".join(shown) + suffix


def _format_shape_mismatches(items: list[tuple[str, tuple[int, ...], tuple[int, ...]]], max_items: int = 6) -> str:
    if not items:
        return "none"
    shown = [f"{key}: checkpoint={ckpt_shape}, model={model_shape}" for key, ckpt_shape, model_shape in items[:max_items]]
    suffix = "" if len(items) <= max_items else f", ... (+{len(items) - max_items} more)"
    return "; ".join(shown) + suffix


def _format_state_dict_report(label: str, report: SimpleNamespace) -> str:
    lines = [
        f"  - {label}: missing={len(report.missing_keys)}, unexpected={len(report.unexpected_keys)}, "
        f"shape_mismatches={len(report.shape_mismatches)}"
    ]
    if report.missing_keys:
        lines.append(f"    missing examples: {_format_examples(report.missing_keys)}")
    if report.unexpected_keys:
        lines.append(f"    unexpected examples: {_format_examples(report.unexpected_keys)}")
    if report.shape_mismatches:
        lines.append(f"    shape mismatches: {_format_shape_mismatches(report.shape_mismatches)}")
    return "\n".join(lines)


def _load_model_weights(model: torch.nn.Module, state_dict: dict[str, torch.Tensor]) -> str:
    candidate_state_dicts = [("original", state_dict)]
    stripped_state_dict, stripped_label = _strip_known_state_dict_prefixes(state_dict)
    if stripped_state_dict is not state_dict:
        candidate_state_dicts.append((stripped_label, stripped_state_dict))

    reports: list[tuple[str, SimpleNamespace]] = []
    for label, candidate in candidate_state_dicts:
        report = _analyze_state_dict_match(model, candidate)
        reports.append((label, report))
        if report.total_issues != 0:
            continue
        model.load_state_dict(candidate, strict=True)
        logger.info(
            "Loaded checkpoint weights with state-dict variant %r: %d tensors (%d total scalars).",
            label,
            len(candidate),
            sum(value.numel() for value in candidate.values()),
        )
        return label

    report_text = "\n".join(_format_state_dict_report(label, report) for label, report in reports)
    raise RuntimeError("Checkpoint weights do not match the constructed model.\n" + report_text)


def _resolve_override(cli_value: Any, saved_args: dict[str, Any], key: str, default: Any = None) -> Any:
    if cli_value is not None:
        return cli_value
    return saved_args.get(key, default)


def _resolve_repo_path(path: Optional[str]) -> Optional[str]:
    if path is None or path == "":
        return None
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = Path(REPO_ROOT) / resolved
    return str(resolved)


def _require_path(path: Optional[str], name: str) -> str:
    if path is None or path == "":
        raise ValueError(f"{name} must be provided explicitly or embedded in the checkpoint args.")
    resolved = _resolve_repo_path(path)
    if resolved is None or not os.path.exists(resolved):
        raise FileNotFoundError(f"{name} does not exist: {path}")
    return resolved


def _resolve_device(device_arg: Optional[str]) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_mixed_precision(mode: str, device: torch.device) -> str:
    if mode != "auto":
        return mode
    if device.type != "cuda":
        return "none"
    if torch.cuda.is_bf16_supported():
        return "bf16"
    return "fp16"


def _dtype_for_mixed_precision(mode: str, device: torch.device) -> torch.dtype:
    if device.type != "cuda" or mode == "none":
        return torch.float32
    if mode == "bf16":
        return torch.bfloat16
    if mode == "fp16":
        return torch.float16
    raise ValueError(f"Unsupported mixed_precision mode: {mode}")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_class_labels(class_map_path: str) -> tuple[list[int], int]:
    with open(class_map_path, "r", encoding="utf-8") as handle:
        class_map = json.load(handle)
    valid_labels = sorted({int(value) for value in class_map.values() if int(value) >= 0})
    if not valid_labels:
        raise ValueError(f"No non-negative class labels found in {class_map_path}")
    return valid_labels, max(valid_labels) + 1


def _fold_sample_for_export(
    sample: torch.Tensor, model: torch.nn.Module
) -> tuple[torch.Tensor, Optional[torch.Tensor], int]:
    spatial_fold_factor = int(getattr(model, "spatial_fold_factor", 1))
    fold_spatial = getattr(model, "fold_spatial", None)
    unfold_spatial = getattr(model, "unfold_spatial", None)

    if not callable(fold_spatial) or not callable(unfold_spatial):
        return sample, None, spatial_fold_factor
    if spatial_fold_factor == 1:
        return sample, None, spatial_fold_factor

    folded_sample = fold_spatial(sample)
    roundtrip_sample = unfold_spatial(folded_sample)
    if not torch.equal(roundtrip_sample, sample):
        max_abs_diff = float((roundtrip_sample - sample).abs().max().item())
        raise RuntimeError(
            "Spatial fold/unfold roundtrip mismatch during inference export: "
            f"spatial_fold_factor={spatial_fold_factor}, max_abs_diff={max_abs_diff}"
        )
    return sample, folded_sample, spatial_fold_factor


def _make_output_dir(results_dir: str, checkpoint_path: str, class_label: int, seed: int, sample_name: Optional[str]) -> Path:
    base = Path(results_dir)
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    name = sample_name or f"{Path(checkpoint_path).stem}_class{class_label:03d}_seed{seed}_{timestamp}"
    output_dir = base / name
    if output_dir.exists():
        output_dir = base / f"{name}_{int(time.time())}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def _save_image(tensor: torch.Tensor, out_path: Path) -> None:
    image = tensor.detach().permute(1, 2, 0).clamp(0.0, 1.0).cpu().numpy()
    image_u8 = (image * 255.0).round().astype(np.uint8)
    Image.fromarray(image_u8).save(out_path)


def _save_render_grid(render_views: torch.Tensor, out_path: Path, max_cols: int = 4) -> None:
    views = [
        (view.detach().permute(1, 2, 0).clamp(0.0, 1.0).cpu().numpy() * 255.0).round().astype(np.uint8)
        for view in render_views
    ]
    if not views:
        return

    cols = min(max_cols, len(views))
    rows = []
    separator = np.full((views[0].shape[0], 4, 3), 255, dtype=np.uint8)
    for start in range(0, len(views), cols):
        chunk = views[start : start + cols]
        row = chunk[0]
        for item in chunk[1:]:
            row = np.concatenate((row, separator, item), axis=1)
        rows.append(row)

    grid = rows[0]
    row_separator = np.full((4, rows[0].shape[1], 3), 255, dtype=np.uint8)
    for row in rows[1:]:
        grid = np.concatenate((grid, row_separator, row), axis=0)
    Image.fromarray(grid).save(out_path)


def _build_runtime_config(args: argparse.Namespace, saved_args: dict[str, Any]) -> SimpleNamespace:
    mean_file = _resolve_override(args.mean_file, saved_args, "mean_file", None)
    std_file = _resolve_override(args.std_file, saved_args, "std_file", None)
    if (mean_file is None) != (std_file is None):
        raise ValueError("mean_file and std_file must either both be set or both be omitted.")

    return SimpleNamespace(
        model=_resolve_override(args.model, saved_args, "model", "UNet-B"),
        class_embed_dim=int(_resolve_override(args.class_embed_dim, saved_args, "class_embed_dim", 768)),
        norm_num_groups=int(_resolve_override(args.norm_num_groups, saved_args, "norm_num_groups", 32)),
        dropout=float(_resolve_override(args.dropout, saved_args, "dropout", 0.0)),
        spatial_fold_factor=int(_resolve_override(args.spatial_fold_factor, saved_args, "spatial_fold_factor", 1)),
        predict_xstart=bool(_resolve_override(args.predict_xstart, saved_args, "predict_xstart", False)),
        noise_schedule=str(_resolve_override(args.noise_schedule, saved_args, "noise_schedule", "linear")),
        sh_degree0_only=bool(_resolve_override(args.sh_degree0_only, saved_args, "sh_degree0_only", False)),
        class_map=_require_path(_resolve_override(args.class_map, saved_args, "class_map", None), "class_map"),
        mean_file=None if mean_file is None else _require_path(mean_file, "mean_file"),
        std_file=None if std_file is None else _require_path(std_file, "std_file"),
        sphere2plane_path=_require_path(
            _resolve_override(args.sphere2plane_path, saved_args, "sphere2plane_path", None),
            "sphere2plane_path",
        ),
        ref_camera_tar=_require_path(
            _resolve_override(
                args.ref_camera_tar,
                saved_args,
                "ref_camera_tar",
                "/home/tiangexiang/gen3d/ref_camera.tar.gz",
            ),
            "ref_camera_tar",
        ),
        render_size=int(_resolve_override(args.render_size, saved_args, "train_render_size", 256)),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Load a pretrained GaussianVerse UNet and run DPM inference.")

    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a UNet training checkpoint.")
    parser.add_argument("--results_dir", type=str, default="output/unet_inference_gsplat")
    parser.add_argument("--sample_name", type=str, default=None, help="Optional output subdirectory name.")
    parser.add_argument(
        "--state_key",
        type=str,
        default="auto",
        choices=["auto", "ema", "model", "state_dict", "root"],
        help="Checkpoint weight entry to load. 'auto' prefers EMA when present.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=sorted(GAUSSIANVERSE_UNET_PRESETS.keys()),
        help="UNet preset override.",
    )
    parser.add_argument("--class_embed_dim", type=int, default=None, help="Class embedding dim override.")
    parser.add_argument("--norm_num_groups", type=int, default=None, help="GroupNorm group count override.")
    parser.add_argument("--dropout", type=float, default=None, help="UNet dropout override.")
    parser.add_argument("--spatial_fold_factor", type=int, default=None, help="Spatial fold factor override.")
    parser.add_argument(
        "--predict_xstart",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override checkpoint diffusion target mode.",
    )
    parser.add_argument(
        "--noise_schedule",
        type=str,
        default=None,
        choices=["linear", "squaredcos_cap_v2"],
        help="Override checkpoint diffusion noise schedule.",
    )
    parser.add_argument(
        "--sh_degree0_only",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override checkpoint feature layout.",
    )

    parser.add_argument("--class_map", type=str, default=None, help="Path to object-to-class JSON.")
    parser.add_argument("--mean_file", type=str, default=None, help="Normalization mean tensor path.")
    parser.add_argument("--std_file", type=str, default=None, help="Normalization std tensor path.")
    parser.add_argument("--sphere2plane_path", type=str, default=None, help="Path to sphere2plane.npy.")
    parser.add_argument("--ref_camera_tar", type=str, default=None, help="Reference camera tar.gz for rendering.")
    parser.add_argument(
        "--render_size",
        "--train_render_size",
        dest="render_size",
        type=int,
        default=None,
        help="Rendering resolution.",
    )

    parser.add_argument("--num_inference_steps", type=int, default=40, help="Number of DPM sampling steps.")
    parser.add_argument("--num_render_views", type=int, default=4, help="Number of rendered views to save.")
    parser.add_argument("--class_label", type=int, default=None, help="Explicit class label. Defaults to a random valid class.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed used for class choice and sampling.")
    parser.add_argument("--device", type=str, default=None, help="Torch device, for example cuda:0.")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="auto",
        choices=["auto", "fp16", "bf16", "none"],
        help="Inference parameter dtype.",
    )

    parser.add_argument("--dpm_solver_order", type=int, default=2, choices=[1, 2, 3], help="DPM solver order.")
    parser.add_argument(
        "--dpm_algorithm_type",
        type=str,
        default="dpmsolver++",
        choices=["dpmsolver", "dpmsolver++", "sde-dpmsolver", "sde-dpmsolver++"],
        help="DPM algorithm variant.",
    )
    parser.add_argument(
        "--dpm_solver_type",
        type=str,
        default="midpoint",
        choices=["midpoint", "heun"],
        help="DPM solver type.",
    )
    parser.add_argument(
        "--dpm_timestep_spacing",
        type=str,
        default="trailing",
        choices=["linspace", "leading", "trailing"],
        help="Diffusers timestep spacing.",
    )
    parser.add_argument(
        "--dpm_use_karras_sigmas",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Karras sigmas in the DPM scheduler.",
    )
    return parser


def main(args: argparse.Namespace) -> None:
    checkpoint_path = _require_path(args.checkpoint, "checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) and isinstance(checkpoint.get("args"), dict) else {}
    config = _build_runtime_config(args, saved_args)
    runtime_config = _runtime_config_as_dict(config)

    if isinstance(checkpoint, dict):
        logger.info("Checkpoint step: %s", checkpoint.get("step"))
        logger.info("Checkpoint top-level keys: %s", sorted(checkpoint.keys()))
    if saved_args:
        logger.info(
            "Checkpoint saved args: %s",
            _format_value_summary(
                saved_args,
                (
                    "model",
                    "class_embed_dim",
                    "norm_num_groups",
                    "dropout",
                    "spatial_fold_factor",
                    "predict_xstart",
                    "noise_schedule",
                    "sh_degree0_only",
                    "train_render_size",
                ),
            ),
        )
    logger.info(
        "Runtime config: %s",
        _format_value_summary(
            runtime_config,
            (
                "model",
                "class_embed_dim",
                "norm_num_groups",
                "dropout",
                "spatial_fold_factor",
                "predict_xstart",
                "noise_schedule",
                "sh_degree0_only",
                "render_size",
            ),
        ),
    )
    _validate_checkpoint_runtime_config(config, saved_args)
    _log_checkpoint_path_overrides(config, saved_args)

    device = _resolve_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("CUDA is required because this inference path renders outputs with gsplat.")

    resolved_mixed_precision = _resolve_mixed_precision(args.mixed_precision, device)
    model_dtype = _dtype_for_mixed_precision(resolved_mixed_precision, device)
    _set_seed(args.seed)

    render_utils = _load_render_utils()
    if isinstance(render_utils, Exception):
        raise RuntimeError(f"Failed to import rendering helpers: {render_utils}") from render_utils
    renderer = render_utils._try_import_renderer()
    if isinstance(renderer, Exception):
        raise RuntimeError(f"Failed to import gsplat renderer: {renderer}") from renderer

    valid_labels, num_classes = _load_class_labels(config.class_map)
    if args.class_label is None:
        class_label = random.choice(valid_labels)
    else:
        class_label = int(args.class_label)
        if class_label not in valid_labels:
            raise ValueError(f"class_label={class_label} is not present in {config.class_map}")

    plane_perm = np.load(config.sphere2plane_path, mmap_mode="r")
    if plane_perm.ndim != 1:
        raise ValueError(f"sphere2plane must be 1D, got shape {tuple(plane_perm.shape)}")
    plane_to_sphere = load_sphere2plane(config.sphere2plane_path, int(plane_perm.shape[0]))

    feature_indices = torch.tensor(DC_ONLY_FEATURE_INDICES, dtype=torch.long) if config.sh_degree0_only else None
    in_channels = len(DC_ONLY_FEATURE_INDICES) if config.sh_degree0_only else FULL_3DGS_FEATURE_DIM

    norm_mean = None
    norm_std = None
    if config.mean_file and config.std_file:
        norm_mean_full = torch.load(config.mean_file, map_location="cpu", weights_only=False).float()
        norm_std_full = torch.load(config.std_file, map_location="cpu", weights_only=False).float()
        if feature_indices is not None:
            norm_mean = norm_mean_full[feature_indices]
            norm_std = norm_std_full[feature_indices]
        else:
            norm_mean = norm_mean_full
            norm_std = norm_std_full

    model = build_gaussianverse_unet(
        config.model,
        sample_size=128,
        in_channels=in_channels,
        out_channels=in_channels,
        num_classes=num_classes,
        class_embedding_dim=config.class_embed_dim,
        norm_num_groups=config.norm_num_groups,
        dropout=config.dropout,
        spatial_fold_factor=config.spatial_fold_factor,
        gradient_checkpointing=False,
    )
    state_dict, loaded_state_key = _select_state_dict(checkpoint, args.state_key)
    try:
        loaded_state_variant = _load_model_weights(model, state_dict)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Failed to load checkpoint entry {loaded_state_key!r} from {checkpoint_path}.\n"
            f"Runtime config: {_format_value_summary(runtime_config, ('model', 'class_embed_dim', 'norm_num_groups', 'dropout', 'spatial_fold_factor', 'predict_xstart', 'noise_schedule', 'sh_degree0_only', 'render_size'))}\n"
            f"Checkpoint saved args: {_format_value_summary(saved_args, ('model', 'class_embed_dim', 'norm_num_groups', 'dropout', 'spatial_fold_factor', 'predict_xstart', 'noise_schedule', 'sh_degree0_only', 'train_render_size'))}\n"
            f"{exc}"
        ) from exc
    model = model.to(device=device, dtype=model_dtype)
    model.eval()
    sample_shape = resolve_sampling_shape(model=model, batch_size=1, in_channels=in_channels)
    logger.info(
        "Sampling unfolded plane shape=%s; internal folded shape=(1, %d, %d, %d); spatial_fold_factor=%d",
        sample_shape,
        int(getattr(model, "folded_in_channels", in_channels)),
        int(getattr(model, "folded_sample_size", sample_shape[-1])),
        int(getattr(model, "folded_sample_size", sample_shape[-1])),
        int(getattr(model, "spatial_fold_factor", 1)),
    )

    ref_cameras = render_utils._load_reference_cameras(config.ref_camera_tar)
    train_cameras = render_utils._prepare_train_cameras(ref_cameras, config.render_size, device)
    output_dir = _make_output_dir(args.results_dir, checkpoint_path, class_label, args.seed, args.sample_name)
    render_dir = output_dir / "renderings"
    render_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loaded checkpoint: %s", checkpoint_path)
    logger.info("Using state dict: %s (%s)", loaded_state_key, loaded_state_variant)
    logger.info("Sampling class=%d with %d DPM steps", class_label, args.num_inference_steps)
    logger.info("Output directory: %s", output_dir)

    y = torch.tensor([class_label], dtype=torch.long, device=device)
    with torch.inference_mode():
        sample = sample_with_dpm(
            model=model,
            shape=sample_shape,
            class_labels=y,
            num_inference_steps=args.num_inference_steps,
            device=device,
            predict_xstart=config.predict_xstart,
            noise_schedule=config.noise_schedule,
            solver_order=args.dpm_solver_order,
            algorithm_type=args.dpm_algorithm_type,
            solver_type=args.dpm_solver_type,
            timestep_spacing=args.dpm_timestep_spacing,
            use_karras_sigmas=args.dpm_use_karras_sigmas,
        )

        pred_pc = render_utils._plane_to_point_cloud_batch(sample.float(), plane_to_sphere)
        pred_pc_raw = render_utils._denormalize_point_cloud(pred_pc, norm_mean, norm_std)
        pred_gaussians = render_utils._point_clouds_to_gsplat_inputs(
            pred_pc_raw.to(device),
            dc_only=config.sh_degree0_only,
            detach_input=True,
        )

        total_cams = int(train_cameras["viewmats"].shape[0])
        if total_cams <= 0:
            raise ValueError("No render cameras available after preparing the camera bundle.")
        cam_indices = random.sample(range(total_cams), min(max(1, args.num_render_views), total_cams))
        rendered_views = render_utils._render_gsplat_batch(
            renderer,
            pred_gaussians,
            train_cameras,
            cam_indices,
            device,
        )[0]

    sample_unfolded, sample_folded, spatial_fold_factor = _fold_sample_for_export(sample, model)
    if sample_folded is not None:
        logger.info(
            "Saved unfolded sample shape=%s and folded sample shape=%s for spatial_fold_factor=%d",
            tuple(sample_unfolded.shape),
            tuple(sample_folded.shape),
            spatial_fold_factor,
        )

    for index, (cam_idx, view) in enumerate(zip(cam_indices, rendered_views)):
        _save_image(view, render_dir / f"view_{index:02d}_cam{cam_idx:03d}.png")
    _save_render_grid(rendered_views, output_dir / "render_grid.png")

    sample_payload = {
        "metadata": {
            "checkpoint": checkpoint_path,
            "checkpoint_step": None if not isinstance(checkpoint, dict) else checkpoint.get("step"),
            "loaded_state_key": loaded_state_key,
            "class_label": class_label,
            "valid_class_labels": valid_labels,
            "seed": args.seed,
            "device": str(device),
            "mixed_precision": resolved_mixed_precision,
            "num_inference_steps": args.num_inference_steps,
            "num_render_views": len(cam_indices),
            "render_camera_indices": cam_indices,
            "dpm_solver_order": args.dpm_solver_order,
            "dpm_algorithm_type": args.dpm_algorithm_type,
            "dpm_solver_type": args.dpm_solver_type,
            "dpm_timestep_spacing": args.dpm_timestep_spacing,
            "dpm_use_karras_sigmas": args.dpm_use_karras_sigmas,
            "predict_xstart": config.predict_xstart,
            "noise_schedule": config.noise_schedule,
            "sh_degree0_only": config.sh_degree0_only,
            "model": config.model,
            "class_embed_dim": config.class_embed_dim,
            "norm_num_groups": config.norm_num_groups,
            "dropout": config.dropout,
            "spatial_fold_factor": config.spatial_fold_factor,
            "render_size": config.render_size,
            "sample_shape": tuple(sample_unfolded.shape),
            "folded_sample_shape": None if sample_folded is None else tuple(sample_folded.shape),
            "folded_in_channels": int(getattr(model, "folded_in_channels", in_channels)),
            "folded_sample_size": int(getattr(model, "folded_sample_size", sample_unfolded.shape[-1])),
        },
        "sample_plane": sample_unfolded[0].float().cpu(),
        "sample_plane_folded": None if sample_folded is None else sample_folded[0].float().cpu(),
        "point_cloud_normalized": pred_pc[0].float().cpu(),
        "point_cloud_denormalized": pred_pc_raw[0].float().cpu(),
        "gaussians": {
            key: value[0].detach().cpu() if torch.is_tensor(value) else value
            for key, value in pred_gaussians.items()
        },
    }
    sample_path = output_dir / "sample_3dgs.pt"
    torch.save(sample_payload, sample_path)

    logger.info("Saved 3DGS payload: %s", sample_path)
    logger.info("Saved renderings to: %s", render_dir)
    logger.info("Saved render grid: %s", output_dir / "render_grid.png")


if __name__ == "__main__":
    main(build_parser().parse_args())
