#!/usr/bin/env python3
"""Render real 3DGS objects with fixed scale/rotation ablations."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dataloaders.standard_3dgen_loader import Standard3DGenDataset
from dit.train_gsplat import (
    _denormalize_point_cloud,
    _load_reference_cameras,
    _point_clouds_to_gsplat_inputs,
    _prepare_train_cameras,
    _render_gsplat_batch,
)


FIXED_QUATERNION = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obj_list", type=str, required=True, help="Path to object list JSON.")
    parser.add_argument("--gs_path", type=str, required=True, help="Root directory of GaussianVerse objects.")
    parser.add_argument("--mean_file", type=str, required=True, help="Normalization mean .pt file.")
    parser.add_argument("--std_file", type=str, required=True, help="Normalization std .pt file.")
    parser.add_argument("--ref_camera_tar", type=str, required=True, help="Reference camera tarball.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for metrics and previews.")
    parser.add_argument("--num_objects", type=int, default=12, help="Number of real objects to evaluate.")
    parser.add_argument("--num_cams", type=int, default=4, help="Number of cameras per object.")
    parser.add_argument("--render_size", type=int, default=256, help="Render resolution.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--stats_objects",
        type=int,
        default=64,
        help="Number of objects used to estimate fixed global scale constants.",
    )
    parser.add_argument(
        "--save_previews",
        type=int,
        default=4,
        help="How many evaluated objects to save as preview strips.",
    )
    return parser.parse_args()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _psnr_from_mse(mse: torch.Tensor) -> torch.Tensor:
    mse = mse.clamp_min(1e-12)
    return -10.0 * torch.log10(mse)


def _compute_scale_constants(
    dataset: Standard3DGenDataset,
    mean: torch.Tensor,
    std: torch.Tensor,
    sample_indices: list[int],
) -> tuple[float, torch.Tensor]:
    scales = []
    for idx in sample_indices:
        sample = dataset[idx]["point_cloud"].float()
        raw = _denormalize_point_cloud(sample, mean, std)
        scales.append(torch.exp(raw[:, 52:55]))
    scales_all = torch.cat(scales, dim=0)
    scalar = float(scales_all.reshape(-1).median().item())
    axiswise = scales_all.median(dim=0).values
    return scalar, axiswise


def _canonicalize_raw_point_cloud(
    raw_point_cloud: torch.Tensor,
    *,
    scale_scalar: float | None = None,
    scale_axiswise: torch.Tensor | None = None,
    fix_rotation: bool = False,
) -> torch.Tensor:
    out = raw_point_cloud.clone()
    if scale_scalar is not None:
        log_scale = math.log(scale_scalar)
        out[:, 52:55] = log_scale
    elif scale_axiswise is not None:
        out[:, 52:55] = scale_axiswise.to(out).log().view(1, 3)

    if fix_rotation:
        out[:, 55:59] = FIXED_QUATERNION.to(out).view(1, 4)
    return out


def _render_variants(
    raw_point_cloud: torch.Tensor,
    renderer_module: Any,
    camera_bundle: dict[str, Any],
    cam_indices: list[int],
    device: torch.device,
    scale_scalar: float,
    scale_axiswise: torch.Tensor,
) -> dict[str, torch.Tensor]:
    variants_raw = {
        "baseline": raw_point_cloud,
        "rot_fixed": _canonicalize_raw_point_cloud(raw_point_cloud, fix_rotation=True),
        "scale_scalar_fixed": _canonicalize_raw_point_cloud(raw_point_cloud, scale_scalar=scale_scalar),
        "scale_axis_fixed": _canonicalize_raw_point_cloud(raw_point_cloud, scale_axiswise=scale_axiswise),
        "scale_rot_fixed": _canonicalize_raw_point_cloud(
            raw_point_cloud,
            scale_scalar=scale_scalar,
            fix_rotation=True,
        ),
    }

    renders: dict[str, torch.Tensor] = {}
    for name, raw_pc in variants_raw.items():
        gaussians = _point_clouds_to_gsplat_inputs(raw_pc.unsqueeze(0).to(device), dc_only=False, detach_input=True)
        renders[name] = _render_gsplat_batch(renderer_module, gaussians, camera_bundle, cam_indices, device)[0].cpu()
    return renders


def _metrics_against_baseline(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    diff = candidate - reference
    l1 = diff.abs().mean()
    mse = diff.square().mean()
    psnr = _psnr_from_mse(mse)
    return {
        "l1": float(l1.item()),
        "mse": float(mse.item()),
        "psnr": float(psnr.item()),
    }


def _make_preview_strip(renders: dict[str, torch.Tensor], out_path: Path) -> None:
    order = ["baseline", "rot_fixed", "scale_scalar_fixed", "scale_axis_fixed", "scale_rot_fixed"]
    labels = {
        "baseline": "baseline",
        "rot_fixed": "rot fixed",
        "scale_scalar_fixed": "scalar scale",
        "scale_axis_fixed": "axis scale",
        "scale_rot_fixed": "scale+rot",
    }

    cam_count = renders["baseline"].shape[0]
    height = renders["baseline"].shape[-2]
    width = renders["baseline"].shape[-1]
    pad = 8
    label_h = 20
    canvas = Image.new(
        "RGB",
        (cam_count * width + (cam_count + 1) * pad, len(order) * (height + label_h + pad) + pad),
        color=(18, 18, 18),
    )
    draw = ImageDraw.Draw(canvas)

    for row, key in enumerate(order):
        y = pad + row * (height + label_h + pad)
        draw.text((pad, y), labels[key], fill=(235, 235, 235))
        row_images = renders[key].permute(0, 2, 3, 1).numpy()
        for col in range(cam_count):
            x = pad + col * (width + pad)
            arr = np.clip(row_images[col] * 255.0, 0.0, 255.0).astype(np.uint8)
            canvas.paste(Image.fromarray(arr), (x, y + label_h))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> None:
    args = _parse_args()
    _set_seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for gsplat rendering.")

    device = torch.device("cuda")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = Standard3DGenDataset(
        obj_list=[args.obj_list],
        gs_path=args.gs_path,
        caption_path=None,
        mean_file=args.mean_file,
        std_file=args.std_file,
    )
    mean = torch.load(args.mean_file, weights_only=True).float()
    std = torch.load(args.std_file, weights_only=True).float()

    total = len(dataset)
    eval_count = min(args.num_objects, total)
    stats_count = min(args.stats_objects, total)
    all_indices = list(range(total))
    random.shuffle(all_indices)
    eval_indices = all_indices[:eval_count]
    stats_indices = all_indices[:stats_count]

    scale_scalar, scale_axiswise = _compute_scale_constants(dataset, mean, std, stats_indices)

    import gsplat  # imported late to avoid hard dependency for non-render use

    ref_cameras = _load_reference_cameras(args.ref_camera_tar)
    camera_bundle = _prepare_train_cameras(ref_cameras, args.render_size, device)
    cam_count = min(args.num_cams, int(camera_bundle["viewmats"].shape[0]))
    cam_indices = list(range(cam_count))

    print(f"Evaluating {eval_count} objects with {cam_count} cameras each")
    print(f"Fixed scalar scale: {scale_scalar:.8f}")
    print(f"Fixed axiswise scale: {[float(v) for v in scale_axiswise.tolist()]}")

    per_object: list[dict[str, Any]] = []
    aggregate: dict[str, list[dict[str, float]]] = {
        "rot_fixed": [],
        "scale_scalar_fixed": [],
        "scale_axis_fixed": [],
        "scale_rot_fixed": [],
    }

    for rank, idx in enumerate(eval_indices):
        sample = dataset[idx]
        raw_pc = _denormalize_point_cloud(sample["point_cloud"].float(), mean, std)
        renders = _render_variants(
            raw_pc,
            renderer_module=gsplat,
            camera_bundle=camera_bundle,
            cam_indices=cam_indices,
            device=device,
            scale_scalar=scale_scalar,
            scale_axiswise=scale_axiswise,
        )

        baseline = renders["baseline"]
        obj_metrics: dict[str, Any] = {
            "dataset_index": int(idx),
            "hash_key": sample["hash_key"],
            "metrics": {},
        }
        for key in aggregate:
            metrics = _metrics_against_baseline(baseline, renders[key])
            aggregate[key].append(metrics)
            obj_metrics["metrics"][key] = metrics

        per_object.append(obj_metrics)

        if rank < args.save_previews:
            preview_path = output_dir / f"preview_{rank:02d}_{sample['hash_key']}.png"
            _make_preview_strip(renders, preview_path)

        print(
            f"[{rank + 1:02d}/{eval_count:02d}] {sample['hash_key']} "
            f"rot_fixed_l1={obj_metrics['metrics']['rot_fixed']['l1']:.4f} "
            f"scale_scalar_l1={obj_metrics['metrics']['scale_scalar_fixed']['l1']:.4f} "
            f"scale_rot_l1={obj_metrics['metrics']['scale_rot_fixed']['l1']:.4f}"
        )

    summary: dict[str, Any] = {
        "config": {
            "obj_list": args.obj_list,
            "gs_path": args.gs_path,
            "mean_file": args.mean_file,
            "std_file": args.std_file,
            "ref_camera_tar": args.ref_camera_tar,
            "num_objects": eval_count,
            "num_cams": cam_count,
            "render_size": args.render_size,
            "seed": args.seed,
            "fixed_scale_scalar": scale_scalar,
            "fixed_scale_axiswise": [float(v) for v in scale_axiswise.tolist()],
        },
        "aggregate": {},
        "per_object": per_object,
    }

    for key, metrics_list in aggregate.items():
        summary["aggregate"][key] = {
            metric_name: float(np.mean([m[metric_name] for m in metrics_list]))
            for metric_name in ("l1", "mse", "psnr")
        }

    summary_path = output_dir / "metrics.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
