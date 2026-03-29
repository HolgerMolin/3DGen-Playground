#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataloaders.class_3dgen_loader import (  # noqa: E402
    DC_ONLY_FEATURE_INDICES,
    FULL_3DGS_FEATURE_DIM,
)
from dataloaders.standard_3dgen_loader import Standard3DGenDataset  # noqa: E402
from utils.plane_utils import load_sphere2plane  # noqa: E402


SH_C0 = 0.28209479177387814


def _load_simple_env(env_path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    var_pattern = re.compile(r"\$(\w+)|\$\{(\w+)\}")

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = re.split(r"\s+#", value.strip(), maxsplit=1)[0].strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]

        def replace_var(match: re.Match[str]) -> str:
            name = match.group(1) or match.group(2)
            return values.get(name, os.environ.get(name, ""))

        values[key] = var_pattern.sub(replace_var, value)

    return values


def _default_path(env_values: dict[str, str], key: str, fallback: str) -> str:
    value = env_values.get(key, fallback)
    return str((REPO_ROOT / value).resolve()) if not os.path.isabs(value) else value


def _full_feature_names() -> list[str]:
    names = ["x", "y", "z", "opacity"]
    for color in ("r", "g", "b"):
        names.extend([f"{color}_sh_{idx:02d}" for idx in range(16)])
    names.extend(["scale_x", "scale_y", "scale_z", "rot_0", "rot_1", "rot_2", "rot_3"])
    if len(names) != FULL_3DGS_FEATURE_DIM:
        raise RuntimeError(f"Expected {FULL_3DGS_FEATURE_DIM} names, got {len(names)}")
    return names


FULL_FEATURE_NAMES = _full_feature_names()


def _selected_feature_names(feature_indices: Optional[torch.Tensor]) -> list[str]:
    if feature_indices is None:
        return FULL_FEATURE_NAMES
    return [FULL_FEATURE_NAMES[int(index)] for index in feature_indices.tolist()]


def _robust_normalize(channel_hw: np.ndarray) -> np.ndarray:
    channel = np.asarray(channel_hw, dtype=np.float32)
    finite = np.isfinite(channel)
    if not finite.any():
        return np.zeros_like(channel, dtype=np.float32)

    values = channel[finite]
    lo = float(np.quantile(values, 0.01))
    hi = float(np.quantile(values, 0.99))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(values.min())
        hi = float(values.max())
    if hi <= lo:
        return np.zeros_like(channel, dtype=np.float32)

    channel = np.nan_to_num(channel, nan=lo, posinf=hi, neginf=lo)
    return np.clip((channel - lo) / (hi - lo), 0.0, 1.0)


def _scalar_to_rgb_uint8(channel_hw: np.ndarray) -> np.ndarray:
    normalized = _robust_normalize(channel_hw)
    return (np.repeat(normalized[:, :, None], 3, axis=2) * 255.0).round().astype(np.uint8)


def _stack_to_rgb_uint8(channels_chw: np.ndarray) -> np.ndarray:
    if channels_chw.shape[0] != 3:
        raise ValueError(f"Expected 3 channels, got {channels_chw.shape[0]}")
    rgb = np.stack([_robust_normalize(channels_chw[idx]) for idx in range(3)], axis=-1)
    return (rgb * 255.0).round().astype(np.uint8)


def _save_png(path: Path, rgb_uint8: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb_uint8, mode="RGB").save(path)


def _denormalize_plane(
    plane_chw: np.ndarray,
    mean_full: Optional[np.ndarray],
    std_full: Optional[np.ndarray],
    feature_indices: Optional[torch.Tensor],
) -> np.ndarray:
    if mean_full is None or std_full is None:
        return plane_chw
    if feature_indices is None:
        mean = mean_full
        std = std_full
    else:
        selected = feature_indices.cpu().numpy()
        mean = mean_full[selected]
        std = std_full[selected]
    return plane_chw * (std[:, None, None] + 1e-8) + mean[:, None, None]


def _point_cloud_to_plane(point_cloud: torch.Tensor, plane_to_sphere: torch.Tensor) -> torch.Tensor:
    num_points, num_features = point_cloud.shape
    side = int(num_points ** 0.5)
    if side * side != num_points:
        raise ValueError(f"N={num_points} is not a perfect square")
    plane = point_cloud[plane_to_sphere]
    return plane.view(side, side, num_features).permute(2, 0, 1).contiguous()


def _plane_point_cloud_to_grid(point_cloud: torch.Tensor) -> torch.Tensor:
    num_points, num_features = point_cloud.shape
    side = int(num_points ** 0.5)
    if side * side != num_points:
        raise ValueError(f"N={num_points} is not a perfect square")
    return point_cloud.reshape(side, side, num_features).permute(2, 0, 1).contiguous()


def _save_named_channels(sample_idx: int, plane_raw: np.ndarray, feature_names: list[str], output_dir: Path) -> None:
    for channel_idx, channel_name in enumerate(feature_names):
        filename = f"{sample_idx}_{channel_name}.png"
        _save_png(output_dir / filename, _scalar_to_rgb_uint8(plane_raw[channel_idx]))


def _save_dc_only_groups(sample_idx: int, plane_raw: np.ndarray, output_dir: Path) -> list[str]:
    saved: list[str] = []

    xyz_rgb = _stack_to_rgb_uint8(plane_raw[0:3])
    xyz_name = f"{sample_idx}_xyz.png"
    _save_png(output_dir / xyz_name, xyz_rgb)
    saved.append(xyz_name)

    opacity = torch.sigmoid(torch.from_numpy(np.clip(plane_raw[3], -12.0, 12.0))).numpy()
    opacity_name = f"{sample_idx}_opacity.png"
    _save_png(output_dir / opacity_name, (np.repeat(opacity[:, :, None], 3, axis=2) * 255.0).round().astype(np.uint8))
    saved.append(opacity_name)

    rgb = np.clip(np.moveaxis(plane_raw[4:7], 0, -1) * SH_C0 + 0.5, 0.0, 1.0)
    rgb_name = f"{sample_idx}_rgb.png"
    _save_png(output_dir / rgb_name, (rgb * 255.0).round().astype(np.uint8))
    saved.append(rgb_name)

    scale_name = f"{sample_idx}_scale.png"
    _save_png(output_dir / scale_name, _stack_to_rgb_uint8(plane_raw[7:10]))
    saved.append(scale_name)

    rot_rgb_name = f"{sample_idx}_rotation_rgb.png"
    _save_png(output_dir / rot_rgb_name, _stack_to_rgb_uint8(plane_raw[10:13]))
    saved.append(rot_rgb_name)

    rot_last_name = f"{sample_idx}_rotation_3.png"
    _save_png(output_dir / rot_last_name, _scalar_to_rgb_uint8(plane_raw[13]))
    saved.append(rot_last_name)

    return saved


def _build_parser() -> argparse.ArgumentParser:
    env_path = REPO_ROOT / ".env"
    env_values = _load_simple_env(env_path) if env_path.is_file() else {}

    parser = argparse.ArgumentParser(
        description="Export a few plane-constructed 3DGS samples from the UNet dataloader as PNGs."
    )
    parser.add_argument(
        "--obj_list",
        type=str,
        default=_default_path(
            env_values,
            "DIT_GSPLAT_OBJ_LIST",
            "/home/tiangexiang/gen3d/gaussianverse/all_obj_list.json",
        ),
    )
    parser.add_argument(
        "--gs_path",
        type=str,
        default=_default_path(env_values, "DIT_GSPLAT_GS_PATH", "/home/tiangexiang/gen3d/gaussianverse"),
    )
    parser.add_argument(
        "--mean_file",
        type=str,
        default=_default_path(
            env_values,
            "DIT_GSPLAT_MEAN_FILE",
            "/home/tiangexiang/gen3d/gaussianverse/gaussianverse_mean.pt",
        ),
    )
    parser.add_argument(
        "--std_file",
        type=str,
        default=_default_path(
            env_values,
            "DIT_GSPLAT_STD_FILE",
            "/home/tiangexiang/gen3d/gaussianverse/gaussianverse_std.pt",
        ),
    )
    parser.add_argument(
        "--class_map",
        type=str,
        default=_default_path(env_values, "DIT_GSPLAT_CLASS_MAP", "object_labels/object_to_class.json"),
    )
    parser.add_argument(
        "--sphere2plane_path",
        type=str,
        default=_default_path(env_values, "DIT_GSPLAT_SPHERE2PLANE_PATH", "data/sphere2plane.npy"),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str((REPO_ROOT / "playground").resolve()),
    )
    parser.add_argument("--num_samples", type=int, default=3, help="Number of samples to export")
    parser.add_argument(
        "--start_index",
        type=int,
        default=3,
        help="Base-dataset index to start scanning from when exporting new examples; defaults to skipping the first exported slice",
    )
    parser.add_argument(
        "--sh_degree0_only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mirror the launcher default and export the 14-channel DC-only plane tensors",
    )
    parser.add_argument(
        "--skip_unlabeled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip base-dataset samples whose class_map entry is missing or -1",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.class_map, "r", encoding="utf-8") as handle:
        class_map = json.load(handle)

    base_dataset = Standard3DGenDataset(
        obj_list=[args.obj_list],
        gs_path=args.gs_path,
        caption_path=None,
        sphere2plane_path=args.sphere2plane_path,
    )

    point_cloud_shape = tuple(base_dataset[0]["point_cloud"].shape)
    num_points = (
        int(point_cloud_shape[-2] * point_cloud_shape[-1])
        if len(point_cloud_shape) == 3
        else int(point_cloud_shape[0])
    )
    plane_to_sphere = load_sphere2plane(args.sphere2plane_path, num_points)
    feature_indices = torch.tensor(DC_ONLY_FEATURE_INDICES, dtype=torch.long) if args.sh_degree0_only else None

    mean_full = None if base_dataset.mean is None else np.asarray(base_dataset.mean, dtype=np.float32)
    std_full = None if base_dataset.std is None else np.asarray(base_dataset.std, dtype=np.float32)
    feature_names = _selected_feature_names(feature_indices)

    export_summary: list[dict[str, object]] = []
    sample_idx = 0
    for base_idx in range(args.start_index, len(base_dataset)):
        if sample_idx >= args.num_samples:
            break

        batch = base_dataset[base_idx]
        tar_gz_path = str(batch["tar_gz_path"])
        class_key = tar_gz_path.replace(".tar.gz", "")
        label = int(class_map.get(class_key, -1))
        if args.skip_unlabeled and label < 0:
            continue

        point_cloud = batch["point_cloud"].detach().cpu()
        if feature_indices is not None:
            point_cloud = point_cloud[:, feature_indices]

        if getattr(base_dataset, "point_cloud_order", "sphere") == "plane":
            plane_raw = _plane_point_cloud_to_grid(point_cloud).numpy().astype(np.float32)
        else:
            plane_raw = _point_cloud_to_plane(point_cloud, plane_to_sphere).numpy().astype(np.float32)
        # plane_raw = _denormalize_plane(plane_norm, mean_full, std_full, feature_indices)

        sample_idx += 1
        saved_files: list[str] = []
        if args.sh_degree0_only:
            saved_files.extend(_save_dc_only_groups(sample_idx, plane_raw, output_dir))
        _save_named_channels(sample_idx, plane_raw, feature_names, output_dir)
        saved_files.extend([f"{sample_idx}_{name}.png" for name in feature_names])
        saved_files = list(dict.fromkeys(saved_files))

        record = {
            "sample_index": sample_idx,
            "base_index": base_idx,
            "hash_key": str(batch["hash_key"]),
            "tar_gz_path": tar_gz_path,
            "label": label,
            "saved_files": saved_files,
        }
        export_summary.append(record)
        with open(output_dir / f"{sample_idx}_metadata.json", "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2)

    with open(output_dir / "export_summary.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "num_samples": len(export_summary),
                "feature_mode": "dc_only" if args.sh_degree0_only else "full",
                "output_dir": str(output_dir),
                "samples": export_summary,
            },
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()
