#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data._utils.collate import default_collate
from tqdm.auto import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataloaders.standard_3dgen_loader import Standard3DGenDataset  # noqa: E402


MAX_SKIP_EXAMPLES = 20


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


def _save_tensor(path: Path, tensor: torch.Tensor, dtype: torch.dtype) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor.to(dtype=dtype).contiguous(), path)


def _finalize_mean_std(sum_tensor: torch.Tensor, sumsq_tensor: torch.Tensor, count: int) -> tuple[torch.Tensor, torch.Tensor]:
    if count <= 0:
        raise ValueError("No samples were accumulated")
    mean = sum_tensor / float(count)
    var = sumsq_tensor / float(count) - mean.square()
    var = torch.clamp(var, min=0.0)
    std = torch.sqrt(var)
    return mean, std


def _resolve_num_workers(requested: int, dataset_len: int) -> int:
    if requested < 0:
        raise ValueError("--num_workers must be >= 0")
    if dataset_len <= 0:
        return 0
    if requested > 0:
        return min(requested, dataset_len)

    cpu_count = os.cpu_count() or 1
    if cpu_count <= 1:
        return 0
    # Cap the default worker fan-out so local disk IO and worker startup
    # overhead do not swamp the simple reduction work in the main process.
    return min(dataset_len, max(4, min(8, cpu_count // 2)))


def _resolve_tar_gz_path(dataset: Dataset, idx: int) -> str | None:
    if isinstance(dataset, Subset):
        subset_idx = int(dataset.indices[idx])
        return _resolve_tar_gz_path(dataset.dataset, subset_idx)

    if isinstance(dataset, Standard3DGenDataset):
        if idx < 0 or idx >= len(dataset.keys):
            return None
        hash_key = dataset.keys[idx]
        return dataset.obj_data.get(hash_key)

    return None


class _SkipMissingSampleDataset(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        try:
            return self.dataset[idx]
        except FileNotFoundError as exc:
            return {
                "__skip__": True,
                "dataset_index": int(idx),
                "tar_gz_path": _resolve_tar_gz_path(self.dataset, int(idx)),
                "missing_path": exc.filename,
                "error": str(exc),
            }


def _collate_skip_missing(samples: list[object]) -> dict[str, object]:
    valid_samples = []
    skipped_records = []

    for sample in samples:
        if isinstance(sample, dict) and sample.get("__skip__"):
            skipped_records.append(sample)
        else:
            valid_samples.append(sample)

    batch = default_collate(valid_samples) if valid_samples else None
    return {
        "batch": batch,
        "skipped": skipped_records,
        "valid_count": len(valid_samples),
        "skipped_count": len(skipped_records),
    }


def _build_parser() -> argparse.ArgumentParser:
    env_path = REPO_ROOT / ".env"
    env_values = _load_simple_env(env_path) if env_path.is_file() else {}

    parser = argparse.ArgumentParser(
        description=(
            "Recompute GaussianVerse plane-domain normalization stats with the "
            "updated Standard3DGenDataset preprocessing path."
        )
    )
    parser.add_argument(
        "--obj_list",
        type=str,
        nargs="+",
        default=[
            _default_path(
                env_values,
                "DIT_GSPLAT_OBJ_LIST",
                "/home/tiangexiang/gen3d/gaussianverse/all_obj_list.json",
            )
        ],
        help="One or more obj list JSON files.",
    )
    parser.add_argument(
        "--gs_path",
        type=str,
        default=_default_path(env_values, "DIT_GSPLAT_GS_PATH", "/home/tiangexiang/gen3d/gaussianverse"),
        help="Local GaussianVerse root directory.",
    )
    parser.add_argument(
        "--sphere2plane_path",
        type=str,
        default=_default_path(env_values, "DIT_GSPLAT_SPHERE2PLANE_PATH", "data/sphere2plane.npy"),
        help="Path to sphere2plane.npy used by the updated standard loader.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str((REPO_ROOT / "playground").resolve()),
        help="Directory where the output tensors and metadata JSON will be written.",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="gaussianverse_plane_stats",
        help="Prefix for output filenames.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for the streaming dataloader.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of dataloader worker processes. 0 selects a capped automatic multiprocessing setting.",
    )
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=2,
        help="Dataloader prefetch factor when --num_workers > 0.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit on the number of objects to process. 0 means the full dataset.",
    )
    parser.add_argument(
        "--save_dtype",
        choices=("float32", "float64"),
        default="float32",
        help="Dtype used for the saved output tensors. Accumulation always uses float64.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch_size must be > 0")
    if args.num_workers < 0:
        raise ValueError("--num_workers must be >= 0")
    if args.prefetch_factor <= 0:
        raise ValueError("--prefetch_factor must be > 0")
    if args.limit < 0:
        raise ValueError("--limit must be >= 0")

    save_dtype = torch.float32 if args.save_dtype == "float32" else torch.float64

    dataset = Standard3DGenDataset(
        obj_list=args.obj_list,
        gs_path=args.gs_path,
        caption_path=None,
        rendering_path=None,
        num_images=1,
        mean_file=None,
        std_file=None,
        sphere2plane_path=args.sphere2plane_path,
    )
    if getattr(dataset, "point_cloud_order", None) != "plane":
        raise ValueError(
            "This script expects Standard3DGenDataset.point_cloud_order == 'plane' "
            "so it matches the updated sphere2plane preprocessing path."
        )

    active_dataset = dataset
    if args.limit > 0:
        active_dataset = Subset(dataset, range(min(args.limit, len(dataset))))

    total_objects = len(active_dataset)
    if total_objects == 0:
        raise ValueError("No objects available to process")
    resolved_num_workers = _resolve_num_workers(args.num_workers, total_objects)
    robust_dataset = _SkipMissingSampleDataset(active_dataset)

    dataloader_kwargs = {
        "dataset": robust_dataset,
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": resolved_num_workers,
        "pin_memory": False,
        "drop_last": False,
        "persistent_workers": resolved_num_workers > 0,
        "collate_fn": _collate_skip_missing,
    }
    if resolved_num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = args.prefetch_factor
    dataloader = DataLoader(**dataloader_kwargs)
    print(
        f"Starting stats recomputation over {total_objects} objects with "
        f"batch_size={args.batch_size}, worker_processes={resolved_num_workers}"
    )

    sum_chw: torch.Tensor | None = None
    sumsq_chw: torch.Tensor | None = None
    processed_objects = 0
    skipped_objects = 0
    skipped_examples: list[dict[str, str | int | None]] = []
    num_channels = 0
    height = 0
    width = 0

    progress = tqdm(
        total=total_objects,
        desc="Accumulating plane stats",
        unit="obj",
        dynamic_ncols=True,
    )
    with torch.no_grad():
        for payload in dataloader:
            skipped_records = payload["skipped"]
            skipped_count = int(payload["skipped_count"])
            if skipped_count > 0:
                skipped_objects += skipped_count
                remaining_slots = max(0, MAX_SKIP_EXAMPLES - len(skipped_examples))
                if remaining_slots > 0:
                    for record in skipped_records[:remaining_slots]:
                        skipped_examples.append(
                            {
                                "dataset_index": int(record["dataset_index"]),
                                "tar_gz_path": record["tar_gz_path"],
                                "missing_path": record["missing_path"],
                                "error": record["error"],
                            }
                        )
                for record in skipped_records:
                    tar_gz_path = record["tar_gz_path"] or f"dataset_index={record['dataset_index']}"
                    missing_path = record["missing_path"] or "<unknown>"
                    progress.write(f"[skip] {tar_gz_path} missing {missing_path}")

            progress.update(int(payload["valid_count"]) + skipped_count)
            progress.set_postfix(
                processed=processed_objects,
                skipped=skipped_objects,
                shape=f"{num_channels}x{height}x{width}" if num_channels > 0 else "-",
            )

            batch = payload["batch"]
            if batch is None:
                continue

            point_cloud = batch["point_cloud"]
            if point_cloud.ndim != 3:
                raise ValueError(f"Expected batched point clouds of shape (B, N, C), got {tuple(point_cloud.shape)}")

            batch_size, num_points, batch_channels = point_cloud.shape
            side = int(math.isqrt(num_points))
            if side * side != num_points:
                raise ValueError(f"Point count {num_points} is not a perfect square")

            plane = point_cloud.view(batch_size, side, side, batch_channels).permute(0, 3, 1, 2).contiguous()
            plane64 = plane.to(dtype=torch.float64)

            finite_mask = torch.isfinite(plane64).reshape(batch_size, -1).all(dim=1)
            if not torch.all(finite_mask):
                hash_keys = batch.get("hash_key", [])
                bad_indices = torch.nonzero(~finite_mask, as_tuple=False).flatten().tolist()
                bad_keys = [str(hash_keys[idx]) for idx in bad_indices]
                raise ValueError(f"Encountered non-finite values in samples: {bad_keys}")

            if sum_chw is None:
                num_channels = int(batch_channels)
                height = int(side)
                width = int(side)
                sum_chw = torch.zeros((num_channels, height, width), dtype=torch.float64)
                sumsq_chw = torch.zeros_like(sum_chw)
            elif tuple(plane64.shape[1:]) != tuple(sum_chw.shape):
                raise ValueError(
                    "Inconsistent plane shape encountered: "
                    f"expected {(num_channels, height, width)}, got {tuple(plane64.shape[1:])}"
                )

            sum_chw += plane64.sum(dim=0)
            sumsq_chw += plane64.square().sum(dim=0)
            processed_objects += batch_size
            progress.set_postfix(
                processed=processed_objects,
                skipped=skipped_objects,
                shape=f"{num_channels}x{height}x{width}",
            )
    progress.close()

    if sum_chw is None or sumsq_chw is None:
        raise ValueError("Failed to accumulate any statistics")

    mean_chw, std_chw = _finalize_mean_std(sum_chw, sumsq_chw, processed_objects)
    channel_sum = sum_chw.sum(dim=(1, 2), keepdim=True)
    channel_sumsq = sumsq_chw.sum(dim=(1, 2), keepdim=True)
    mean_c11, std_c11 = _finalize_mean_std(channel_sum, channel_sumsq, processed_objects * height * width)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix

    mean_c11_path = output_dir / f"{prefix}_mean_c11.pt"
    std_c11_path = output_dir / f"{prefix}_std_c11.pt"
    mean_chw_path = output_dir / f"{prefix}_mean_chw.pt"
    std_chw_path = output_dir / f"{prefix}_std_chw.pt"
    metadata_path = output_dir / f"{prefix}_metadata.json"

    _save_tensor(mean_c11_path, mean_c11, save_dtype)
    _save_tensor(std_c11_path, std_c11, save_dtype)
    _save_tensor(mean_chw_path, mean_chw, save_dtype)
    _save_tensor(std_chw_path, std_chw, save_dtype)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "obj_list": list(args.obj_list),
        "gs_path": args.gs_path,
        "sphere2plane_path": args.sphere2plane_path,
        "point_cloud_order": str(dataset.point_cloud_order),
        "processed_objects": int(processed_objects),
        "skipped_objects": int(skipped_objects),
        "limit": int(args.limit),
        "batch_size": int(args.batch_size),
        "num_workers_requested": int(args.num_workers),
        "num_workers_resolved": int(resolved_num_workers),
        "prefetch_factor": int(args.prefetch_factor),
        "accumulation_dtype": "float64",
        "saved_dtype": args.save_dtype,
        "multiprocessing_backend": "torch.utils.data.DataLoader",
        "shapes": {
            "mean_c11": list(mean_c11.shape),
            "std_c11": list(std_c11.shape),
            "mean_chw": list(mean_chw.shape),
            "std_chw": list(std_chw.shape),
        },
        "outputs": {
            "mean_c11": str(mean_c11_path),
            "std_c11": str(std_c11_path),
            "mean_chw": str(mean_chw_path),
            "std_chw": str(std_chw_path),
        },
        "skip_examples": skipped_examples,
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Processed objects: {processed_objects}")
    print(f"Skipped objects: {skipped_objects}")
    print(f"Plane shape: ({num_channels}, {height}, {width})")
    print(f"Saved mean/std (C,1,1) to {mean_c11_path} and {std_c11_path}")
    print(f"Saved mean/std (C,H,W) to {mean_chw_path} and {std_chw_path}")
    print(f"Saved metadata to {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
