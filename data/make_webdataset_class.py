#!/usr/bin/env python3
"""Create class-conditioned WebDataset shards with precomputed 3DGS tensors.

This script mirrors the class-conditional training preprocessing path:
- load raw 3DGS point clouds
- reorder Gaussians into sphere order via gs2sphere
- optionally normalize with dataset mean/std
- remap from sphere order to plane layout
- attach the class label

The output shards are intended to remove the expensive PLY parsing and
point-cloud-to-plane transform from the training hot path.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from functools import partial
import json
import logging
import math
import os
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm
import webdataset as wds

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataloaders.class_3dgen_loader import DC_ONLY_FEATURE_INDICES, load_sphere2plane
from dataloaders.standard_3dgen_loader import extract_directory_info, load_ply


logger = logging.getLogger(__name__)


def setup_logging(output_dir: Path, log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    log_path = output_dir / "webdataset_class_creation.log"
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )


def load_obj_list(obj_list_paths: Sequence[str]) -> Dict[str, str]:
    """Load one or more obj_list JSON files and merge them."""
    merged_data: Dict[str, str] = {}
    for obj_list_path in obj_list_paths:
        logger.info("Loading obj list from %s", obj_list_path)
        with open(obj_list_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise TypeError(f"Expected dict in {obj_list_path}, got {type(data).__name__}")
        logger.info("Loaded %d entries from %s", len(data), obj_list_path)
        merged_data.update(data)
    logger.info("Total merged entries: %d", len(merged_data))
    return merged_data


def load_class_map(class_map_path: str) -> Dict[str, int]:
    """Load object_to_class.json mapping."""
    logger.info("Loading class map from %s", class_map_path)
    with open(class_map_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict in {class_map_path}, got {type(data).__name__}")
    logger.info("Loaded %d class labels", len(data))
    return {str(key): int(value) for key, value in data.items()}


def load_normalization_stats(
    mean_file: Optional[str],
    std_file: Optional[str],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Load normalization stats with the same semantics as training."""
    if mean_file is None and std_file is None:
        logger.warning("Normalization is disabled; mean/std files not provided.")
        return None, None
    if mean_file is None or std_file is None:
        raise ValueError("mean_file and std_file must be provided together")
    logger.info("Loading normalization statistics from %s and %s", mean_file, std_file)
    mean = torch.load(mean_file, map_location="cpu", weights_only=False).cpu().numpy().astype(np.float32)
    std = torch.load(std_file, map_location="cpu", weights_only=False).cpu().numpy().astype(np.float32)
    return mean, std


def resolve_feature_indices(feature_mode: str) -> Optional[np.ndarray]:
    """Return the selected feature indices for the requested mode."""
    if feature_mode == "full":
        return None
    if feature_mode == "dc_only":
        return np.asarray(DC_ONLY_FEATURE_INDICES, dtype=np.int64)
    raise ValueError(f"Unsupported feature_mode: {feature_mode}")


def point_cloud_to_plane_numpy(point_cloud: np.ndarray, plane_to_sphere: np.ndarray) -> np.ndarray:
    """Convert sphere-ordered (N, D) point cloud to plane grid (D, H, W)."""
    num_points, num_features = point_cloud.shape
    side = int(math.isqrt(num_points))
    if side * side != num_points:
        raise ValueError(f"N={num_points} is not a perfect square")
    plane = point_cloud[plane_to_sphere]
    return np.ascontiguousarray(plane.reshape(side, side, num_features).transpose(2, 0, 1))


def convert_storage_dtype(array: np.ndarray, storage_dtype: str) -> np.ndarray:
    """Convert output arrays to the requested storage dtype."""
    array = np.ascontiguousarray(array)
    if storage_dtype == "bfloat16":
        bf16 = torch.from_numpy(array).to(torch.bfloat16)
        return np.ascontiguousarray(bf16.view(torch.uint16).cpu().numpy())
    if storage_dtype == "float32":
        return array.astype(np.float32, copy=False)
    if storage_dtype == "float16":
        return array.astype(np.float16, copy=False)
    raise ValueError(f"Unsupported storage dtype: {storage_dtype}")


def storage_encoding(storage_dtype: str) -> str:
    """Describe how the tensor payload is physically encoded inside the shard."""
    if storage_dtype == "bfloat16":
        return "numpy_uint16_bfloat16_bits"
    if storage_dtype in {"float32", "float16"}:
        return f"numpy_{storage_dtype}"
    raise ValueError(f"Unsupported storage dtype: {storage_dtype}")


def resolve_num_workers(requested_workers: int, total_items: int) -> int:
    """Resolve the worker count used for shard generation."""
    if requested_workers < 0:
        raise ValueError("num_workers must be >= 0")
    if total_items <= 0:
        return 0
    if requested_workers == 0:
        return min(total_items, os.cpu_count() or 1)
    return min(total_items, requested_workers)


def infer_expected_points(items: Sequence[Tuple[str, str, int]], gs_path: str) -> int:
    """Infer the point count from the first valid sample."""
    for _, tar_gz_path, _ in items:
        directory_number, filename = extract_directory_info(tar_gz_path)
        gs2sphere_path = Path(gs_path) / directory_number / filename / "gs2sphere.npy"
        if gs2sphere_path.is_file():
            return int(np.load(str(gs2sphere_path), mmap_mode="r").shape[0])
    raise FileNotFoundError("Could not find a valid gs2sphere.npy to infer point count")


def build_precomputed_sample(
    item: Tuple[str, str, int],
    *,
    gs_path: str,
    mean: Optional[np.ndarray],
    std: Optional[np.ndarray],
    plane_to_sphere: np.ndarray,
    feature_indices: Optional[np.ndarray],
    include_full_plane: bool,
    storage_dtype: str,
    feature_mode: str,
) -> Dict[str, object]:
    """Build a precomputed WebDataset sample for one object."""
    hash_key, tar_gz_path, label = item
    directory_number, filename = extract_directory_info(tar_gz_path)
    data_dir = Path(gs_path) / directory_number / filename

    gs2sphere = np.load(str(data_dir / "gs2sphere.npy"))
    point_cloud = load_ply(str(data_dir / "point_cloud.ply"))

    if gs2sphere.ndim != 1:
        raise ValueError(f"Expected 1D gs2sphere, got shape {gs2sphere.shape}")
    if gs2sphere.shape[0] != point_cloud.shape[0]:
        raise ValueError(
            f"Point count mismatch for {tar_gz_path}: "
            f"point_cloud={point_cloud.shape[0]} vs gs2sphere={gs2sphere.shape[0]}"
        )

    sphere_to_gs = np.empty_like(gs2sphere)
    sphere_to_gs[gs2sphere] = np.arange(gs2sphere.shape[0], dtype=gs2sphere.dtype)
    point_cloud = point_cloud[sphere_to_gs].astype(np.float32, copy=False)

    if mean is not None and std is not None:
        point_cloud = (point_cloud - mean[None]) / (std[None] + 1e-8)

    point_cloud_full_plane = point_cloud_to_plane_numpy(point_cloud, plane_to_sphere)
    if feature_indices is not None:
        point_cloud_plane = point_cloud_to_plane_numpy(point_cloud[:, feature_indices], plane_to_sphere)
    else:
        point_cloud_plane = point_cloud_full_plane

    class_key = tar_gz_path.replace(".tar.gz", "")
    encoding = storage_encoding(storage_dtype)
    sample: Dict[str, object] = {
        "__key__": hash_key,
        "pc_plane.npy": convert_storage_dtype(point_cloud_plane, storage_dtype),
        "label.npy": np.asarray(label, dtype=np.int64),
        "metadata.json": json.dumps(
            {
                "hash_key": hash_key,
                "tar_gz_path": tar_gz_path,
                "class_key": class_key,
                "label": int(label),
                "directory_number": directory_number,
                "filename": filename,
                "feature_mode": feature_mode,
                "storage_dtype": storage_dtype,
                "storage_encoding": encoding,
                "layout": "plane_chw",
                "normalized": mean is not None and std is not None,
            }
        ),
    }
    if include_full_plane and feature_indices is not None:
        sample["pc_full_plane.npy"] = convert_storage_dtype(point_cloud_full_plane, storage_dtype)
    return sample


def iter_precomputed_samples(
    shard_items: Sequence[Tuple[str, str, int]],
    worker_fn,
    executor: Optional[ProcessPoolExecutor],
    max_pending: int,
) -> Iterable[Tuple[Tuple[str, str, int], Optional[Dict[str, object]], Optional[Exception]]]:
    """Yield processed samples either sequentially or from a worker pool."""
    if executor is None:
        for item in shard_items:
            try:
                yield item, worker_fn(item), None
            except Exception as exc:  # pragma: no cover - exercised in integration
                yield item, None, exc
        return

    items_iter = iter(shard_items)
    pending = {}

    while len(pending) < max_pending:
        item = next(items_iter, None)
        if item is None:
            break
        pending[executor.submit(worker_fn, item)] = item

    while pending:
        done, _ = wait(pending, return_when=FIRST_COMPLETED)
        for future in done:
            item = pending.pop(future)
            try:
                yield item, future.result(), None
            except Exception as exc:  # pragma: no cover - exercised in integration
                yield item, None, exc

            next_item = next(items_iter, None)
            if next_item is not None:
                pending[executor.submit(worker_fn, next_item)] = next_item


def create_webdataset_shards(
    *,
    obj_data: Dict[str, str],
    class_map: Dict[str, int],
    gs_path: str,
    sphere2plane_path: str,
    mean: Optional[np.ndarray],
    std: Optional[np.ndarray],
    output_dir: Path,
    shard_size: int,
    max_shards: Optional[int],
    num_workers: int,
    feature_mode: str,
    storage_dtype: str,
    include_full_plane: bool,
    shard_prefix: str,
) -> None:
    """Create class-conditioned WebDataset shards."""
    feature_indices = resolve_feature_indices(feature_mode)
    if include_full_plane and feature_indices is None:
        logger.info("feature_mode=full already stores all channels; ignoring include_full_plane")
        include_full_plane = False

    valid_items: List[Tuple[str, str, int]] = []
    skipped = 0
    for hash_key, tar_gz_path in obj_data.items():
        class_key = tar_gz_path.replace(".tar.gz", "")
        label = class_map.get(class_key, -1)
        if label == -1:
            skipped += 1
            continue
        valid_items.append((hash_key, tar_gz_path, int(label)))

    if max_shards is not None:
        valid_items = valid_items[: max_shards * shard_size]

    if not valid_items:
        raise RuntimeError("No valid class-labeled samples found to shard")

    resolved_num_workers = resolve_num_workers(num_workers, len(valid_items))
    expected_points = infer_expected_points(valid_items, gs_path)
    plane_to_sphere = load_sphere2plane(sphere2plane_path, expected_points).cpu().numpy()
    num_shards = (len(valid_items) + shard_size - 1) // shard_size

    worker_fn = partial(
        build_precomputed_sample,
        gs_path=gs_path,
        mean=mean,
        std=std,
        plane_to_sphere=plane_to_sphere,
        feature_indices=feature_indices,
        include_full_plane=include_full_plane,
        storage_dtype=storage_dtype,
        feature_mode=feature_mode,
    )

    build_info = {
        "input_entries": len(obj_data),
        "valid_entries": len(valid_items),
        "skipped_entries": skipped,
        "shard_size": shard_size,
        "num_shards": num_shards,
        "requested_num_workers": num_workers,
        "resolved_num_workers": resolved_num_workers,
        "feature_mode": feature_mode,
        "feature_indices": feature_indices.tolist() if feature_indices is not None else None,
        "include_full_plane": include_full_plane,
        "storage_dtype": storage_dtype,
        "storage_encoding": storage_encoding(storage_dtype),
        "sphere2plane_path": sphere2plane_path,
        "gs_path": gs_path,
        "normalized": mean is not None and std is not None,
    }
    with open(output_dir / "build_config.json", "w", encoding="utf-8") as handle:
        json.dump(build_info, handle, indent=2, sort_keys=True)

    logger.info("Creating %d shards with up to %d samples each", num_shards, shard_size)
    logger.info("Valid class-labeled items: %d", len(valid_items))
    logger.info("Skipped items with class=-1 or missing label: %d", skipped)
    logger.info("Using %d worker processes", resolved_num_workers)

    successful_entries = 0
    failed_entries = 0

    executor = ProcessPoolExecutor(max_workers=resolved_num_workers) if resolved_num_workers > 0 else None
    max_pending = max(1, resolved_num_workers * 4)
    try:
        for shard_idx in tqdm(range(num_shards), desc="Shards"):
            start_idx = shard_idx * shard_size
            end_idx = min(start_idx + shard_size, len(valid_items))
            shard_items = valid_items[start_idx:end_idx]
            shard_path = output_dir / f"{shard_prefix}-{shard_idx:06d}.tar"
            logger.info("Creating shard %d/%d: %s", shard_idx + 1, num_shards, shard_path)

            with wds.TarWriter(str(shard_path)) as writer:
                progress = tqdm(
                    total=len(shard_items),
                    desc=f"Shard {shard_idx + 1}/{num_shards}",
                    leave=False,
                )
                try:
                    for item, sample, error in iter_precomputed_samples(
                        shard_items,
                        worker_fn,
                        executor,
                        max_pending,
                    ):
                        hash_key, tar_gz_path, _ = item
                        if error is not None:
                            logger.error("Error processing %s (%s): %s", hash_key, tar_gz_path, error)
                            failed_entries += 1
                        else:
                            writer.write(sample)
                            successful_entries += 1
                        progress.update(1)
                finally:
                    progress.close()
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    summary = {
        **build_info,
        "successful_entries": successful_entries,
        "failed_entries": failed_entries,
    }
    with open(output_dir / "build_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    logger.info("WebDataset class shard creation completed")
    logger.info("Successfully processed: %d entries", successful_entries)
    logger.info("Failed to process: %d entries", failed_entries)
    logger.info("Created %d shard files in %s", num_shards, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create class-conditioned WebDataset shards with precomputed plane-layout 3DGS tensors"
    )
    parser.add_argument(
        "--obj_list",
        type=str,
        nargs="+",
        required=True,
        help="One or more paths to obj_list JSON files (will be merged)",
    )
    parser.add_argument(
        "--gs_path",
        type=str,
        required=True,
        help="Path to the directory containing 3DGS fittings",
    )
    parser.add_argument(
        "--class_map",
        type=str,
        required=True,
        help='Path to class map JSON (mapping "dir/file" to class id)',
    )
    parser.add_argument(
        "--sphere2plane_path",
        type=str,
        required=True,
        help="Path to sphere2plane.npy permutation used by class training",
    )
    parser.add_argument(
        "--mean_file",
        type=str,
        default=None,
        help="Optional path to dataset mean tensor used for normalization",
    )
    parser.add_argument(
        "--std_file",
        type=str,
        default=None,
        help="Optional path to dataset std tensor used for normalization",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for WebDataset shards",
    )
    parser.add_argument(
        "--shard_size",
        type=int,
        default=1000,
        help="Number of samples per shard (default: 1000)",
    )
    parser.add_argument(
        "--max_shards",
        type=int,
        default=None,
        help="Maximum number of shards to create (for testing only)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Worker processes used to precompute samples (0 = auto, uses all available CPU workers)",
    )
    parser.add_argument(
        "--feature_mode",
        type=str,
        default="full",
        choices=["full", "dc_only"],
        help="Which feature channels to materialize in pc_plane.npy",
    )
    parser.add_argument(
        "--include_full_plane",
        action="store_true",
        help="Also store the full 59-channel plane tensor as pc_full_plane.npy when feature_mode=dc_only",
    )
    parser.add_argument(
        "--storage_dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float32", "float16"],
        help="Storage dtype for precomputed plane tensors inside the shards "
             "(bfloat16 is stored as uint16 bit patterns in .npy files)",
    )
    parser.add_argument(
        "--shard_prefix",
        type=str,
        default="gaussianverse-class",
        help="Prefix used for output shard tar names",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir, args.log_level)

    for obj_list_file in args.obj_list:
        if not os.path.exists(obj_list_file):
            raise FileNotFoundError(f"Object list file not found: {obj_list_file}")
    if not os.path.exists(args.gs_path):
        raise FileNotFoundError(f"3DGS path not found: {args.gs_path}")
    if not os.path.exists(args.class_map):
        raise FileNotFoundError(f"Class map file not found: {args.class_map}")
    if not os.path.exists(args.sphere2plane_path):
        raise FileNotFoundError(f"sphere2plane file not found: {args.sphere2plane_path}")
    if args.mean_file is not None and not os.path.exists(args.mean_file):
        raise FileNotFoundError(f"Mean file not found: {args.mean_file}")
    if args.std_file is not None and not os.path.exists(args.std_file):
        raise FileNotFoundError(f"Std file not found: {args.std_file}")
    if args.shard_size <= 0:
        raise ValueError("shard_size must be > 0")
    if args.num_workers < 0:
        raise ValueError("num_workers must be >= 0")
    if args.max_shards is not None and args.max_shards <= 0:
        raise ValueError("max_shards must be > 0 when provided")

    obj_data = load_obj_list(args.obj_list)
    class_map = load_class_map(args.class_map)
    mean, std = load_normalization_stats(args.mean_file, args.std_file)

    create_webdataset_shards(
        obj_data=obj_data,
        class_map=class_map,
        gs_path=args.gs_path,
        sphere2plane_path=args.sphere2plane_path,
        mean=mean,
        std=std,
        output_dir=output_dir,
        shard_size=args.shard_size,
        max_shards=args.max_shards,
        num_workers=args.num_workers,
        feature_mode=args.feature_mode,
        storage_dtype=args.storage_dtype,
        include_full_plane=args.include_full_plane,
        shard_prefix=args.shard_prefix,
    )


if __name__ == "__main__":
    main()
