"""Class-conditional 3DGS dataset wrapper.

Wraps Standard3DGenDataset to pair each sample with its class label,
filter out invalid classes, optionally select a subset of feature channels,
and remap points from sphere order to a 2D plane grid via sphere2plane permutation.
"""

import fcntl
import hashlib
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from dataloaders.standard_3dgen_loader import Standard3DGenDataset

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

logger = logging.getLogger(__name__)

FULL_3DGS_FEATURE_DIM = 59
DC_ONLY_FEATURE_INDICES = (0, 1, 2, 3, 4, 20, 36, 52, 53, 54, 55, 56, 57, 58)
PRELOAD_CACHE_VERSION = 2


def _default_preload_cache_root() -> Path:
    """Return the RAM-backed cache directory used for shared CPU preload."""
    candidate = Path("/dev/shm") / "3dgen_preload_cache"
    try:
        candidate.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(
            "preload_to_cpu requires a writable /dev/shm so the shared cache stays in RAM only"
        ) from exc
    if not os.access(candidate, os.W_OK):
        raise RuntimeError(
            "preload_to_cpu requires a writable /dev/shm so the shared cache stays in RAM only"
        )
    return candidate


def _hash_strings(hasher: "hashlib._Hash", values) -> None:
    for value in values:
        encoded = value.encode("utf-8")
        hasher.update(len(encoded).to_bytes(8, "little"))
        hasher.update(encoded)


def _hash_array(hasher: "hashlib._Hash", array) -> None:
    arr = np.ascontiguousarray(np.asarray(array))
    hasher.update(str(arr.dtype).encode("utf-8"))
    hasher.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
    hasher.update(arr.view(np.uint8).tobytes())


def _torch_dtype_to_numpy(dtype: torch.dtype) -> np.dtype:
    return np.dtype(torch.empty((), dtype=dtype).numpy().dtype)


def load_sphere2plane(sphere2plane_path: str, expected_points: int) -> torch.Tensor:
    """Load and validate the sphere-to-plane permutation array.

    Args:
        sphere2plane_path: Path to sphere2plane.npy file.
        expected_points: Expected number of points (must match array length).

    Returns:
        Long tensor of shape (N,) mapping sphere-order indices to plane-order indices.
    """
    arr = np.load(sphere2plane_path).astype(np.int64)
    if arr.ndim != 1:
        raise ValueError(f"sphere2plane must be 1D, got shape {arr.shape}")
    if arr.shape[0] != expected_points:
        raise ValueError(
            f"sphere2plane has {arr.shape[0]} entries, expected {expected_points}"
        )
    perm = torch.from_numpy(arr).long()
    expected = torch.arange(expected_points, dtype=perm.dtype)
    if not torch.equal(torch.sort(perm).values, expected):
        raise ValueError(f"sphere2plane at {sphere2plane_path} is not a valid permutation")
    return perm


def point_cloud_to_plane(point_cloud: torch.Tensor, plane_to_sphere: torch.Tensor) -> torch.Tensor:
    """Convert sphere-ordered (N, D) point cloud to plane grid (D, H, W).

    Args:
        point_cloud: (N, D) tensor in sphere order.
        plane_to_sphere: (N,) permutation tensor.

    Returns:
        (D, H, W) tensor where H = W = sqrt(N).
    """
    n, d = point_cloud.shape
    side = int(math.isqrt(n))
    assert side * side == n, f"N={n} is not a perfect square"
    plane = point_cloud[plane_to_sphere]  # reorder to plane order
    return plane.view(side, side, d).permute(2, 0, 1).contiguous()


def plane_to_point_cloud(plane_chw: torch.Tensor, plane_to_sphere: torch.Tensor) -> torch.Tensor:
    """Convert plane grid (D, H, W) back to sphere-ordered (N, D) point cloud.

    Args:
        plane_chw: (D, H, W) tensor in plane order.
        plane_to_sphere: (N,) permutation tensor.

    Returns:
        (N, D) tensor in sphere order.
    """
    d, h, w = plane_chw.shape
    n = h * w
    flat = plane_chw.permute(1, 2, 0).reshape(n, d)  # (N, D) in plane order
    # Invert permutation: sphere_to_plane[sphere_idx] = plane_idx
    sphere_to_plane = torch.empty_like(plane_to_sphere)
    sphere_to_plane[plane_to_sphere] = torch.arange(n, dtype=plane_to_sphere.dtype)
    return flat[sphere_to_plane]


class Class3DGenDataset(Dataset):
    """Wraps Standard3DGenDataset for class-conditional training.

    Filters out samples with class label -1 (noise), looks up class labels,
    optionally selects feature channels, and remaps points from sphere order
    to a 2D plane grid using the sphere2plane permutation.
    """

    def __init__(
        self,
        base_dataset: Standard3DGenDataset,
        class_map: dict,
        plane_to_sphere: torch.Tensor,
        feature_indices: Optional[torch.Tensor] = None,
        return_full_for_render: bool = False,
        preload_to_cpu: bool = False,
    ):
        """
        Args:
            base_dataset: Standard3DGenDataset instance.
            class_map: Dict mapping "dir/file" keys to class label ints.
            plane_to_sphere: Permutation tensor from load_sphere2plane().
            feature_indices: Optional tensor of feature column indices to select
                (e.g. for sh_degree0_only mode).
            return_full_for_render: If True and feature_indices is set, also return
                the full 59-channel plane grid for render loss GT.
            preload_to_cpu: If True, eagerly materialize the transformed training
                samples in CPU memory during initialization.
        """
        self.base_dataset = base_dataset
        self.class_map = class_map
        self.plane_to_sphere = plane_to_sphere
        self.feature_indices = feature_indices
        self.return_full_for_render = return_full_for_render and (feature_indices is not None)
        self.preload_to_cpu = preload_to_cpu
        self.cached_pc = None
        self.cached_pc_full = None
        self.cached_labels = None
        self.cached_hash_keys = None
        self._cache_backing = {}

        # Build index of valid samples (class label != -1)
        self.valid_indices = []
        self.valid_labels = []
        skipped = 0
        for idx in range(len(base_dataset)):
            hash_key = base_dataset.keys[idx]
            tar_gz_path = base_dataset.obj_data[hash_key]
            class_key = tar_gz_path.replace('.tar.gz', '')
            label = class_map.get(class_key, -1)
            if label != -1:
                self.valid_indices.append(idx)
                self.valid_labels.append(label)
            else:
                skipped += 1

        logger.info(
            f"Class3DGenDataset: {len(self.valid_indices)} valid samples, "
            f"{skipped} skipped (class -1 or missing)"
        )

        if self.preload_to_cpu:
            self._attach_or_build_preload_cache()

    def __len__(self):
        return len(self.valid_indices)

    def _build_sample(self, real_idx, label: Optional[int] = None):
        sample = self.base_dataset[real_idx]
        hash_key = sample['hash_key']

        # Get class label
        if label is None:
            tar_gz_path = self.base_dataset.obj_data[self.base_dataset.keys[real_idx]]
            class_key = tar_gz_path.replace('.tar.gz', '')
            label = self.class_map[class_key]

        # Point cloud from base dataset is in sphere order: (N, 59)
        pc_full = sample['point_cloud']

        # Select features if requested
        if self.feature_indices is not None:
            pc = pc_full[:, self.feature_indices]  # (N, F)
        else:
            pc = pc_full

        # Remap from sphere order to plane grid: (N, F) -> (F, H, W)
        pc = point_cloud_to_plane(pc, self.plane_to_sphere)

        if self.return_full_for_render:
            pc_full_grid = point_cloud_to_plane(pc_full, self.plane_to_sphere)
            return pc, label, pc_full_grid, hash_key

        return pc, label, hash_key

    def _build_preload_cache_key(self) -> str:
        hasher = hashlib.sha256()
        hasher.update(f"class3dgen_preload_v{PRELOAD_CACHE_VERSION}".encode("utf-8"))
        _hash_strings(hasher, self.base_dataset.keys)
        _hash_strings(hasher, [self.base_dataset.obj_data[key] for key in self.base_dataset.keys])
        _hash_array(hasher, np.asarray(self.valid_indices, dtype=np.int64))
        _hash_array(hasher, np.asarray(self.valid_labels, dtype=np.int64))
        _hash_array(hasher, self.plane_to_sphere.cpu().numpy())
        if self.feature_indices is None:
            hasher.update(b"feature_indices:none")
        else:
            _hash_array(hasher, self.feature_indices.cpu().numpy())
        if self.base_dataset.mean is None:
            hasher.update(b"mean:none")
        else:
            _hash_array(hasher, self.base_dataset.mean)
        if self.base_dataset.std is None:
            hasher.update(b"std:none")
        else:
            _hash_array(hasher, self.base_dataset.std)
        hasher.update(str(self.base_dataset.gs_path).encode("utf-8"))
        hasher.update(
            str(self.base_dataset.rendering_path).encode("utf-8")
            if self.base_dataset.rendering_path is not None else b"rendering_path:none"
        )
        hasher.update(str(self.base_dataset.num_images).encode("utf-8"))
        hasher.update(b"return_full:1" if self.return_full_for_render else b"return_full:0")
        return hasher.hexdigest()[:32]

    def _cache_paths(self):
        cache_root = _default_preload_cache_root()
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_key = self._build_preload_cache_key()
        cache_dir = cache_root / cache_key
        return cache_root, cache_dir, cache_root / f"{cache_key}.lock"

    def _attach_shared_cache(self, meta_path: Path) -> None:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("version") != PRELOAD_CACHE_VERSION:
            raise ValueError(
                f"Unsupported preload cache version {meta.get('version')} at {meta_path}"
            )

        pc_memmap = np.memmap(
            meta["pc_path"],
            mode="r+",
            dtype=np.dtype(meta["pc_dtype"]),
            shape=tuple(meta["pc_shape"]),
        )
        labels_memmap = np.memmap(
            meta["labels_path"],
            mode="r+",
            dtype=np.dtype(meta["labels_dtype"]),
            shape=tuple(meta["labels_shape"]),
        )
        with open(meta["hash_keys_path"], "r", encoding="utf-8") as f:
            hash_keys = json.load(f)

        self.cached_pc = torch.from_numpy(pc_memmap)
        self.cached_labels = torch.from_numpy(labels_memmap)
        self.cached_hash_keys = hash_keys
        self._cache_backing = {
            "pc": pc_memmap,
            "labels": labels_memmap,
        }

        if self.return_full_for_render:
            pc_full_memmap = np.memmap(
                meta["pc_full_path"],
                mode="r+",
                dtype=np.dtype(meta["pc_full_dtype"]),
                shape=tuple(meta["pc_full_shape"]),
            )
            self.cached_pc_full = torch.from_numpy(pc_full_memmap)
            self._cache_backing["pc_full"] = pc_full_memmap
        else:
            self.cached_pc_full = None

    def _build_shared_cache(self, cache_dir: Path, meta_path: Path) -> None:
        preload_start = time.time()
        logger.info(
            "Preloading %d class-conditioned samples into shared CPU cache at %s",
            len(self.valid_indices),
            cache_dir,
        )

        first_real_idx = self.valid_indices[0]
        first_label = self.valid_labels[0]
        first_sample = self._build_sample(first_real_idx, label=first_label)
        first_pc = first_sample[0]
        dataset_len = len(self.valid_indices)

        pc_shape = (dataset_len,) + tuple(first_pc.shape)
        pc_dtype = _torch_dtype_to_numpy(first_pc.dtype)
        pc_path = cache_dir / "pc.dat"
        labels_path = cache_dir / "labels.dat"
        hash_keys_path = cache_dir / "hash_keys.json"

        pc_memmap = np.memmap(pc_path, mode="w+", dtype=pc_dtype, shape=pc_shape)
        labels_memmap = np.memmap(labels_path, mode="w+", dtype=np.int64, shape=(dataset_len,))
        pc_tensor = torch.from_numpy(pc_memmap)
        labels_tensor = torch.from_numpy(labels_memmap)

        pc_full_memmap = None
        pc_full_tensor = None
        pc_full_path = None
        pc_full_shape = None
        pc_full_dtype = None
        if self.return_full_for_render:
            first_pc_full = first_sample[2]
            pc_full_shape = (dataset_len,) + tuple(first_pc_full.shape)
            pc_full_dtype = _torch_dtype_to_numpy(first_pc_full.dtype)
            pc_full_path = cache_dir / "pc_full.dat"
            pc_full_memmap = np.memmap(
                pc_full_path, mode="w+", dtype=pc_full_dtype, shape=pc_full_shape
            )
            pc_full_tensor = torch.from_numpy(pc_full_memmap)

        hash_keys = [None] * dataset_len

        def write_sample(slot: int, cached_sample) -> None:
            pc_tensor[slot].copy_(cached_sample[0])
            labels_tensor[slot] = int(cached_sample[1])
            if self.return_full_for_render:
                pc_full_tensor[slot].copy_(cached_sample[2])
                hash_keys[slot] = cached_sample[3]
            else:
                hash_keys[slot] = cached_sample[2]

        progress = None
        if tqdm is not None:
            progress = tqdm(
                total=dataset_len,
                desc="preload_to_cpu",
                unit="sample",
                dynamic_ncols=True,
            )

        try:
            write_sample(0, first_sample)
            if progress is not None:
                progress.update(1)
            for slot, (real_idx, label) in enumerate(
                zip(self.valid_indices[1:], self.valid_labels[1:]), start=1
            ):
                write_sample(slot, self._build_sample(real_idx, label=label))
                if progress is not None:
                    progress.update(1)
        finally:
            if progress is not None:
                progress.close()

        pc_memmap.flush()
        labels_memmap.flush()
        if pc_full_memmap is not None:
            pc_full_memmap.flush()

        hash_keys_tmp = hash_keys_path.with_suffix(".json.tmp")
        with open(hash_keys_tmp, "w", encoding="utf-8") as f:
            json.dump(hash_keys, f)
        os.replace(hash_keys_tmp, hash_keys_path)

        meta = {
            "version": PRELOAD_CACHE_VERSION,
            "pc_path": str(pc_path),
            "pc_dtype": str(pc_dtype),
            "pc_shape": list(pc_shape),
            "labels_path": str(labels_path),
            "labels_dtype": "int64",
            "labels_shape": [dataset_len],
            "hash_keys_path": str(hash_keys_path),
            "return_full_for_render": self.return_full_for_render,
        }
        if self.return_full_for_render:
            meta.update(
                {
                    "pc_full_path": str(pc_full_path),
                    "pc_full_dtype": str(pc_full_dtype),
                    "pc_full_shape": list(pc_full_shape),
                }
            )

        meta_tmp = meta_path.with_suffix(".tmp")
        with open(meta_tmp, "w", encoding="utf-8") as f:
            json.dump(meta, f)
        os.replace(meta_tmp, meta_path)

        total_bytes = first_pc.numel() * first_pc.element_size() * dataset_len
        if self.return_full_for_render:
            total_bytes += first_sample[2].numel() * first_sample[2].element_size() * dataset_len
        elapsed = time.time() - preload_start
        logger.info(
            "Finished shared CPU preload: %d samples cached in %.1fs (tensor storage %.2f GiB)",
            dataset_len,
            elapsed,
            total_bytes / (1024 ** 3),
        )

    def _attach_or_build_preload_cache(self) -> None:
        cache_root, cache_dir, lock_path = self._cache_paths()
        meta_path = cache_dir / "meta.json"
        with open(lock_path, "a+b") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            if meta_path.exists():
                try:
                    self._attach_shared_cache(meta_path)
                    logger.info("Attached to existing shared CPU preload cache: %s", cache_dir)
                    return
                except (FileNotFoundError, KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
                    logger.warning(
                        "Shared preload cache at %s is invalid (%s); rebuilding",
                        cache_dir,
                        exc,
                    )
                    meta_path.unlink(missing_ok=True)

            cache_dir.mkdir(parents=True, exist_ok=True)
            self._build_shared_cache(cache_dir, meta_path)
            self._attach_shared_cache(meta_path)

    def __getitem__(self, idx):
        if self.cached_pc is not None:
            label = int(self.cached_labels[idx])
            hash_key = self.cached_hash_keys[idx]
            if self.return_full_for_render:
                return self.cached_pc[idx], label, self.cached_pc_full[idx], hash_key
            return self.cached_pc[idx], label, hash_key

        real_idx = self.valid_indices[idx]
        return self._build_sample(real_idx, label=self.valid_labels[idx])
