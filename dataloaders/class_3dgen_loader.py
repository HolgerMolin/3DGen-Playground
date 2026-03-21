"""Class-conditional 3DGS dataset wrapper.

Wraps Standard3DGenDataset to pair each sample with its class label,
filter out invalid classes, optionally select a subset of feature channels,
and remap points from sphere order to a 2D plane grid via sphere2plane permutation.
"""

import logging
import math
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from dataloaders.standard_3dgen_loader import Standard3DGenDataset

logger = logging.getLogger(__name__)

FULL_3DGS_FEATURE_DIM = 59
DC_ONLY_FEATURE_INDICES = (0, 1, 2, 3, 4, 20, 36, 52, 53, 54, 55, 56, 57, 58)


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
        """
        self.base_dataset = base_dataset
        self.class_map = class_map
        self.plane_to_sphere = plane_to_sphere
        self.feature_indices = feature_indices
        self.return_full_for_render = return_full_for_render and (feature_indices is not None)

        # Build index of valid samples (class label != -1)
        self.valid_indices = []
        skipped = 0
        for idx in range(len(base_dataset)):
            hash_key = base_dataset.keys[idx]
            tar_gz_path = base_dataset.obj_data[hash_key]
            class_key = tar_gz_path.replace('.tar.gz', '')
            label = class_map.get(class_key, -1)
            if label != -1:
                self.valid_indices.append(idx)
            else:
                skipped += 1

        logger.info(
            f"Class3DGenDataset: {len(self.valid_indices)} valid samples, "
            f"{skipped} skipped (class -1 or missing)"
        )

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        sample = self.base_dataset[real_idx]
        hash_key = sample['hash_key']

        # Get class label
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
