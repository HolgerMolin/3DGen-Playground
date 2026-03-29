from __future__ import annotations

from typing import Final

import torch

from dataloaders.standard_3dgen_loader import load_sphere2plane as _load_sphere2plane_numpy


def load_sphere2plane(sphere2plane_path: str, expected_points: int) -> torch.Tensor:
    """Load `sphere2plane.npy` as a validated long tensor."""
    arr = _load_sphere2plane_numpy(sphere2plane_path)
    if arr.shape[0] != expected_points:
        raise ValueError(
            f"sphere2plane has {arr.shape[0]} entries, expected {expected_points}"
        )
    return torch.from_numpy(arr.copy()).long()


def plane_to_point_cloud(plane_chw: torch.Tensor, plane_to_sphere: torch.Tensor) -> torch.Tensor:
    """Convert a plane grid (D, H, W) back to sphere-ordered rows (N, D)."""
    channels, height, width = plane_chw.shape
    num_points: Final[int] = height * width
    flat = plane_chw.permute(1, 2, 0).reshape(num_points, channels)
    sphere_to_plane = torch.empty_like(plane_to_sphere)
    sphere_to_plane[plane_to_sphere] = torch.arange(
        num_points,
        device=plane_to_sphere.device,
        dtype=plane_to_sphere.dtype,
    )
    return flat.index_select(0, sphere_to_plane)


__all__ = ["load_sphere2plane", "plane_to_point_cloud"]
