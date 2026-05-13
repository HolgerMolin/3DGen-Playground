"""Empirical test: do dead + needle gaussians actually contribute to renders?

Procedure per object:
  1. Load atlas and flatten to (N=16384, 59) point cloud.
  2. Identify positions whose count-of-large-axes (log_scale >= --threshold)
     is ≤ --max_large_axes (default 1 → dead + needle population).
  3. Render baseline with all gaussians.
  4. Render ablated version with those gaussians' log-scales set to the
     gsplat clip floor (-12 ≈ exp(-12) ≈ 6e-6, sub-pixel by orders of magnitude).
  5. Compute PSNR(baseline, ablated) across all reference cameras.

If PSNR is high (e.g. > 40 dB) the ablated population genuinely contributes
nothing to renders and is safe to mask in training. Lower PSNR means we'd be
discarding visible content.

Run from repo root with the project venv activated. Requires GPU + gsplat.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataloaders.standard_3dgen_loader import Standard3DGenDataset
from utils.gsplat_render_util import (
    RENDER_SCALE_RAW_MIN,
    _load_reference_cameras,
    _point_clouds_to_gsplat_inputs,
    _prepare_train_cameras,
    _render_gsplat_batch,
    _try_import_renderer,
)


def _ablate_scales(pc: torch.Tensor, ablate_mask: torch.Tensor, fill_value: float) -> torch.Tensor:
    """Return a copy of pc (B, N, 59) with log_scale channels (52-54) replaced
    by `fill_value` at positions where ablate_mask is True."""
    out = pc.clone()
    fill = torch.full_like(out[..., 52], fill_value)
    for ch in (52, 53, 54):
        out[..., ch] = torch.where(ablate_mask, fill, out[..., ch])
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--obj_list", required=True, nargs="+")
    p.add_argument("--gs_path", required=True)
    p.add_argument("--sphere2plane_path", required=True)
    p.add_argument("--ref_camera_tar", required=True,
                   help="Path to reference camera tar.gz (REF_CAMERA_TAR in .env).")
    p.add_argument("--exclude_keys_file", default="data/outlier_keys_8sigma.json")
    p.add_argument("--num_objects", type=int, default=30,
                   help="Number of objects to evaluate (loaded sequentially on GPU).")
    p.add_argument("--threshold", type=float, default=-6.0,
                   help="An axis is 'large' iff log_scale >= threshold.")
    p.add_argument("--max_large_axes", type=int, default=1,
                   help="Ablate positions where count_of_large_axes <= this (default 1 = dead+needle).")
    p.add_argument("--render_size", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    exclude_keys_file = args.exclude_keys_file or None

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for gsplat rendering.")
    device = torch.device("cuda")

    renderer = _try_import_renderer()
    if isinstance(renderer, Exception):
        raise renderer

    ds = Standard3DGenDataset(
        obj_list=args.obj_list,
        gs_path=args.gs_path,
        mean_file=None,
        std_file=None,
        sphere2plane_path=args.sphere2plane_path,
        exclude_keys_file=exclude_keys_file,
    )
    n_total = len(ds)
    rng = np.random.default_rng(args.seed)
    indices = rng.choice(n_total, size=min(args.num_objects, n_total), replace=False).tolist()

    cams = _load_reference_cameras(args.ref_camera_tar)
    cam_bundle = _prepare_train_cameras(cams, train_render_size=args.render_size, device=device)
    cam_indices = list(range(len(cams)))
    print(f"Loaded {len(cams)} reference cameras at {args.render_size}²")
    print(f"Evaluating {len(indices)} objects;  threshold={args.threshold}  "
          f"max_large_axes={args.max_large_axes}")
    print(f"Ablation: setting log_scale → {RENDER_SCALE_RAW_MIN} (gsplat scale floor) for "
          f"flagged positions; opacity & SH untouched.\n")

    psnrs = []
    abl_fracs = []
    pixel_diff_max = []
    t0 = time.time()
    for k, idx in enumerate(indices):
        sample = ds[idx]
        pc_np = sample["point_cloud"].numpy() if isinstance(sample["point_cloud"], torch.Tensor) else sample["point_cloud"]
        # (C, H, W) → (1, N, C)
        n_pixels = pc_np.shape[1] * pc_np.shape[2]
        pc_flat = pc_np.reshape(pc_np.shape[0], n_pixels).T
        pc = torch.from_numpy(pc_flat).unsqueeze(0).to(device).contiguous()

        log_scales = pc[..., 52:55]                                   # (1, N, 3)
        n_large = (log_scales >= args.threshold).sum(dim=-1)          # (1, N)
        ablate_mask = n_large <= args.max_large_axes                  # (1, N)
        abl_frac = ablate_mask.float().mean().item()

        pc_ablated = _ablate_scales(pc, ablate_mask, RENDER_SCALE_RAW_MIN)

        with torch.no_grad():
            base_inputs = _point_clouds_to_gsplat_inputs(pc, dc_only=False, detach_input=True)
            abl_inputs = _point_clouds_to_gsplat_inputs(pc_ablated, dc_only=False, detach_input=True)
            base_render = _render_gsplat_batch(renderer, base_inputs, cam_bundle, cam_indices, device)
            abl_render = _render_gsplat_batch(renderer, abl_inputs, cam_bundle, cam_indices, device)

        # Renders are (1, num_cam, 3, H, W) in [0, 1].
        diff = (base_render - abl_render)
        mse_per_view = (diff ** 2).mean(dim=(2, 3, 4))                # (1, num_cam)
        psnr_per_view = -10.0 * torch.log10(mse_per_view.clamp(min=1e-12))
        # Use per-pixel max-abs difference as a worst-case sanity signal.
        max_abs = diff.abs().amax().item()
        avg_psnr = psnr_per_view.mean().item()
        worst_psnr = psnr_per_view.min().item()

        psnrs.append((avg_psnr, worst_psnr))
        abl_fracs.append(abl_frac)
        pixel_diff_max.append(max_abs)

        elapsed = time.time() - t0
        rate = (k + 1) / max(elapsed, 1e-6)
        eta = (len(indices) - k - 1) / max(rate, 1e-6)
        print(f"  obj {k+1:>3d}/{len(indices)}  ablated={abl_frac*100:>5.2f}%  "
              f"PSNR mean={avg_psnr:>5.1f} dB  worst={worst_psnr:>5.1f} dB  "
              f"max|Δ|={max_abs:.4f}   ({rate:.1f} obj/s, eta {eta:.0f}s)")

        del pc, pc_ablated, base_inputs, abl_inputs, base_render, abl_render
        torch.cuda.empty_cache()

    psnrs_arr = np.array(psnrs)              # (N, 2): (mean, worst)
    abl_arr = np.array(abl_fracs)
    pix_arr = np.array(pixel_diff_max)

    print(f"\n=== Summary over {len(indices)} objects ===")
    print(f"  Ablated fraction:       mean={abl_arr.mean()*100:.2f}%  "
          f"min={abl_arr.min()*100:.2f}%  max={abl_arr.max()*100:.2f}%")
    print(f"  Per-view-mean PSNR:     mean={psnrs_arr[:,0].mean():.2f} dB  "
          f"median={np.median(psnrs_arr[:,0]):.2f}  "
          f"min(obj)={psnrs_arr[:,0].min():.2f}")
    print(f"  Worst-view PSNR:        mean={psnrs_arr[:,1].mean():.2f} dB  "
          f"median={np.median(psnrs_arr[:,1]):.2f}  "
          f"min(obj)={psnrs_arr[:,1].min():.2f}")
    print(f"  Max |pixel diff|:       mean={pix_arr.mean():.4f}  "
          f"max(obj)={pix_arr.max():.4f}")
    print("\nReading: PSNR > 40 dB ≈ visually identical (1 part in 100 RMS).")
    print("         PSNR > 50 dB ≈ rounding noise; ablated population truly inert.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
