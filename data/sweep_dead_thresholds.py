"""Diagnostic sweep: characterize opacity / scale distributions and the
resulting dead-fraction at many threshold combinations.

For each scanned object, collects three flattened (16384,) arrays per atlas:
  - opacity logit         (ch 3, raw pre-sigmoid)
  - max(log-scale xyz)    (max of ch 52-54, "invisible from any angle" criterion)
  - min(log-scale xyz)    (min of ch 52-54, "at least one axis sub-pixel" criterion)

Then prints:
  1. Marginal percentile tables for each quantity (across all atlas positions).
  2. A 2D sweep grid of dead-fraction over (opacity_thresh × max_log_scale_thresh)
     under the strict "max < thresh" rule — gaussians invisible from every view.
  3. Same grid under the loose "min < thresh" rule for comparison —
     gaussians sub-pixel along at least one axis (visible side-on).

Holds raw distributions in RAM (~128 KB / object), so use --limit to keep
total memory bounded. 200-1000 objects is plenty for threshold selection.

Run from repo root with the project venv activated.
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataloaders.standard_3dgen_loader import Standard3DGenDataset


OPACITY_CHANNEL = 3
SCALE_CHANNELS = (52, 53, 54)


_DATASET: Optional[Standard3DGenDataset] = None


def _worker_init(obj_list, gs_path, sphere2plane_path, exclude_keys_file):
    global _DATASET
    logging.getLogger().setLevel(logging.WARNING)
    _DATASET = Standard3DGenDataset(
        obj_list=obj_list,
        gs_path=gs_path,
        mean_file=None,
        std_file=None,
        sphere2plane_path=sphere2plane_path,
        exclude_keys_file=exclude_keys_file,
    )


def _worker_fn(idx: int):
    sample = _DATASET[idx]
    pc = sample["point_cloud"]
    if isinstance(pc, torch.Tensor):
        pc = pc.detach().cpu().numpy()
    pc = pc.astype(np.float32)
    if pc.ndim != 3:
        raise RuntimeError(f"expected (C, H, W) point cloud, got shape {pc.shape}")

    opacity = pc[OPACITY_CHANNEL].reshape(-1)              # (N,)
    log_scales = pc[list(SCALE_CHANNELS)].reshape(3, -1)   # (3, N)
    return idx, opacity, log_scales.max(axis=0), log_scales.min(axis=0)


def _print_percentile_table(name: str, values: np.ndarray) -> None:
    qs = (0.001, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.999)
    print(f"\n--- Percentiles of {name} (over all {values.size:,} atlas positions) ---")
    header = "  " + "".join(f"  p{int(q*1000)/10:>5.1f}" for q in qs)
    print(header)
    line = "  " + "".join(f"  {np.quantile(values, q):>+6.2f}" for q in qs)
    print(line)
    print(f"  min={values.min():+.3f}  mean={values.mean():+.3f}  "
          f"max={values.max():+.3f}  std={values.std():.3f}")


def _print_sweep_grid(label: str,
                      opacities: np.ndarray,
                      scale_quantity: np.ndarray,
                      opacity_grid: list[float],
                      scale_grid: list[float]) -> None:
    print(f"\n=== Dead-fraction grid: {label} ===")
    print("    rows = opacity_logit_thresh   (gaussian flagged if opacity < row)")
    print("    cols = log_scale threshold    (gaussian flagged if scale-quantity < col)")
    print("    cell = fraction flagged under (opacity < row) OR (scale < col)\n")

    n = opacities.size
    header = "    op_thresh \\ s_thresh   " + "  ".join(f"{c:>+6.2f}" for c in scale_grid)
    print(header)
    print("    " + "-" * (len(header) - 4))
    for op_t in opacity_grid:
        op_mask = opacities < op_t                                # (N,)
        op_label = "   none " if not np.isfinite(op_t) else f"{op_t:>+7.2f}"
        cells = []
        for sc_t in scale_grid:
            if not np.isfinite(sc_t):
                sc_mask = np.zeros_like(op_mask)
            else:
                sc_mask = scale_quantity < sc_t
            dead_frac = (op_mask | sc_mask).sum() / n
            cells.append(f"{dead_frac * 100:>5.2f}%")
        print(f"    {op_label}             " + "  ".join(cells))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--obj_list", required=True, nargs="+")
    p.add_argument("--gs_path", required=True)
    p.add_argument("--sphere2plane_path", required=True)
    p.add_argument("--exclude_keys_file", default="data/outlier_keys_8sigma.json",
                   help="JSON list of hash_keys to skip (set to '' to disable).")
    p.add_argument("--num_workers", type=int, default=16)
    p.add_argument("--limit", type=int, default=500,
                   help="Number of objects to scan (held in RAM ~128KB each).")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    exclude_keys_file = args.exclude_keys_file or None

    ds = Standard3DGenDataset(
        obj_list=args.obj_list,
        gs_path=args.gs_path,
        mean_file=None,
        std_file=None,
        sphere2plane_path=args.sphere2plane_path,
        exclude_keys_file=exclude_keys_file,
    )
    n_total = len(ds)
    n = min(args.limit, n_total)
    print(f"Scanning {n:,} of {n_total:,} objects with {args.num_workers} workers")
    if exclude_keys_file:
        print(f"Excluding objects listed in: {exclude_keys_file}")

    # Pre-allocate (n, 16384) for each quantity. 16384 = 128*128.
    pc0 = ds[0]["point_cloud"]
    if isinstance(pc0, torch.Tensor):
        pc0 = pc0.numpy()
    n_pixels = pc0.shape[1] * pc0.shape[2]
    print(f"Atlas size: {n_pixels:,} positions/object  →  buffer ~{3 * n * n_pixels * 4 / 1e6:.0f} MB")

    opacities = np.zeros((n, n_pixels), dtype=np.float32)
    max_log_scales = np.zeros((n, n_pixels), dtype=np.float32)
    min_log_scales = np.zeros((n, n_pixels), dtype=np.float32)

    indices = list(range(n))
    t0 = time.time()
    log_every = max(1, n // 50)

    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=args.num_workers,
        initializer=_worker_init,
        initargs=(args.obj_list, args.gs_path, args.sphere2plane_path, exclude_keys_file),
    ) as pool:
        for done, (idx, op, mx, mn) in enumerate(
            pool.imap_unordered(_worker_fn, indices, chunksize=4), 1
        ):
            opacities[idx] = op
            max_log_scales[idx] = mx
            min_log_scales[idx] = mn
            if done % log_every == 0 or done == n:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (n - done) / rate if rate > 0 else 0
                print(f"  {done:>6d}/{n}  {rate:>5.1f} obj/s  eta {eta / 60:>4.1f} min")

    opacities = opacities.reshape(-1)
    max_log_scales = max_log_scales.reshape(-1)
    min_log_scales = min_log_scales.reshape(-1)

    _print_percentile_table("opacity_logit", opacities)
    _print_percentile_table("max(log_scale_xyz)", max_log_scales)
    _print_percentile_table("min(log_scale_xyz)", min_log_scales)

    # Threshold grids — finite values are real thresholds; -inf means "do not flag on this axis".
    opacity_grid = [float("-inf"), -6.0, -4.6, -3.0, -2.0, -1.0]
    scale_grid = [float("-inf"), -8.0, -7.6, -7.0, -6.5, -6.0, -5.0, -4.0]

    _print_sweep_grid(
        "STRICT  (max(log_scale) < col)  -- invisible from every angle",
        opacities, max_log_scales, opacity_grid, scale_grid,
    )
    _print_sweep_grid(
        "LOOSE   (min(log_scale) < col)  -- at least one axis sub-pixel",
        opacities, min_log_scales, opacity_grid, scale_grid,
    )

    # Saturation diagnostic: how many positions are pinned at the load_ply -7.6 floor?
    floor_max = (max_log_scales <= -7.59).sum() / max_log_scales.size
    floor_min = (min_log_scales <= -7.59).sum() / min_log_scales.size
    print(f"\nFraction at -7.6 clip floor:  max-axis pinned={floor_max*100:.2f}%   "
          f"min-axis pinned={floor_min*100:.2f}%")
    print("(load_ply clips log_scale at -7.6 = half-pixel @ 512²; 'pinned' means "
          "the original was at-or-below that floor.)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
