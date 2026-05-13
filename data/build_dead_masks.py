"""Build per-object binary masks marking atlas positions whose gaussian cannot
contribute meaningful pixels under any view.

A gaussian is flagged "dead" when EITHER:
  - opacity (raw logit, ch 3) < opacity_logit_thresh
        → sigmoid(logit) below the contribution floor regardless of scale.
  - max(log-scale over xyz, ch 52-54) < max_log_scale_thresh
        → world-extent is sub-pixel from every view direction (max scale is
          the largest projected axis under any rotation).

Default thresholds:
  --opacity_logit_thresh = -4.6   ≈ sigmoid 0.01  (≤ 1% alpha contribution)
  --max_log_scale_thresh = -6.9   ≈ exp(-6.9) ≈ 1e-3 world units, ~1 pixel
                                    at 512² render (load_ply already clips
                                    the dead tail at -7.6 = half-pixel).

Output: one .npy per object at OUTPUT_DIR/{hash_key}.npy, shape (128, 128) bool.
True means "dead — safe to mask in non-opacity channels". Idempotent:
existing files are skipped so the script can be resumed.

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
_OUTPUT_DIR: Optional[Path] = None
_OPACITY_THRESH: float = 0.0
_MAX_LOG_SCALE_THRESH: float = 0.0


def _worker_init(obj_list, gs_path, sphere2plane_path, exclude_keys_file,
                 output_dir, opacity_thresh, max_log_scale_thresh):
    global _DATASET, _OUTPUT_DIR, _OPACITY_THRESH, _MAX_LOG_SCALE_THRESH
    logging.getLogger().setLevel(logging.WARNING)
    _DATASET = Standard3DGenDataset(
        obj_list=obj_list,
        gs_path=gs_path,
        mean_file=None,
        std_file=None,
        sphere2plane_path=sphere2plane_path,
        exclude_keys_file=exclude_keys_file,
    )
    _OUTPUT_DIR = Path(output_dir)
    _OPACITY_THRESH = float(opacity_thresh)
    _MAX_LOG_SCALE_THRESH = float(max_log_scale_thresh)


def _worker_fn(idx: int):
    hash_key = _DATASET.keys[idx]
    out_path = _OUTPUT_DIR / f"{hash_key}.npy"
    if out_path.exists():
        existing = np.load(out_path)
        return idx, int(existing.sum()), int(existing.size), True

    sample = _DATASET[idx]
    pc = sample["point_cloud"]
    if isinstance(pc, torch.Tensor):
        pc = pc.detach().cpu().numpy()
    pc = pc.astype(np.float32)
    if pc.ndim != 3:
        raise RuntimeError(f"expected (C, H, W) point cloud, got shape {pc.shape}")

    opacity = pc[OPACITY_CHANNEL]                          # (H, W)
    log_scales = pc[list(SCALE_CHANNELS)]                  # (3, H, W)
    max_log_scale = log_scales.max(axis=0)                 # (H, W)

    dead = (opacity < _OPACITY_THRESH) | (max_log_scale < _MAX_LOG_SCALE_THRESH)
    np.save(out_path, dead)
    return idx, int(dead.sum()), int(dead.size), False


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--obj_list", required=True, nargs="+")
    p.add_argument("--gs_path", required=True)
    p.add_argument("--sphere2plane_path", required=True)
    p.add_argument("--exclude_keys_file", default="data/outlier_keys_8sigma.json",
                   help="JSON list of hash_keys to skip (set to '' to disable).")
    p.add_argument("--output_dir", default="data/dead_masks",
                   help="Directory to write per-object {hash_key}.npy masks.")
    p.add_argument("--opacity_logit_thresh", type=float, default=-4.6,
                   help="Pre-sigmoid opacity threshold; below this the gaussian "
                        "alpha is < ~1%% regardless of size.")
    p.add_argument("--max_log_scale_thresh", type=float, default=-6.9,
                   help="Threshold on max(log-scale over xyz); below this the "
                        "gaussian is sub-pixel at 512² render from every view.")
    p.add_argument("--num_workers", type=int, default=24)
    p.add_argument("--limit", type=int, default=None,
                   help="Optional cap on objects scanned (testing only).")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    exclude_keys_file = args.exclude_keys_file or None

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ds = Standard3DGenDataset(
        obj_list=args.obj_list,
        gs_path=args.gs_path,
        mean_file=None,
        std_file=None,
        sphere2plane_path=args.sphere2plane_path,
        exclude_keys_file=exclude_keys_file,
    )
    n_total = len(ds)
    n = n_total if args.limit is None else min(args.limit, n_total)
    print(f"Building dead masks for {n:,} objects (post-exclusion); workers={args.num_workers}")
    print(f"Output dir: {output_dir}")
    print(f"Thresholds: opacity_logit < {args.opacity_logit_thresh}  OR  "
          f"max(log_scale) < {args.max_log_scale_thresh}")
    if exclude_keys_file:
        print(f"Excluding objects listed in: {exclude_keys_file}")

    indices = list(range(n))
    t0 = time.time()
    log_every = max(1, n // 100)

    total_dead = 0
    total_pixels = 0
    skipped = 0
    per_obj_frac = np.zeros(n, dtype=np.float32)

    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=args.num_workers,
        initializer=_worker_init,
        initargs=(args.obj_list, args.gs_path, args.sphere2plane_path, exclude_keys_file,
                  str(output_dir), args.opacity_logit_thresh, args.max_log_scale_thresh),
    ) as pool:
        for done, (idx, n_dead, n_total_pixels, was_skipped) in enumerate(
            pool.imap_unordered(_worker_fn, indices, chunksize=4), 1
        ):
            total_dead += n_dead
            total_pixels += n_total_pixels
            per_obj_frac[idx] = n_dead / n_total_pixels
            if was_skipped:
                skipped += 1
            if done % log_every == 0 or done == n:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (n - done) / rate if rate > 0 else 0
                running_frac = total_dead / max(total_pixels, 1)
                print(
                    f"  {done:>7d}/{n}  {rate:>5.1f} obj/s  eta {eta / 60:>5.1f} min  "
                    f"running dead-frac={running_frac:.3f}  skipped(existing)={skipped}"
                )

    fracs = per_obj_frac[:n]
    print(f"\n=== Done. Aggregate: {total_dead:,} / {total_pixels:,} pixels dead "
          f"({100 * total_dead / max(total_pixels, 1):.2f}%) ===")
    print(f"Per-object dead-fraction distribution:")
    for q in (0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99):
        print(f"  p{int(q*100):>2d}: {np.quantile(fracs, q):.3f}")
    print(f"  min: {fracs.min():.3f}   mean: {fracs.mean():.3f}   max: {fracs.max():.3f}")
    print(f"Wrote masks to {output_dir}  ({skipped} pre-existing skipped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
