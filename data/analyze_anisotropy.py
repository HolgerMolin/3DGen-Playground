"""How many gaussians are large in 0 / 1 / 2 / 3 axes?

For each atlas position, count how many of the three log-scales (ch 52,53,54)
exceed a "large" threshold. Reports the marginal distribution under several
threshold choices so you can pick the cliff that separates "real geometry"
from "needle/sliver/dead":

  count = 0  → dead, sub-pixel from every angle
  count = 1  → needle (1D structure; visible from views perpendicular to the long axis)
  count = 2  → sliver / pancake (2D structure; visible except edge-on)
  count = 3  → blob (full 3D extent)

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
    log_scales = pc[list(SCALE_CHANNELS)].reshape(3, -1)   # (3, N)
    return idx, log_scales


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--obj_list", required=True, nargs="+")
    p.add_argument("--gs_path", required=True)
    p.add_argument("--sphere2plane_path", required=True)
    p.add_argument("--exclude_keys_file", default="data/outlier_keys_8sigma.json")
    p.add_argument("--num_workers", type=int, default=16)
    p.add_argument("--limit", type=int, default=500)
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

    pc0 = ds[0]["point_cloud"]
    if isinstance(pc0, torch.Tensor):
        pc0 = pc0.numpy()
    n_pixels = pc0.shape[1] * pc0.shape[2]
    buf = np.zeros((n, 3, n_pixels), dtype=np.float32)
    print(f"Buffer ~{buf.nbytes / 1e6:.0f} MB")

    indices = list(range(n))
    t0 = time.time()
    log_every = max(1, n // 50)
    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=args.num_workers,
        initializer=_worker_init,
        initargs=(args.obj_list, args.gs_path, args.sphere2plane_path, exclude_keys_file),
    ) as pool:
        for done, (idx, scales) in enumerate(
            pool.imap_unordered(_worker_fn, indices, chunksize=4), 1
        ):
            buf[idx] = scales
            if done % log_every == 0 or done == n:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (n - done) / rate if rate > 0 else 0
                print(f"  {done:>6d}/{n}  {rate:>5.1f} obj/s  eta {eta / 60:>4.1f} min")

    flat = buf.reshape(3, -1)                              # (3, n*N)
    total = flat.shape[1]

    print(f"\nTotal atlas positions analyzed: {total:,}")
    print("\n=== How many of the 3 axes are 'large' (i.e. log_scale >= threshold)? ===\n")

    thresholds = [-7.0, -6.5, -6.0, -5.5, -5.0, -4.5]
    print(f"  large_thresh  |   0 axes   |   1 axis    |   2 axes    |   3 axes    |")
    print(f"                | (dead)     | (needle)    | (sliver)    | (blob)      |")
    print(f"  --------------+------------+-------------+-------------+-------------+")
    for t in thresholds:
        large = flat >= t                                  # (3, total)
        n_large = large.sum(axis=0)                        # (total,) in {0,1,2,3}
        c = np.bincount(n_large, minlength=4)
        pct = c / total * 100
        line = (f"  {t:>+5.1f}        | {pct[0]:>6.2f}%   | {pct[1]:>6.2f}%    | "
                f"{pct[2]:>6.2f}%    | {pct[3]:>6.2f}%    |")
        print(line)

    print("\nReading: rows = where you draw the 'large' cutoff. Each cell is the % of "
          "atlas positions whose count of large-axes equals the column.")
    print("Needles ('large in only one direction') = column '1 axis'.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
