"""Recompute per-channel mean/std on the post-fix dataset.

Loads each (un-excluded) object through the now-fixed `load_ply`
(quaternion unit-norm + sign canon, xyz clip at ±3, log-scale clip at -7.6),
streams running sum / sum-of-squares per channel in float64, and writes
flat (C,) tensors to disk.

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


_DATASET: Optional[Standard3DGenDataset] = None


def _worker_init(obj_list, gs_path, sphere2plane_path, exclude_keys_file):
    global _DATASET
    logging.getLogger().setLevel(logging.WARNING)
    # Pass mean/std=None so __getitem__ returns raw (un-normalized) data.
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
    if pc.ndim == 3:
        flat = pc.reshape(pc.shape[0], -1)              # (C, H*W)
    elif pc.ndim == 2:
        flat = pc.T                                     # (C, N)
    else:
        raise RuntimeError(f"unexpected pc shape {pc.shape}")
    flat64 = flat.astype(np.float64)
    return flat64.sum(axis=1), (flat64 ** 2).sum(axis=1), flat.shape[1]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--obj_list", required=True, nargs="+")
    p.add_argument("--gs_path", required=True)
    p.add_argument("--sphere2plane_path", required=True)
    p.add_argument("--exclude_keys_file", default="data/outlier_keys_8sigma.json",
                   help="Path to JSON list of hash_keys to drop (set to '' to disable).")
    p.add_argument("--out_mean", default="data/stats/all_mean_postfix.pt")
    p.add_argument("--out_std", default="data/stats/all_std_postfix.pt")
    p.add_argument("--num_workers", type=int, default=24)
    p.add_argument("--limit", type=int, default=None,
                   help="Optional cap on objects scanned (testing only).")
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
    n = n_total if args.limit is None else min(args.limit, n_total)
    print(f"Computing stats over {n:,} objects (post-exclusion); workers={args.num_workers}")
    if exclude_keys_file:
        print(f"Excluding objects listed in: {exclude_keys_file}")
    print("Source: Standard3DGenDataset → load_ply (post-fix transformations applied)")

    sample0 = ds[0]
    pc0 = sample0["point_cloud"]
    if isinstance(pc0, torch.Tensor):
        pc0 = pc0.numpy()
    n_channels = pc0.shape[0] if pc0.ndim == 3 else pc0.shape[-1]
    print(f"Channels: {n_channels}")

    total_sum = np.zeros(n_channels, dtype=np.float64)
    total_sumsq = np.zeros(n_channels, dtype=np.float64)
    total_count = 0

    indices = list(range(n))
    t0 = time.time()
    log_every = max(1, n // 100)

    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=args.num_workers,
        initializer=_worker_init,
        initargs=(args.obj_list, args.gs_path, args.sphere2plane_path, exclude_keys_file),
    ) as pool:
        for done, (sums, sumsqs, count) in enumerate(
            pool.imap_unordered(_worker_fn, indices, chunksize=4), 1
        ):
            total_sum += sums
            total_sumsq += sumsqs
            total_count += count
            if done % log_every == 0 or done == n:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (n - done) / rate if rate > 0 else 0
                print(f"  {done:>7d}/{n}  {rate:>5.1f} obj/s  eta {eta / 60:>5.1f} min")

    mean = total_sum / total_count
    var = total_sumsq / total_count - mean ** 2
    std = np.sqrt(np.maximum(var, 0.0))
    print(f"\nTotal points scanned per channel: {total_count:,}")

    DC_INDICES = [0, 1, 2, 3, 4, 20, 36, 52, 53, 54, 55, 56, 57, 58]
    LABELS = {0:"x",1:"y",2:"z",3:"opacity",4:"sh_dc_r",20:"sh_dc_g",36:"sh_dc_b",
              52:"scale_x",53:"scale_y",54:"scale_z",55:"rot_w",56:"rot_x",57:"rot_y",58:"rot_z"}
    print("\nNew per-channel stats (DC-only channels shown):")
    print(f"  ch  label        mean         std")
    for c in DC_INDICES:
        print(f"  {c:>2}  {LABELS[c]:>9}   {mean[c]:+9.4f}   {std[c]:8.4f}")

    out_mean = Path(args.out_mean)
    out_std = Path(args.out_std)
    out_mean.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.from_numpy(mean.astype(np.float32)), out_mean)
    torch.save(torch.from_numpy(std.astype(np.float32)), out_std)
    print(f"\nWrote {out_mean}")
    print(f"Wrote {out_std}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
