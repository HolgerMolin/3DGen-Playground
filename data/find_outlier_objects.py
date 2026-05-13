"""Scan the full dataset and write a JSON list of objects whose residual-channel
max-abs (post-standardization, post-load_ply fixes) exceeds a threshold.

Residual channels = xyz (0,1,2), opacity (3), sh_dc_{r,g,b} (4, 20, 36).
These are unaffected by the in-loader quaternion-norm and xyz-clip fixes, so
existing mean.pt / std.pt files give valid sigma units for them.

Output JSON is a flat list of hash_keys to exclude. Pass this file to
`Standard3DGenDataset(..., exclude_keys_file=PATH)` to drop them at training.

Run from repo root with the project venv activated.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
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


RESIDUAL_INDICES = np.array([0, 1, 2, 3, 4, 20, 36], dtype=np.int64)
RESIDUAL_LABELS = ["x", "y", "z", "opacity", "sh_dc_r", "sh_dc_g", "sh_dc_b"]


_DATASET: Optional[Standard3DGenDataset] = None


def _worker_init(obj_list, gs_path, mean_file, std_file, sphere2plane_path):
    global _DATASET
    logging.getLogger().setLevel(logging.WARNING)
    _DATASET = Standard3DGenDataset(
        obj_list=obj_list,
        gs_path=gs_path,
        mean_file=mean_file,
        std_file=std_file,
        sphere2plane_path=sphere2plane_path,
    )


def _worker_fn(idx: int) -> tuple[int, np.ndarray]:
    sample = _DATASET[idx]
    pc = sample["point_cloud"]
    if isinstance(pc, torch.Tensor):
        pc = pc.detach().cpu().numpy()
    pc = pc.astype(np.float32)
    if pc.ndim == 3:
        sub = pc[RESIDUAL_INDICES].reshape(len(RESIDUAL_INDICES), -1)
    elif pc.ndim == 2:
        sub = pc[:, RESIDUAL_INDICES].T
    else:
        raise RuntimeError(f"unexpected pc shape {pc.shape}")
    return idx, np.abs(sub).max(axis=1).astype(np.float32)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--obj_list", required=True, nargs="+")
    p.add_argument("--gs_path", required=True)
    p.add_argument("--mean_file", required=True)
    p.add_argument("--std_file", required=True)
    p.add_argument("--sphere2plane_path", required=True)
    p.add_argument("--threshold", type=float, default=8.0,
                   help="Discard objects whose max-abs over residual channels exceeds this many sigma.")
    p.add_argument("--num_workers", type=int, default=16)
    p.add_argument("--output", default="data/outlier_keys_8sigma.json")
    p.add_argument("--limit", type=int, default=None,
                   help="Optional cap on how many objects to scan (for testing).")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    ds = Standard3DGenDataset(
        obj_list=args.obj_list,
        gs_path=args.gs_path,
        mean_file=args.mean_file,
        std_file=args.std_file,
        sphere2plane_path=args.sphere2plane_path,
    )
    n_total = len(ds)
    n = n_total if args.limit is None else min(args.limit, n_total)
    print(f"Dataset size: {n_total:,}  scanning {n:,} with {args.num_workers} workers")
    print(f"Threshold: any residual-channel value > {args.threshold}σ flags the object")
    print(f"Residual channels: {RESIDUAL_LABELS}")

    keys_in_order = ds.keys[:n]

    per_obj_max = np.zeros((n, len(RESIDUAL_INDICES)), dtype=np.float32)
    indices = list(range(n))

    t0 = time.time()
    log_every = max(1, n // 100)

    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=args.num_workers,
        initializer=_worker_init,
        initargs=(args.obj_list, args.gs_path, args.mean_file, args.std_file, args.sphere2plane_path),
    ) as pool:
        for done, (idx, channel_max) in enumerate(
            pool.imap_unordered(_worker_fn, indices, chunksize=4), 1
        ):
            per_obj_max[idx] = channel_max
            if done % log_every == 0 or done == n:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (n - done) / rate if rate > 0 else 0
                running_flagged = (per_obj_max[:done].max(axis=1) > args.threshold).sum()
                print(
                    f"  {done:>7d}/{n}  {rate:>5.1f} obj/s  eta {eta / 60:>5.1f} min  "
                    f"running flagged={running_flagged}"
                )

    per_obj_global_max = per_obj_max.max(axis=1)
    flagged_mask = per_obj_global_max > args.threshold
    flagged_keys = [keys_in_order[i] for i in np.where(flagged_mask)[0]]

    print(f"\n=== Flagged {len(flagged_keys):,} of {n:,} objects "
          f"({100 * len(flagged_keys) / n:.3f}%) above {args.threshold}σ ===")
    for label, ch_pos in zip(RESIDUAL_LABELS, range(len(RESIDUAL_INDICES))):
        ch_flagged = (per_obj_max[:, ch_pos] > args.threshold).sum()
        print(f"  {label:>9s}: {ch_flagged:>6d} objects ({100 * ch_flagged / n:.3f}%)")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(flagged_keys, f, indent=2)
    print(f"Wrote {len(flagged_keys):,} keys to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
