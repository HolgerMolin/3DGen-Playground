"""Report-only round-trip evaluation for an existing Gaussian rank-transform table.

Loads a prebuilt rank_quantiles payload (data/build_rank_transform.py output) and
measures the encode->decode round-trip error  |x - inv(fwd(x))|  on freshly sampled
real channel values, plus the forward-transform N(0,1) quality. Does NOT rebuild the
table. Faithful to the build path: reuses the same dataset sampling and the clip /
exclude settings the table was built with (read from the payload).

    fwd (data -> N(0,1)):  np.interp(x, data_quantiles[c], gauss_quantiles)
    inv (N(0,1) -> data):  np.interp(z, gauss_quantiles, data_quantiles[c])

Run from repo root with the project venv (.3dgen) activated.
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Reuse the exact worker sampling logic the builder used.
from data.build_rank_transform import _worker_init, _worker_fn, CHANNEL_LABELS
from dataloaders.standard_3dgen_loader import Standard3DGenDataset


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rank_file", default="data/stats/rank_quantiles_8ch_clipped.pt")
    p.add_argument("--gs_path", default="data/gaussianverse/")
    p.add_argument("--sphere2plane_path",
                   default="data/gaussianverse/sphere2plane.npy")
    p.add_argument("--obj_list", nargs="+", default=None,
                   help="Override obj_list; default = the one stored in the payload.")
    p.add_argument("--samples_per_object", type=int, default=80)
    p.add_argument("--num_workers", type=int, default=24)
    p.add_argument("--limit", type=int, default=4000,
                   help="Objects to sample for the round-trip test (random subset).")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    payload = torch.load(args.rank_file, map_location="cpu", weights_only=False)
    channels = list(payload["channels"])
    data_quantiles = payload["data_quantiles"].numpy().astype(np.float32)   # (C, K)
    gauss_quantiles = payload["gauss_quantiles"].numpy().astype(np.float32)  # (K,)
    K = int(payload.get("num_quantiles", gauss_quantiles.size))
    clip_thresholds_file = payload.get("clip_thresholds_file", "") or None
    exclude_keys_file = payload.get("exclude_keys_file", "") or None
    obj_list = args.obj_list or list(payload["obj_list"])
    labels = [CHANNEL_LABELS.get(c, f"ch{c}") for c in channels]

    print(f"Loaded {args.rank_file}")
    print(f"  channels         : {channels} ({labels})")
    print(f"  K (quantiles)    : {K}")
    print(f"  built on samples : {payload.get('samples_per_channel', '?'):,}/channel")
    print(f"  clip_thresholds  : {clip_thresholds_file}")
    print(f"  exclude_keys     : {exclude_keys_file}")
    print(f"  obj_list         : {obj_list}")

    # ---- table self-consistency (data-independent): is fwd a proper inverse of inv on the knots? ----
    print("\n[A] Table self-check on the K knots (inv(fwd(knot)) vs knot):")
    for ci, (c, lab) in enumerate(zip(channels, labels)):
        dq = data_quantiles[ci]
        mono = bool(np.all(np.diff(dq) > 0))
        rt = np.interp(np.interp(dq, dq, gauss_quantiles), gauss_quantiles, dq)
        self_err = np.abs(dq - rt)
        print(f"  ch {c:>2} ({lab:>9}): strictly-increasing={mono}, "
              f"knot round-trip max-err {self_err.max():.4g}, "
              f"range [{dq[0]:+.5f}, {dq[-1]:+.5f}]")

    # ---- data round-trip: sample real values, fwd then inv, compare ----
    ds = Standard3DGenDataset(
        obj_list=obj_list, gs_path=args.gs_path, mean_file=None, std_file=None,
        sphere2plane_path=args.sphere2plane_path,
        exclude_keys_file=exclude_keys_file, clip_thresholds_file=clip_thresholds_file,
    )
    n_total = len(ds)
    n = min(args.limit, n_total)
    rng = np.random.default_rng(args.seed)
    indices = rng.choice(n_total, size=n, replace=False).tolist()
    print(f"\nSampling {n:,}/{n_total:,} objects x {args.samples_per_object} pts/channel "
          f"(workers={args.num_workers}) ...")

    per_channel = [[] for _ in channels]
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=args.num_workers, initializer=_worker_init,
        initargs=(obj_list, args.gs_path, args.sphere2plane_path, exclude_keys_file,
                  tuple(channels), args.samples_per_object, clip_thresholds_file),
    ) as pool:
        for done, picks in enumerate(pool.imap_unordered(_worker_fn, indices, chunksize=4), 1):
            for buf, pk in zip(per_channel, picks):
                buf.append(pk)
            if done % max(1, n // 20) == 0 or done == n:
                print(f"  {done:>7d}/{n}  {done / (time.time() - t0):.1f} obj/s")
    arrays = [np.concatenate(b) for b in per_channel]

    print("\n[B] Data round-trip  |x - inv(fwd(x))|  on real sampled values:")
    print(f"  {'ch':>3} {'label':>9} {'in-range%':>9} {'max-err':>11} {'mean-err':>11} "
          f"{'p99-err':>11} {'fwd-mean':>9} {'fwd-std':>8} {'oor-lo%':>8} {'oor-hi%':>8}")
    for ci, (c, lab) in enumerate(zip(channels, labels)):
        arr = arrays[ci]
        dq = data_quantiles[ci]
        fwd = np.interp(arr, dq, gauss_quantiles)
        inv = np.interp(fwd, gauss_quantiles, dq)
        in_range = (arr >= dq[0]) & (arr <= dq[-1])
        oor_lo = float((arr < dq[0]).mean() * 100)
        oor_hi = float((arr > dq[-1]).mean() * 100)
        if in_range.any():
            err = np.abs(arr[in_range] - inv[in_range])
            mx, mn, p99 = err.max(), err.mean(), np.percentile(err, 99)
        else:
            mx = mn = p99 = float("nan")
        print(f"  {c:>3} {lab:>9} {in_range.mean()*100:8.2f}% {mx:11.4g} {mn:11.4g} "
              f"{p99:11.4g} {fwd.mean():+9.4f} {fwd.std():8.4f} {oor_lo:7.3f}% {oor_hi:7.3f}%")

    print(f"\n  ({arrays[0].size:,} samples/channel; round-trip err is in raw channel units, "
          f"i.e. logit-opacity / log-scale space)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
