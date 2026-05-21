"""Build per-channel Gaussian rank-transform tables for non-Gaussian channels.

For each target channel (default: opacity, scale_x, scale_y, scale_z), reservoir-
samples raw values across all (post-exclusion) objects in the un-normalized data
stream, then computes K evenly-spaced empirical quantiles. Saves a payload with:

    channels         : list[int]     -- channel indices, length C
    data_quantiles   : (C, K) float32 -- per-channel data-domain quantile values
    gauss_quantiles  : (K,)  float32 -- shared standard-normal quantile values
    num_quantiles    : int           -- K
    samples_per_channel : int        -- reservoir size used per channel

Forward transform (data -> N(0,1)):  np.interp(x, data_quantiles[c], gauss_quantiles)
Inverse transform (N(0,1) -> data):  np.interp(z, gauss_quantiles, data_quantiles[c])

Both rely on strictly-increasing breakpoints; this script enforces strict
monotonicity by nudging exact ties with the smallest float32 step.

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

DEFAULT_CHANNELS = (3, 52, 53, 54)
CHANNEL_LABELS = {
    3: "opacity",
    4: "sh_dc_r",
    20: "sh_dc_g",
    36: "sh_dc_b",
    52: "scale_x",
    53: "scale_y",
    54: "scale_z",
    55: "rot_w",
}

_DATASET: Optional[Standard3DGenDataset] = None
_TARGET_CHANNELS: Optional[tuple] = None
_SAMPLES_PER_OBJECT: Optional[int] = None


def _worker_init(obj_list, gs_path, sphere2plane_path, exclude_keys_file,
                 channels, samples_per_object, clip_thresholds_file):
    global _DATASET, _TARGET_CHANNELS, _SAMPLES_PER_OBJECT
    logging.getLogger().setLevel(logging.WARNING)
    _DATASET = Standard3DGenDataset(
        obj_list=obj_list,
        gs_path=gs_path,
        mean_file=None,
        std_file=None,
        sphere2plane_path=sphere2plane_path,
        exclude_keys_file=exclude_keys_file,
        clip_thresholds_file=clip_thresholds_file,
    )
    _TARGET_CHANNELS = channels
    _SAMPLES_PER_OBJECT = samples_per_object


def _worker_fn(idx: int):
    sample = _DATASET[idx]
    pc = sample["point_cloud"]
    if isinstance(pc, torch.Tensor):
        pc = pc.detach().cpu().numpy()
    pc = pc.astype(np.float32, copy=False)
    if pc.ndim == 3:
        flat = pc.reshape(pc.shape[0], -1)              # (C, H*W)
    elif pc.ndim == 2:
        flat = pc.T                                     # (C, N)
    else:
        raise RuntimeError(f"unexpected pc shape {pc.shape}")

    n_pts = flat.shape[1]
    rng = np.random.default_rng(seed=idx)
    out = []
    for c in _TARGET_CHANNELS:
        if _SAMPLES_PER_OBJECT >= n_pts:
            picks = flat[c]
        else:
            sel = rng.choice(n_pts, size=_SAMPLES_PER_OBJECT, replace=False)
            picks = flat[c, sel]
        out.append(picks.astype(np.float32, copy=False))
    return out


def _build_quantiles(samples: np.ndarray, K: int) -> np.ndarray:
    """Take K evenly-spaced empirical quantiles at positions (k + 0.5) / K."""
    samples = np.sort(samples)
    qs = (np.arange(K, dtype=np.float64) + 0.5) / K
    idx = (qs * (samples.size - 1)).round().astype(np.int64)
    return samples[idx].astype(np.float32)


def _enforce_strict_monotonic(arr: np.ndarray) -> np.ndarray:
    """Bump tied breakpoints upward by one float32 ULP so np.interp is well-defined.

    Only modifies entries where arr[i] <= arr[i-1]; intact entries are unchanged.
    """
    out = arr.astype(np.float32, copy=True)
    for i in range(1, out.size):
        if out[i] <= out[i - 1]:
            out[i] = np.nextafter(out[i - 1], np.float32(np.inf))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--obj_list", required=True, nargs="+")
    p.add_argument("--gs_path", required=True)
    p.add_argument("--sphere2plane_path", required=True)
    p.add_argument("--exclude_keys_file", default="data/outlier_keys_8sigma.json",
                   help="Path to JSON list of hash_keys to drop (set to '' to disable).")
    p.add_argument("--clip_thresholds_file", default=None,
                   help="Optional clip-thresholds payload (data/build_clip_thresholds.py). "
                        "When set, listed channels are hard-clipped BEFORE quantiles are "
                        "sampled, so the rank tables are built on the clipped distribution.")
    p.add_argument("--out", default="data/stats/rank_quantiles.pt")
    p.add_argument("--channels", type=int, nargs="+", default=list(DEFAULT_CHANNELS),
                   help="Channel indices to build a rank transform for.")
    p.add_argument("--num_quantiles", type=int, default=4096,
                   help="Number of empirical quantiles per channel (K).")
    p.add_argument("--samples_per_object", type=int, default=80,
                   help="Reservoir samples per object per channel.")
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
        clip_thresholds_file=args.clip_thresholds_file,
    )
    n_total = len(ds)
    n = n_total if args.limit is None else min(args.limit, n_total)

    channel_labels = [CHANNEL_LABELS.get(c, f"ch{c}") for c in args.channels]
    print(f"Building rank transform over {n:,} objects (post-exclusion); workers={args.num_workers}")
    print(f"Channels: {args.channels} ({channel_labels})")
    print(f"K = {args.num_quantiles} quantiles, {args.samples_per_object} samples/object/channel")
    print(f"Total samples per channel ~= {n * args.samples_per_object:,}")
    if exclude_keys_file:
        print(f"Excluding objects listed in: {exclude_keys_file}")

    per_channel_buffers = [[] for _ in args.channels]

    indices = list(range(n))
    t0 = time.time()
    log_every = max(1, n // 100)

    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=args.num_workers,
        initializer=_worker_init,
        initargs=(args.obj_list, args.gs_path, args.sphere2plane_path,
                  exclude_keys_file, tuple(args.channels), args.samples_per_object,
                  args.clip_thresholds_file),
    ) as pool:
        for done, picks_per_channel in enumerate(
            pool.imap_unordered(_worker_fn, indices, chunksize=4), 1
        ):
            for buf, picks in zip(per_channel_buffers, picks_per_channel):
                buf.append(picks)
            if done % log_every == 0 or done == n:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (n - done) / rate if rate > 0 else 0
                print(f"  {done:>7d}/{n}  {rate:>5.1f} obj/s  eta {eta / 60:>5.1f} min")

    print("\nConcatenating per-channel buffers ...")
    per_channel_arrays = [np.concatenate(buf) for buf in per_channel_buffers]
    for c, label, arr in zip(args.channels, channel_labels, per_channel_arrays):
        print(f"  ch {c:>2} ({label:>9}): {arr.size:,} samples, "
              f"min={arr.min():+.4f} max={arr.max():+.4f} "
              f"mean={arr.mean():+.4f} std={arr.std():.4f}")

    K = args.num_quantiles
    print(f"\nBuilding K={K} quantile breakpoints per channel ...")
    raw_quantiles = [_build_quantiles(arr, K) for arr in per_channel_arrays]
    data_quantiles = np.stack(
        [_enforce_strict_monotonic(q) for q in raw_quantiles],
        axis=0,
    )  # (C, K) float32

    qs = (np.arange(K, dtype=np.float64) + 0.5) / K
    gauss_quantiles = torch.special.ndtri(torch.from_numpy(qs)).float().numpy()

    print("\nPer-channel breakpoint diagnostics:")
    for c, label, raw_q, dq in zip(args.channels, channel_labels, raw_quantiles, data_quantiles):
        ties = K - np.unique(raw_q).size
        bumped = int((dq != raw_q).sum())
        print(f"  ch {c:>2} ({label:>9}): {ties} ties (pre-bump), {bumped} bumped, "
              f"data range [{dq[0]:+.4f}, {dq[-1]:+.4f}]")
    print(f"  gauss range [{gauss_quantiles[0]:+.4f}, {gauss_quantiles[-1]:+.4f}]")

    print("\nRound-trip + N(0,1) check (up to 1M samples per channel):")
    n_test = 1_000_000
    for ci, (c, label) in enumerate(zip(args.channels, channel_labels)):
        arr = per_channel_arrays[ci]
        test_arr = arr[:n_test] if arr.size >= n_test else arr
        dq = data_quantiles[ci]
        fwd = np.interp(test_arr, dq, gauss_quantiles)
        inv = np.interp(fwd, gauss_quantiles, dq)
        in_range = (test_arr >= dq[0]) & (test_arr <= dq[-1])
        if in_range.any():
            err = np.abs(test_arr[in_range] - inv[in_range])
            err_str = f"max-err {err.max():.4g}"
        else:
            err_str = "max-err n/a"
        print(f"  ch {c:>2} ({label:>9}): "
              f"in-range {in_range.mean()*100:5.2f}%, {err_str}, "
              f"forward N(0,1): mean={fwd.mean():+.4f} std={fwd.std():.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "channels": list(args.channels),
        "data_quantiles": torch.from_numpy(data_quantiles),
        "gauss_quantiles": torch.from_numpy(gauss_quantiles),
        "num_quantiles": K,
        "samples_per_channel": int(per_channel_arrays[0].size),
        "exclude_keys_file": exclude_keys_file or "",
        "clip_thresholds_file": args.clip_thresholds_file or "",
        "obj_list": list(args.obj_list),
    }
    torch.save(payload, out_path)
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
