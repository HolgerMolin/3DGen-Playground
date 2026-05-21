"""Find per-channel clip thresholds (lower/upper percentiles) for every channel.

Runs through the (post-exclusion) un-normalized data stream, reservoir-samples
raw values across all objects for each of the 59 channels, then computes a lower
and an upper percentile (default 5th / 95th). Saves a payload with:

    channels       : list[int]   -- channel indices, length C
    lower          : (C,) float32 -- per-channel lower-percentile threshold
    upper          : (C,) float32 -- per-channel upper-percentile threshold
    lower_pct      : float        -- lower percentile used (e.g. 5.0)
    upper_pct      : float        -- upper percentile used (e.g. 95.0)
    labels         : list[str]    -- human-readable channel names
    samples_per_channel : int     -- reservoir size used per channel

These thresholds live in the RAW data domain (the values returned by
`load_ply`, i.e. before per-channel mean/std normalization and before any
Gaussian rank transform). To use them, clip the point cloud right after it is
loaded — mirroring the existing `xyz = np.clip(xyz, -3.0, 3.0)` in load_ply —
e.g.  `pc = np.clip(pc, lower[None, :], upper[None, :])`.

Note: load_ply already clips xyz to +-3 and unit-normalizes rotations, so those
channels' thresholds reflect that pre-processing.

Run from repo root with the project venv activated.
"""

from __future__ import annotations

import argparse
import json
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
from dataloaders.class_3dgen_loader import DC_ONLY_FEATURE_INDICES

# Full 59-channel layout produced by load_ply:
#   0-2   xyz
#   3     opacity
#   4-51  SH features, color-major: 16 coeffs per color (r, g, b); coeff 0 = DC
#   52-54 scale_x/y/z
#   55-58 rot_w/x/y/z
# Channel indices in the saved payload are always in this 59-dim grid space, so
# they line up with DC_ONLY_FEATURE_INDICES and with how mean/std files are
# sliced via feature_indices in train_gsplat.py.
N_CHANNELS = 59


def _default_labels() -> dict:
    labels = {0: "x", 1: "y", 2: "z", 3: "opacity"}
    for ci, col in enumerate(("r", "g", "b")):
        base = 4 + ci * 16
        labels[base] = f"sh_dc_{col}"
        for k in range(1, 16):
            labels[base + k] = f"sh_{col}_{k:02d}"
    labels[52], labels[53], labels[54] = "scale_x", "scale_y", "scale_z"
    labels[55], labels[56], labels[57], labels[58] = "rot_w", "rot_x", "rot_y", "rot_z"
    return labels


CHANNEL_LABELS = _default_labels()

_DATASET: Optional[Standard3DGenDataset] = None
_TARGET_CHANNELS: Optional[tuple] = None
_SAMPLES_PER_OBJECT: Optional[int] = None


def _worker_init(obj_list, gs_path, sphere2plane_path, exclude_keys_file,
                 channels, samples_per_object):
    global _DATASET, _TARGET_CHANNELS, _SAMPLES_PER_OBJECT
    logging.getLogger().setLevel(logging.WARNING)
    _DATASET = Standard3DGenDataset(
        obj_list=obj_list,
        gs_path=gs_path,
        mean_file=None,           # raw data domain: no normalization
        std_file=None,
        sphere2plane_path=sphere2plane_path,
        exclude_keys_file=exclude_keys_file,
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
    if _SAMPLES_PER_OBJECT >= n_pts:
        sel = slice(None)
    else:
        sel = rng.choice(n_pts, size=_SAMPLES_PER_OBJECT, replace=False)
    return [flat[c, sel].astype(np.float32, copy=False) for c in _TARGET_CHANNELS]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--obj_list", required=True, nargs="+")
    p.add_argument("--gs_path", required=True)
    p.add_argument("--sphere2plane_path", required=True)
    p.add_argument("--exclude_keys_file", default="data/outlier_keys_8sigma.json",
                   help="JSON list of hash_keys to drop (set to '' to disable). "
                        "Defaults to the same exclusion used during training so "
                        "the thresholds match the training distribution.")
    p.add_argument("--out", default="data/stats/clip_thresholds.pt")
    p.add_argument("--dc_only", action=argparse.BooleanOptionalAction, default=True,
                   help="Only compute thresholds for the 14 DC-only channels the "
                        "model trains on under --sh_degree0_only (default). Pass "
                        "--no-dc_only to cover all 59 channels.")
    p.add_argument("--channels", type=int, nargs="+", default=None,
                   help="Explicit channel indices (59-dim grid space) to compute "
                        "thresholds for. Overrides --dc_only when given.")
    p.add_argument("--lower_pct", type=float, default=5.0,
                   help="Lower clip percentile (default 5.0 = bottom 5%%).")
    p.add_argument("--upper_pct", type=float, default=95.0,
                   help="Upper clip percentile (default 95.0 = top 95%%).")
    p.add_argument("--samples_per_object", type=int, default=64,
                   help="Reservoir samples per object per channel. ~64 over the "
                        "full set gives millions of samples per channel, plenty "
                        "for stable 5/95 percentiles while bounding memory.")
    p.add_argument("--num_workers", type=int, default=24)
    p.add_argument("--limit", type=int, default=None,
                   help="Optional cap on objects scanned (testing only).")
    args = p.parse_args()

    if not (0.0 <= args.lower_pct < args.upper_pct <= 100.0):
        p.error("require 0 <= lower_pct < upper_pct <= 100")

    # Channel selection: explicit --channels wins; else DC-only (default) or all 59.
    if args.channels is not None:
        channels = list(args.channels)
    elif args.dc_only:
        channels = list(DC_ONLY_FEATURE_INDICES)
    else:
        channels = list(range(N_CHANNELS))
    args.channels = channels

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

    channel_labels = [CHANNEL_LABELS.get(c, f"ch{c}") for c in args.channels]
    print(f"Finding clip thresholds over {n:,} objects (post-exclusion); workers={args.num_workers}")
    print(f"Percentiles: lower={args.lower_pct}%  upper={args.upper_pct}%")
    mode = "DC-only (sh_degree0_only)" if args.dc_only and len(args.channels) == len(DC_ONLY_FEATURE_INDICES) else "custom/all"
    if len(args.channels) <= 16:
        print(f"Channels [{mode}]: {list(zip(args.channels, channel_labels))}")
    else:
        print(f"Channels [{mode}]: {len(args.channels)} "
              f"(e.g. {channel_labels[:4]} ... {channel_labels[-4:]})")
    print(f"{args.samples_per_object} samples/object/channel "
          f"-> ~{n * args.samples_per_object:,} samples/channel")
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
                  exclude_keys_file, tuple(args.channels), args.samples_per_object),
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

    lower = np.empty(len(args.channels), dtype=np.float32)
    upper = np.empty(len(args.channels), dtype=np.float32)
    print(f"\nPer-channel thresholds (p{args.lower_pct:g} / p{args.upper_pct:g}):")
    print(f"  {'ch':>3} {'label':>9} {'samples':>12} "
          f"{'min':>10} {'lower':>10} {'upper':>10} {'max':>10}  {'%clipped':>8}")
    for i, (c, label) in enumerate(zip(args.channels, channel_labels)):
        arr = per_channel_arrays[i]
        lo, hi = np.percentile(arr, [args.lower_pct, args.upper_pct])
        lower[i], upper[i] = np.float32(lo), np.float32(hi)
        frac = float(((arr < lo) | (arr > hi)).mean()) * 100.0
        print(f"  {c:>3} {label:>9} {arr.size:>12,} "
              f"{arr.min():>10.4f} {lo:>10.4f} {hi:>10.4f} {arr.max():>10.4f}  {frac:>7.2f}%")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "channels": list(args.channels),
        "lower": torch.from_numpy(lower),
        "upper": torch.from_numpy(upper),
        "lower_pct": float(args.lower_pct),
        "upper_pct": float(args.upper_pct),
        "labels": channel_labels,
        "samples_per_channel": int(per_channel_arrays[0].size),
        "exclude_keys_file": exclude_keys_file or "",
        "obj_list": list(args.obj_list),
    }
    torch.save(payload, out_path)

    # Human-readable sidecar for quick inspection / diffing.
    json_path = out_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump({
            "lower_pct": float(args.lower_pct),
            "upper_pct": float(args.upper_pct),
            "samples_per_channel": int(per_channel_arrays[0].size),
            "exclude_keys_file": exclude_keys_file or "",
            "thresholds": {
                str(c): {"label": lbl, "lower": float(lo), "upper": float(hi)}
                for c, lbl, lo, hi in zip(args.channels, channel_labels, lower, upper)
            },
        }, f, indent=2)

    print(f"\nWrote {out_path}")
    print(f"Wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
