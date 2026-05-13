"""Estimate how many objects in the dataset are outlier-contaminated.

For a random subset of objects, computes per-channel max-abs over the 14 DC
channels (post-normalization), then reports the fraction with any value above
each of several sigma thresholds. Extrapolates the count to the full dataset.

Run from repo root with the project venv activated.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataloaders.standard_3dgen_loader import Standard3DGenDataset


CHANNEL_LABELS: dict[int, str] = {
    0: "x", 1: "y", 2: "z", 3: "opacity",
    4: "sh_dc_r", 20: "sh_dc_g", 36: "sh_dc_b",
    52: "scale_x", 53: "scale_y", 54: "scale_z",
    55: "rot_w", 56: "rot_x", 57: "rot_y", 58: "rot_z",
}
DC_INDICES = np.array(list(CHANNEL_LABELS.keys()), dtype=np.int64)
THRESHOLDS = (3.0, 5.0, 8.0, 10.0, 15.0, 20.0, 30.0, 50.0)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--obj_list", required=True, nargs="+")
    p.add_argument("--gs_path", required=True)
    p.add_argument("--mean_file", default=None,
                   help="Optional. If omitted, stats are estimated from a small warm-up subset of the scan.")
    p.add_argument("--std_file", default=None)
    p.add_argument("--sphere2plane_path", required=True)
    p.add_argument("--auto_stats_warmup", type=int, default=200,
                   help="When stats files are omitted, use this many objects to estimate per-channel mean/std.")
    p.add_argument("--num_samples", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--scale_clip_log", type=float, default=None,
                   help="If set, clamp log-scale channels (52,53,54) at this minimum value before stats and outlier counting. "
                        "e.g. -7.6 == real-space scale ~5e-4 (~sub-pixel cutoff at 512² render).")
    args = p.parse_args()

    ds = Standard3DGenDataset(
        obj_list=args.obj_list,
        gs_path=args.gs_path,
        mean_file=args.mean_file,
        std_file=args.std_file,
        sphere2plane_path=args.sphere2plane_path,
    )
    n_total = len(ds)
    print(f"Dataset size: {n_total:,}")

    rng = np.random.default_rng(args.seed)
    n = min(args.num_samples, n_total)
    indices = rng.choice(n_total, size=n, replace=False)
    print(f"Scanning {n:,} objects (seed={args.seed})...")

    use_auto_stats = ds.mean is None or ds.std is None

    # We store per-object min and max in RAW units, plus running sum / sumsq per
    # channel so we can compute global stats once the scan is done. After that
    # we convert min/max to sigma using (mean, std).
    per_obj_min_raw = np.full((n, len(DC_INDICES)), np.inf, dtype=np.float64)
    per_obj_max_raw = np.full((n, len(DC_INDICES)), -np.inf, dtype=np.float64)
    chan_sum = np.zeros(len(DC_INDICES), dtype=np.float64)
    chan_sumsq = np.zeros(len(DC_INDICES), dtype=np.float64)
    chan_count = 0

    log_every = max(1, n // 20)
    for i, idx in enumerate(indices):
        sample = ds[int(idx)]
        pc = sample["point_cloud"]
        if isinstance(pc, torch.Tensor):
            pc = pc.detach().cpu().numpy()
        pc = pc.astype(np.float32)  # (C, H, W) or (N, D)
        if pc.ndim == 3:
            sub = pc[DC_INDICES].reshape(len(DC_INDICES), -1)  # (14, H*W)
        elif pc.ndim == 2:
            sub = pc[:, DC_INDICES].T                          # (14, N)
        else:
            raise RuntimeError(f"unexpected pc shape {pc.shape}")
        # Optional scale clip applied at the gaussian level, before any stats.
        if args.scale_clip_log is not None:
            scale_rows = np.array([7, 8, 9])  # positions of ch 52, 53, 54 in DC_INDICES
            sub = sub.copy()
            sub[scale_rows] = np.maximum(sub[scale_rows], args.scale_clip_log)
        per_obj_min_raw[i] = sub.min(axis=1)
        per_obj_max_raw[i] = sub.max(axis=1)
        chan_sum += sub.sum(axis=1, dtype=np.float64)
        chan_sumsq += (sub.astype(np.float64) ** 2).sum(axis=1)
        chan_count += sub.shape[1]
        if (i + 1) % log_every == 0:
            print(f"  {i + 1:>5d}/{n}  loaded")

    # Decide which (mean, std) to use for sigma conversion.
    if use_auto_stats:
        chan_mean = chan_sum / chan_count
        chan_var = chan_sumsq / chan_count - chan_mean ** 2
        chan_std = np.sqrt(np.maximum(chan_var, 1e-16))
        print("\nComputed per-channel mean/std on the fly (no stats files provided).")
        print("  ch  label        mean         std")
        for j, c in enumerate(DC_INDICES):
            print(f"  {int(c):>2d}  {CHANNEL_LABELS[int(c)]:>9s}   {chan_mean[j]:+9.4f}   {chan_std[j]:8.4f}")
    else:
        chan_mean = ds.mean[DC_INDICES].astype(np.float64)
        chan_std = ds.std[DC_INDICES].astype(np.float64)
        print("\nUsing per-channel mean/std from the provided stats files.")

    # Convert raw per-object min/max to standardized sigma units, then take max-abs.
    per_obj_min_std = (per_obj_min_raw - chan_mean[None, :]) / (chan_std[None, :] + 1e-12)
    per_obj_max_std = (per_obj_max_raw - chan_mean[None, :]) / (chan_std[None, :] + 1e-12)
    per_obj_max = np.maximum(np.abs(per_obj_min_std), np.abs(per_obj_max_std)).astype(np.float32)
    per_obj_any = per_obj_max.max(axis=1)
    print(f"After sigma conversion: median(any)={np.median(per_obj_any):.2f}  "
          f"p99={np.percentile(per_obj_any, 99):.2f}  max={per_obj_any.max():.1f}")

    print("\n=== Distribution of per-object max-abs across the 14 DC channels ===")
    for q in (50, 75, 90, 95, 99, 99.5, 99.9, 100):
        print(f"  p{q:>5.1f} = {np.percentile(per_obj_any, q):8.2f} sigma")

    print("\n=== Objects with ANY DC channel value exceeding threshold ===")
    print(f"  {'thresh':>8s}  {'sample %':>10s}  {'sample N':>10s}  {'extrapolated to full dataset':>32s}")
    for t in THRESHOLDS:
        mask = per_obj_any > t
        frac = mask.mean()
        ci_halfwidth = 1.96 * np.sqrt(frac * (1 - frac) / n)
        est = frac * n_total
        est_lo = max(0, (frac - ci_halfwidth) * n_total)
        est_hi = (frac + ci_halfwidth) * n_total
        print(
            f"  {t:>6.1f}σ  {frac * 100:>9.3f}%  {mask.sum():>10d}  "
            f"~{est:>10,.0f}   (95% CI: {est_lo:,.0f} – {est_hi:,.0f})"
        )

    print("\n=== Per-channel: fraction of objects with that channel exceeding 8σ ===")
    for ch_pos, ch_idx in enumerate(DC_INDICES):
        label = CHANNEL_LABELS[int(ch_idx)]
        for t in (5.0, 8.0, 15.0):
            frac = (per_obj_max[:, ch_pos] > t).mean()
            print(f"  ch{int(ch_idx):>2d} ({label:>9s})  >{t:>4.1f}σ : {frac * 100:>6.3f}% "
                  f"(~{frac * n_total:>8,.0f} objects)")
        print()

    # ----- Residual count after assumed fixes -----
    # Group channels by which "simple fix" would clear their outliers:
    rot_idx = np.array([55, 56, 57, 58])
    scale_idx = np.array([52, 53, 54])
    fixed_idx = np.concatenate([rot_idx, scale_idx])
    residual_idx = np.array([c for c in DC_INDICES if c not in fixed_idx])

    pos_in_dc = {int(c): i for i, c in enumerate(DC_INDICES)}
    rot_pos = np.array([pos_in_dc[int(c)] for c in rot_idx])
    scale_pos = np.array([pos_in_dc[int(c)] for c in scale_idx])
    residual_pos = np.array([pos_in_dc[int(c)] for c in residual_idx])

    rot_max = per_obj_max[:, rot_pos].max(axis=1)
    scale_max = per_obj_max[:, scale_pos].max(axis=1)
    residual_max = per_obj_max[:, residual_pos].max(axis=1)

    print("=== Residual outliers AFTER quaternion unit-norm fix and scale handling ===")
    print(f"    (residual channels: xyz, opacity, sh_dc_r/g/b)")
    print(f"  {'thresh':>8s}  {'sample %':>10s}  {'sample N':>10s}  {'extrapolated':>20s}")
    for t in THRESHOLDS:
        mask = residual_max > t
        frac = mask.mean()
        ci = 1.96 * np.sqrt(frac * (1 - frac) / n)
        est = frac * n_total
        est_lo = max(0, (frac - ci) * n_total)
        est_hi = (frac + ci) * n_total
        print(
            f"  {t:>6.1f}σ  {frac * 100:>9.3f}%  {mask.sum():>10d}  "
            f"~{est:>10,.0f}   (95% CI: {est_lo:,.0f} – {est_hi:,.0f})"
        )

    print("\n=== Outliers AFTER quaternion fix only (scales NOT touched) ===")
    print(f"    (counted channels: xyz, opacity, sh_dc, scale_x/y/z)")
    no_rot_max = np.maximum(scale_max, residual_max)
    for t in THRESHOLDS:
        mask = no_rot_max > t
        frac = mask.mean()
        est = frac * n_total
        print(f"  {t:>6.1f}σ  {frac * 100:>9.3f}%  ~{est:>10,.0f}")

    print("\n=== Decomposition at 8σ: which channel group flags each outlier? ===")
    t = 8.0
    rot_only = (rot_max > t) & (scale_max <= t) & (residual_max <= t)
    scale_only = (scale_max > t) & (rot_max <= t) & (residual_max <= t)
    res_only = (residual_max > t) & (rot_max <= t) & (scale_max <= t)
    rot_and_scale = (rot_max > t) & (scale_max > t) & (residual_max <= t)
    has_residual = (residual_max > t)
    none = (rot_max <= t) & (scale_max <= t) & (residual_max <= t)
    for name, mask in [
        ("clean (no >8σ anywhere)", none),
        ("flagged by rotations ONLY", rot_only),
        ("flagged by scales ONLY", scale_only),
        ("flagged by rotations AND scales (no residual)", rot_and_scale),
        ("flagged by residual channels (xyz/opacity/sh_dc) at all", has_residual),
    ]:
        frac = mask.mean()
        print(f"  {frac * 100:>6.2f}%  (~{frac * n_total:>8,.0f})  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
