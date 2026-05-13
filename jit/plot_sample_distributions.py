"""Per-channel distribution diagnostic for normalized 3DGS samples.

For each non-higher-order-SH channel (the 14 DC indices), draws a 2-panel figure:
left = histogram (>=100 bins) overlaid with the standard normal PDF,
right = QQ-plot against scipy.stats.norm with the y=x diagonal annotated.

Header annotation reports mean, std, skewness, excess kurtosis, min, max, and
the fraction of values within +/- 3 sigma. Useful for eyeballing how Gaussian
the per-channel marginals are post-normalization.

Run from repo root with the project venv activated:
    source .3dgen/bin/activate && python jit/plot_sample_distributions.py ...
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataloaders.standard_3dgen_loader import Standard3DGenDataset


# 59-channel layout: xyz(3) + opacity(1) + SH-R(16) + SH-G(16) + SH-B(16) + scale(3) + rot(4).
# DC-only indices skip the 45 higher-order SH coefficients.
CHANNEL_LABELS: dict[int, str] = {
    0: "x",
    1: "y",
    2: "z",
    3: "opacity",
    4: "sh_dc_r",
    20: "sh_dc_g",
    36: "sh_dc_b",
    52: "scale_x",
    53: "scale_y",
    54: "scale_z",
    55: "rot_w",
    56: "rot_x",
    57: "rot_y",
    58: "rot_z",
}
DC_INDICES: tuple[int, ...] = tuple(CHANNEL_LABELS.keys())


def _load_samples(
    ds: Standard3DGenDataset,
    indices: np.ndarray,
) -> np.ndarray:
    """Pull a batch of (C, H, W) samples and stack into (N, C, H, W) float32."""
    out = []
    for i in indices:
        sample = ds[int(i)]
        pc = sample["point_cloud"]
        if isinstance(pc, torch.Tensor):
            pc = pc.detach().cpu().numpy()
        out.append(pc.astype(np.float32))
    return np.stack(out, axis=0)


def _summary_text(values: np.ndarray) -> str:
    mean = float(values.mean())
    std = float(values.std())
    skew = float(stats.skew(values, bias=False))
    exkurt = float(stats.kurtosis(values, fisher=True, bias=False))
    vmin = float(values.min())
    vmax = float(values.max())
    if std > 0:
        within3 = float(np.mean(np.abs(values - mean) <= 3 * std))
    else:
        within3 = float("nan")
    return (
        f"mean={mean:+.4f}  std={std:.4f}  skew={skew:+.3f}  ex_kurt={exkurt:+.3f}\n"
        f"min={vmin:+.3f}  max={vmax:+.3f}  frac(|x-mean|<=3sigma)={within3:.4f}"
    )


def _plot_channel(
    values: np.ndarray,
    ch_idx: int,
    label: str,
    out_path: Path,
    bins: int,
    qq_subsample: int,
    rng: np.random.Generator,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"channel {ch_idx} ({label})  N={values.size:,}\n{_summary_text(values)}",
        fontsize=10,
        family="monospace",
    )

    # --- Histogram + standard normal PDF ---
    ax_h = axes[0]
    lo, hi = float(np.percentile(values, 0.05)), float(np.percentile(values, 99.95))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(values.min()), float(values.max())
        if hi <= lo:
            hi = lo + 1.0
    ax_h.hist(
        values,
        bins=bins,
        range=(lo, hi),
        density=True,
        color="#3a7bd5",
        alpha=0.75,
        edgecolor="none",
    )
    xs = np.linspace(lo, hi, 512)
    ax_h.plot(xs, stats.norm.pdf(xs), color="#d9534f", lw=1.6, label="N(0, 1)")
    ax_h.axvline(0, color="black", lw=0.5, alpha=0.5)
    ax_h.set_xlabel("value")
    ax_h.set_ylabel("density")
    ax_h.set_title(f"histogram ({bins} bins, clipped to [{lo:.2f}, {hi:.2f}])")
    ax_h.legend(loc="upper right", fontsize=9)

    # --- QQ plot vs N(0, 1) ---
    ax_q = axes[1]
    if values.size > qq_subsample:
        sample = rng.choice(values, size=qq_subsample, replace=False)
    else:
        sample = values
    # With fit=False, probplot returns (theoretical_quantiles, ordered_values).
    theor, ordered = stats.probplot(sample, dist="norm", fit=False, plot=None)
    ax_q.scatter(theor, ordered, s=4, alpha=0.4, color="#3a7bd5")

    qmin = float(min(theor.min(), ordered.min()))
    qmax = float(max(theor.max(), ordered.max()))
    ax_q.plot([qmin, qmax], [qmin, qmax], color="#d9534f", lw=1.4, label="y = x")
    ax_q.set_xlabel("theoretical quantiles (N(0, 1))")
    ax_q.set_ylabel("sample quantiles")
    ax_q.set_title(f"QQ plot vs N(0, 1)  (n={sample.size:,})")
    ax_q.legend(loc="upper left", fontsize=9)
    ax_q.grid(alpha=0.2)

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obj_list", required=True, nargs="+")
    parser.add_argument("--gs_path", required=True)
    parser.add_argument("--mean_file", default=None,
                        help="Optional. If omitted, mean/std are computed from the loaded sample.")
    parser.add_argument("--std_file", default=None)
    parser.add_argument("--rank_transform_file", default=None,
                        help="Optional Gaussian rank-transform tables built by "
                             "data/build_rank_transform.py. Channels listed in the "
                             "tables are mapped to N(0,1) before the standardize step "
                             "(stats for those channels are forced to (0,1) inside the "
                             "dataset, so the standardize round-trip becomes a no-op).")
    parser.add_argument("--sphere2plane_path", required=True)
    parser.add_argument(
        "--output_dir",
        default="output/distribution_plots",
        help="Directory to write per-channel PNGs.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=32,
        help="Number of objects to draw and pool together per channel.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=128,
        help="Histogram bin count (>=100 recommended).",
    )
    parser.add_argument(
        "--qq_subsample",
        type=int,
        default=50000,
        help="Subsample size for the QQ plot (probplot is O(n log n) and busy beyond ~50k points).",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.bins < 100:
        print(f"[warn] --bins {args.bins} is below the requested 100+, continuing anyway.", file=sys.stderr)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = Standard3DGenDataset(
        obj_list=args.obj_list,
        gs_path=args.gs_path,
        mean_file=args.mean_file,
        std_file=args.std_file,
        sphere2plane_path=args.sphere2plane_path,
        rank_transform_file=args.rank_transform_file,
    )
    print(f"Dataset size: {len(ds)}")

    rng = np.random.default_rng(args.seed)
    n = min(args.num_samples, len(ds))
    indices = rng.choice(len(ds), size=n, replace=False)
    print(f"Drawing {n} samples (seed={args.seed})...")

    batch = _load_samples(ds, indices)  # (N, C, H, W)
    if batch.ndim != 4:
        raise RuntimeError(f"Unexpected batch shape {batch.shape}; expected (N, C, H, W)")
    n, c, h, w = batch.shape
    print(f"Loaded batch: shape={batch.shape}  dtype={batch.dtype}")

    flat = batch.transpose(1, 0, 2, 3).reshape(c, -1)  # (C, N*H*W)

    # If the dataset wasn't given mean/std, compute them on the fly from the sample.
    if ds.mean is None or ds.std is None:
        sample_mean = flat.mean(axis=1)
        sample_std = flat.std(axis=1) + 1e-8
        flat = (flat - sample_mean[:, None]) / sample_std[:, None]
        print("Computed per-channel stats from the loaded sample (no mean/std files provided).")

    print(f"Writing per-channel plots to {out_dir.resolve()}")
    for plot_pos, ch_idx in enumerate(DC_INDICES):
        if ch_idx >= c:
            print(f"  [skip] channel {ch_idx} not present (C={c})")
            continue
        label = CHANNEL_LABELS[ch_idx]
        values = flat[ch_idx].astype(np.float64, copy=False)
        out_path = out_dir / f"ch{ch_idx:02d}_{label}.png"
        _plot_channel(
            values=values,
            ch_idx=ch_idx,
            label=label,
            out_path=out_path,
            bins=args.bins,
            qq_subsample=args.qq_subsample,
            rng=rng,
        )
        print(f"  [{plot_pos + 1:>2d}/{len(DC_INDICES)}] ch{ch_idx:02d} ({label:>9s}) -> {out_path.name}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
