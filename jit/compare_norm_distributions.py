"""Compare post-normalization x/y distributions under two stats files.

Pulls N samples through the dataloader, applies each (mean, std) pair, and
reports first/second/third/fourth moments plus tail percentiles for the
position channels. Saves overlay histograms next to N(0,1) for x and y.
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


def _load(p: str) -> np.ndarray:
    t = torch.load(p, map_location="cpu", weights_only=True).cpu().numpy().reshape(-1)
    return t.astype(np.float64)


def _moments(x: np.ndarray) -> dict:
    m = x.mean()
    s = x.std()
    z = (x - m) / (s + 1e-12)
    skew = (z ** 3).mean()
    kurt_excess = (z ** 4).mean() - 3.0
    return {
        "mean": float(m),
        "std": float(s),
        "skew": float(skew),
        "kurt_excess": float(kurt_excess),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def _summarize(label: str, x: np.ndarray) -> None:
    M = _moments(x)
    p = np.percentile(x, [0.1, 1, 5, 50, 95, 99, 99.9])
    print(
        f"  {label:14s}  mean={M['mean']:+.4f}  std={M['std']:.4f}  "
        f"skew={M['skew']:+.3f}  excess_kurt={M['kurt_excess']:+.3f}  "
        f"min={M['min']:+.2f}  max={M['max']:+.2f}"
    )
    print(
        f"                  pctiles  0.1%={p[0]:+.3f}  1%={p[1]:+.3f}  5%={p[2]:+.3f}  "
        f"50%={p[3]:+.3f}  95%={p[4]:+.3f}  99%={p[5]:+.3f}  99.9%={p[6]:+.3f}"
    )


def _ks_to_normal(x: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic against N(0,1) — proper, no scipy needed."""
    from math import erf, sqrt
    xs = np.sort(x)
    n = xs.shape[0]
    cdf_emp = np.arange(1, n + 1) / n
    cdf_th = 0.5 * (1.0 + np.vectorize(lambda u: erf(u / sqrt(2.0)))(xs))
    return float(np.max(np.abs(cdf_emp - cdf_th)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj_list", required=True, nargs="+")
    ap.add_argument("--gs_path", required=True)
    ap.add_argument("--sphere2plane_path", required=True)
    ap.add_argument("--orig_mean", required=True)
    ap.add_argument("--orig_std", required=True)
    ap.add_argument("--new_mean", required=True)
    ap.add_argument("--new_std", required=True)
    ap.add_argument("--num_samples", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="runs/norm_distribution_check")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    om = _load(args.orig_mean); os_ = _load(args.orig_std)
    nm = _load(args.new_mean);  ns_ = _load(args.new_std)

    ds = Standard3DGenDataset(
        obj_list=args.obj_list, gs_path=args.gs_path,
        mean_file=None, std_file=None,
        sphere2plane_path=args.sphere2plane_path,
    )
    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(ds), size=min(args.num_samples, len(ds)), replace=False)
    print(f"Sampling {len(indices)} objects (seed {args.seed}) from {len(ds)} total.")

    raws = []
    for i in indices:
        s = ds[int(i)]
        pc = s["point_cloud"]
        if isinstance(pc, torch.Tensor):
            pc = pc.detach().cpu().numpy()
        raws.append(pc.astype(np.float64))
    batch = np.stack(raws)  # (N, 59, H, W) plane-ordered

    # Flatten everything except channel
    flat = batch.transpose(1, 0, 2, 3).reshape(batch.shape[1], -1)  # (59, N*H*W)
    print(f"Per-channel sample size: {flat.shape[1]:,} points")

    for ch_name, ch in [("x (ch 0)", 0), ("y (ch 1)", 1), ("z (ch 2)", 2)]:
        raw = flat[ch]
        n_orig = (raw - om[ch]) / (os_[ch] + 1e-8)
        n_new  = (raw - nm[ch]) / (ns_[ch] + 1e-8)

        print(f"\n=== {ch_name} ===")
        _summarize("RAW",         raw)
        _summarize("current_run", n_orig)
        _summarize("last_run",    n_new)
        ks_orig = _ks_to_normal(n_orig)
        ks_new  = _ks_to_normal(n_new)
        print(f"  KS distance to N(0,1):  current_run={ks_orig:.4f}   last_run={ks_new:.4f}   "
              f"(smaller = closer to standard normal)")

        # Histograms: shared bins so they're directly comparable
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("  matplotlib not available, skipping plot")
            continue

        # Plot all three side by side
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
        bins_norm = np.linspace(-5, 5, 121)
        # Pick a wider range for orig if needed (post-norm std ~0.2)
        bins_raw = np.linspace(np.percentile(raw, 0.1), np.percentile(raw, 99.9), 121)
        x_axis = np.linspace(-5, 5, 400)
        ref_pdf = (1.0 / np.sqrt(2*np.pi)) * np.exp(-x_axis**2 / 2)

        axes[0].hist(raw, bins=bins_raw, density=True, color="gray", alpha=0.7)
        axes[0].set_title(f"RAW {ch_name}\nstd={raw.std():.3f}")
        axes[0].set_xlabel("value"); axes[0].set_ylabel("density")

        axes[1].hist(n_orig, bins=bins_norm, density=True, color="C0", alpha=0.7,
                     label=f"std={n_orig.std():.3f}\nkurt+={_moments(n_orig)['kurt_excess']:+.2f}")
        axes[1].plot(x_axis, ref_pdf, "k--", linewidth=1.5, label="N(0,1)")
        axes[1].set_title(
            "current_run\n"
            "playground/gaussianverse_plane_stats_{mean,std}_c11.pt\n"
            f"KS={ks_orig:.3f}"
        )
        axes[1].set_xlabel("z = (x - mean) / std"); axes[1].legend(loc="upper right")
        axes[1].set_xlim(-5, 5)

        axes[2].hist(n_new, bins=bins_norm, density=True, color="C1", alpha=0.7,
                     label=f"std={n_new.std():.3f}\nkurt+={_moments(n_new)['kurt_excess']:+.2f}")
        axes[2].plot(x_axis, ref_pdf, "k--", linewidth=1.5, label="N(0,1)")
        axes[2].set_title(
            "last_run\n"
            "data/stats/all_{mean,std}.pt\n"
            f"KS={ks_new:.3f}"
        )
        axes[2].set_xlabel("z = (x - mean) / std"); axes[2].legend(loc="upper right")
        axes[2].set_xlim(-5, 5)

        fig.suptitle(f"Channel {ch} — {ch_name}", fontsize=13)
        fig.tight_layout()
        out_path = out_dir / f"ch{ch}_distribution.png"
        fig.savefig(out_path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
