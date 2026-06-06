#!/usr/bin/env python
"""Combined 'lower half' of cost_atlas_summed for two runs, stacked.

The lower half of jit/sinkhorn_perchannel_probe.py::_summed_atlas_grid is the
*per-panel-scale* row of the channel-summed per-GT-cell HARD cost atlas: one panel
per t, each autoscaled with its own colorbar. This script reuses the arrays already
saved in each run's maps.npz (key `summed_hard_atlas`, shape (n_t, H, W)) — NO GPU
recompute — and draws that row for two runs as two stacked rows in one figure.

Edit the RUNS / styling constants below and re-run; it reads only the cached npz.

    .3dgen/bin/python jit/plot_cost_atlas_compare.py
"""
from __future__ import annotations

import os
from pathlib import Path

import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
PROBE_OUT = REPO_ROOT / "jit" / "sinkhorn_perchannel_probe_out"


def _use_times_like_font() -> str:
    """Register and select a Times-New-Roman-compatible serif for CVPR styling.

    Prefers real Times New Roman, else its metric-identical open clones (Tinos /
    Liberation Serif). Returns the resolved family name."""
    search_dirs = [str(Path.home() / ".fonts"), "/usr/share/fonts", "/usr/local/share/fonts"]
    for d in search_dirs:
        for f in glob.glob(os.path.join(d, "**", "*.ttf"), recursive=True):
            base = os.path.basename(f).lower()
            if any(k in base for k in ("times", "tinos", "liberationserif")):
                try:
                    fm.fontManager.addfont(f)
                except Exception:
                    pass
    available = {f.name for f in fm.fontManager.ttflist}
    for fam in ("Times New Roman", "Tinos", "Liberation Serif"):
        if fam in available:
            plt.rcParams["font.family"] = "serif"
            plt.rcParams["font.serif"] = [fam]
            plt.rcParams["mathtext.fontset"] = "stix"
            return fam
    plt.rcParams["font.family"] = "serif"   # fall back to whatever serif exists
    return "serif (no Times clone found)"

# ── What to plot ─────────────────────────────────────────────────────────────
# (output-dir name, row label shown on the left). Top row first.
RUNS = [
    ("huber_d1.7_step5k", "Sinkhorn"),
    ("mse_match5k_b8_5k", "MSE"),
]

# ── Styling knobs (tweak freely; no recompute needed) ────────────────────────
CMAP = "magma"
PANEL_W = 2.4            # inches per t column (tight: panels are square, no bars)
PANEL_H = 3.0            # inches per run row
EXCLUDE_T = {0.9}        # t values to drop (column dropped if |t - x| < 1e-6)
SUPTITLE = "Loss Distribution over Splat Locations"
OUT_PATH = PROBE_OUT / "cost_atlas_summed_lowerhalf_compare.png"
DPI = 150

# Font sizes (pt). Bumped for a single CVPR column; tweak freely.
FS_SUPTITLE = 28.0      # overhead title
FS_COL = 22.5           # noise-% column headers
FS_ROW = 24.5           # Sinkhorn / MSE row labels


def load_run(name: str):
    d = np.load(PROBE_OUT / name / "maps.npz", allow_pickle=True)
    return d["summed_hard_atlas"], d["t_values"]   # (n_t, H, W), (n_t,)


def main() -> None:
    print(f"[font] using {_use_times_like_font()!r}")
    runs = [(label, *load_run(name)) for name, label in RUNS]

    # All runs share the same t grid; pick the columns to keep from the first.
    t_values = runs[0][2]
    keep = [i for i, t in enumerate(t_values)
            if not any(abs(float(t) - x) < 1e-6 for x in EXCLUDE_T)]
    t_keep = [float(t_values[i]) for i in keep]
    n_t = len(keep)
    n_runs = len(runs)

    fig, axes = plt.subplots(
        n_runs, n_t, figsize=(PANEL_W * n_t, PANEL_H * n_runs), squeeze=False,
        layout="constrained",   # spaces big titles/colorbars without overlap
    )

    for r, (label, summed, _t) in enumerate(runs):
        for c, src_i in enumerate(keep):
            ax = axes[r, c]
            ax.imshow(summed[src_i], cmap=CMAP)            # per-panel autoscale
            ax.set_xticks([]); ax.set_yticks([])
            # No colorbar: per-panel autoscaled, so absolute loss values aren't
            # meaningful — dropping the bars gives wider/taller heatmaps.
            if r == 0:                                      # noise % on top row
                ax.set_title(f"{round(100 * (1 - t_keep[c]))}% noise", fontsize=FS_COL)
            if c == 0:                                      # loss name on the left
                ax.set_ylabel(label, fontsize=FS_ROW)

    fig.suptitle(SUPTITLE, fontsize=FS_SUPTITLE)
    os.makedirs(OUT_PATH.parent, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
