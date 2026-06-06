#!/usr/bin/env python3
"""Compare the three render-loss components between two training runs at matched steps.

Parses each run's train.log for the per-step `Render_L1`, `Alpha_L1`, `Render_LPIPS`
values, de-duplicates resume overlaps (keeping the last occurrence of each step),
trims both runs to their common step range, and renders one publication-quality
figure per render-loss component (plus a combined 1x3 panel).

Style: ggplot/seaborn "gray panel + white gridlines" background, Liberation Serif
(metric-compatible with Times New Roman) at 10 pt — CVPR camera-ready friendly.
"""
import argparse
import os
import re
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

REPO = "."
OUTDIR = os.path.join(REPO, "output", "report")

RUNS = [
    dict(
        name="cmp_mse_b8_render5k_20260603_165812",
        label="MSE reconstruction",
        color="#3274A1",      # muted blue
    ),
    dict(
        name="cmp_sinkhuber_b8_render5k_20260603_180222",
        label="Sinkhorn Reconstruction",
        color="#DD3497",      # magenta/pink (echoes the reference plot)
    ),
]

METRICS = [
    dict(key="Render_L1",    ylabel=r"Render $L_1$",         title="Photometric $L_1$"),
    dict(key="Alpha_L1",     ylabel=r"Alpha $L_1$",          title="Alpha (silhouette) $L_1$"),
    dict(key="Render_LPIPS", ylabel="LPIPS",                 title="Render LPIPS"),
]

LINE_RE = re.compile(r"Step\s+(\d+)\b")


def parse_log(path):
    """Return OrderedDict step -> {metric: value}, last occurrence per step wins."""
    pat = {m["key"]: re.compile(rf"{m['key']}:\s*([0-9.eE+-]+)") for m in METRICS}
    rows = OrderedDict()
    with open(path) as f:
        for line in f:
            ms = LINE_RE.search(line)
            if not ms or "Render_L1" not in line:
                continue
            step = int(ms.group(1))
            vals = {}
            ok = True
            for k, p in pat.items():
                mm = p.search(line)
                if not mm:
                    ok = False
                    break
                vals[k] = float(mm.group(1))
            if ok:
                rows[step] = vals  # last occurrence overwrites (handles resumes)
    steps = sorted(rows)
    return steps, rows


def ema(xs, span=7):
    if not xs:
        return xs
    alpha = 2.0 / (span + 1.0)
    out = [xs[0]]
    for v in xs[1:]:
        out.append(alpha * v + (1 - alpha) * out[-1])
    return out


def register_fonts():
    fdir = os.path.expanduser("~/.fonts")
    for fn in ("LiberationSerif-Regular.ttf", "LiberationSerif-Bold.ttf",
               "LiberationSerif-Italic.ttf", "LiberationSerif-BoldItalic.ttf"):
        fp = os.path.join(fdir, fn)
        if os.path.exists(fp):
            font_manager.fontManager.addfont(fp)
    fam = "Liberation Serif"
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": [fam, "Nimbus Roman", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",   # Times-like math glyphs
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "legend.fontsize": 13,
        "xtick.labelsize": 12.5,
        "ytick.labelsize": 12.5,
        "pdf.fonttype": 42,           # embed TrueType (camera-ready safe)
        "ps.fonttype": 42,
    })


# ggplot/seaborn "theme_gray" look: neutral gray panel, white gridlines, no spines.
PANEL = "#EAEAEA"
GRID = "#FFFFFF"


def style_axes(ax):
    ax.set_facecolor(PANEL)
    ax.grid(True, which="major", color=GRID, linewidth=0.9, zorder=0)
    ax.grid(True, which="minor", color=GRID, linewidth=0.5, zorder=0)
    ax.minorticks_on()
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0, colors="#4D4D4D")
    ax.set_axisbelow(True)


def kfmt(x, _):
    if x >= 1000 and x % 1000 == 0:
        return f"{int(x/1000)}k"
    if x >= 1000:
        return f"{x/1000:g}k"
    return f"{int(x)}"


def load_all():
    data = {}
    lo, hi = None, None
    for run in RUNS:
        steps, rows = parse_log(os.path.join(REPO, "output", run["name"], "train.log"))
        data[run["name"]] = (steps, rows)
        if steps:
            lo = steps[0] if lo is None else max(lo, steps[0])
            hi = steps[-1] if hi is None else min(hi, steps[-1])
    return data, lo, hi


def series(run, rows_tuple, key, lo, hi):
    steps, rows = rows_tuple
    xs, ys = [], []
    for s in steps:
        if lo <= s <= hi:
            xs.append(s)
            ys.append(rows[s][key])
    return xs, ys


def draw_metric(ax, data, metric, lo, hi, ymax_from_step=None):
    ymin, ymax = float("inf"), 0.0
    for run in RUNS:
        xs, ys = series(run, data[run["name"]], metric["key"], lo, hi)
        ax.plot(xs, ys, color=run["color"], lw=0.8, alpha=0.20, zorder=2)
        ax.plot(xs, ema(ys), color=run["color"], lw=2.3, alpha=0.95,
                zorder=3, label=run["label"], solid_capstyle="round")
        if ys:
            ymin = min(ymin, min(ys))
            # For the TOP of the axis, optionally ignore the early post-engagement
            # transient so the converged run separation fills the panel; curves above
            # the cap simply run off the top edge.
            top = [y for x, y in zip(xs, ys) if ymax_from_step is None or x >= ymax_from_step]
            if top:
                ymax = max(ymax, max(top))
    # Offset the baseline to the lowest value attained so the run separation fills
    # the panel; small symmetric padding keeps the curves off the frame edges.
    pad = 0.05 * (ymax - ymin)
    ax.set_xlim(lo, hi)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.xaxis.set_major_formatter(FuncFormatter(kfmt))
    ax.set_xlabel("Training step")
    # No y-axis label: the panel title already names the metric, and dropping it
    # frees horizontal width (numeric ticks stay to convey scale).


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", nargs=3, metavar=("NAME", "LABEL", "COLOR"),
                    action="append",
                    help="Run dir name under output/, legend label, hex color. "
                         "Repeat for each run. If omitted, the built-in default pair is used.")
    ap.add_argument("--out_prefix", default="",
                    help="Inserted into output filenames: render_loss_{prefix}{metric}.")
    ap.add_argument("--ymax_from_step", type=int, default=None,
                    help="Set each panel's y-top from values at steps >= this, ignoring "
                         "the early transient so the converged run separation fills the panel.")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.run:
        global RUNS
        RUNS = [dict(name=n, label=l, color=c) for n, l, c in args.run]
    pfx = args.out_prefix
    register_fonts()
    data, lo, hi = load_all()
    print(f"matched step range: {lo} .. {hi}")
    for run in RUNS:
        steps = data[run["name"]][0]
        print(f"  {run['name']}: {len(steps)} pts, steps {steps[0]}..{steps[-1]}")

    # --- three standalone single-column figures ---
    for m in METRICS:
        fig, ax = plt.subplots(figsize=(3.3, 3.3))
        fig.patch.set_facecolor("white")
        style_axes(ax)
        draw_metric(ax, data, m, lo, hi, ymax_from_step=args.ymax_from_step)
        ax.set_title(m["title"], pad=6)
        leg = ax.legend(frameon=True, framealpha=0.92, edgecolor="#CCCCCC",
                        facecolor="white", loc="upper right", handlelength=1.6,
                        borderpad=0.5)
        leg.get_frame().set_linewidth(0.6)
        fig.tight_layout(pad=0.4)
        base = os.path.join(OUTDIR, f"render_loss_{pfx}{m['key'].lower()}")
        fig.savefig(base + ".pdf", bbox_inches="tight")
        fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("wrote", base + ".pdf / .png")

    # --- combined 1x3 panel (bonus, double-column width) ---
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 3.6))
    fig.patch.set_facecolor("white")
    for ax, m in zip(axes, METRICS):
        style_axes(ax)
        draw_metric(ax, data, m, lo, hi, ymax_from_step=args.ymax_from_step)
        ax.set_title(m["title"], pad=6)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center",
               ncol=2, bbox_to_anchor=(0.5, -0.01), handlelength=1.8,
               columnspacing=1.6)
    fig.tight_layout(rect=(0, 0.10, 1, 1), w_pad=1.2)
    base = os.path.join(OUTDIR, f"render_loss_{pfx}compare_panel")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", base + ".pdf / .png")


if __name__ == "__main__":
    main()
