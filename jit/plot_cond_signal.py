"""Parse [cond] lines from a train_gsplat .err and plot cfg_signal / cond_signal.

Usage: python jit/plot_cond_signal.py <err_path> [<out_png>]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt

STEP_RE = re.compile(r"Step\s+(\d+)\s*\|")
COND_RE = re.compile(
    r"\[cond\]\s+cfg_signal=([0-9.]+)\s+\(min=([0-9.]+)\s+max=([0-9.]+),\s+n=\d+\)\s+\|\s+"
    r"cond_signal=([0-9.]+)\s+\(min=([0-9.]+)\s+max=([0-9.]+),\s+n=\d+\)\s+\|\s+"
    r"pred_rms=([0-9.]+)"
)


def parse(err_path: Path):
    rows = []
    last_step = None
    for line in err_path.read_text().splitlines():
        m_step = STEP_RE.search(line)
        if m_step:
            last_step = int(m_step.group(1))
            continue
        m_cond = COND_RE.search(line)
        if m_cond and last_step is not None:
            cfg_m, cfg_lo, cfg_hi, c_m, c_lo, c_hi, pr = (float(x) for x in m_cond.groups())
            rows.append((last_step, cfg_m, cfg_lo, cfg_hi, c_m, c_lo, c_hi, pr))
    return rows


def main():
    err_path = Path(sys.argv[1])
    out_png = Path(sys.argv[2]) if len(sys.argv) > 2 else err_path.parent / "loss_plots" / "cond_signal.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)

    rows = parse(err_path)
    if not rows:
        print(f"No [cond] entries paired with Step lines in {err_path}", file=sys.stderr)
        sys.exit(1)

    steps = [r[0] for r in rows]
    cfg_m = [r[1] for r in rows]
    cfg_lo = [r[2] for r in rows]
    cfg_hi = [r[3] for r in rows]
    c_m = [r[4] for r in rows]
    c_lo = [r[5] for r in rows]
    c_hi = [r[6] for r in rows]
    pr = [r[7] for r in rows]

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    ax = axes[0]
    ax.fill_between(steps, cfg_lo, cfg_hi, alpha=0.2, color="C0", label="min/max (n=16 prompts)")
    ax.plot(steps, cfg_m, color="C0", linewidth=1.5, label="mean")
    ax.set_ylabel("cfg_signal\n(pred(y) vs pred(null), RMS-normalized)")
    ax.set_title(f"Conditioning probe trajectory — {err_path.parent.name}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    ax = axes[1]
    ax.fill_between(steps, c_lo, c_hi, alpha=0.2, color="C1", label="min/max (n=16 pairs)")
    ax.plot(steps, c_m, color="C1", linewidth=1.5, label="mean")
    ax.set_ylabel("cond_signal\n(pred(a) vs pred(b), RMS-normalized)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    ax = axes[2]
    ax.plot(steps, pr, color="C2", linewidth=1.5)
    ax.set_ylabel("pred_rms")
    ax.set_xlabel("training step")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    print(f"Wrote {out_png}  ({len(rows)} probe points, step {steps[0]}-{steps[-1]})")


if __name__ == "__main__":
    main()
