"""Retroactively render class/cfg-signal trajectories from training logs.

Parses `[cond]` and `Step N` lines out of a JiT train_gsplat.py .err log,
pairs each [cond] record with the immediately preceding Step record (so we
get an opt-step for each conditioning probe), and writes one PNG per metric
plus a combined cfg+class+cos plot.

Usage:
    python jit/plot_class_signal.py \\
        --log output/jit_JiT-B_8_kmeans_k103_v2/train_*.err \\
        [--log output/jit_JiT-B_8_kmeans_k103/train_*.err] \\
        --out-dir output/jit_JiT-B_8_kmeans_k103_v2/loss_plots

Multiple --log args concatenate their records in order (use this to splice the
parent run with the resumed child for a full-lineage view). EMA span is in
records, not steps; tune with --ema-span. NaN/missing values are dropped.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


_STEP_RE = re.compile(
    r"Step\s+(\d+)\s+\|.*?cos\(c,null\):\s*([-+]?\d+\.\d+)"
)
_COND_RE = re.compile(
    r"\[cond\]\s+cfg_signal=(\d+\.\d+)\s+\(min=(\d+\.\d+)\s+max=(\d+\.\d+),\s+n=\d+\)\s+\|"
    r"\s+class_signal=(\d+\.\d+)\s+\(min=(\d+\.\d+)\s+max=(\d+\.\d+),\s+n=\d+\)\s+\|"
    r"\s+pred_rms=(\d+\.\d+)"
)


@dataclass
class Record:
    step: int
    cfg: float
    cfg_min: float
    cfg_max: float
    class_sig: float
    class_min: float
    class_max: float
    pred_rms: float
    cos_c_null: float | None


def _parse_log(path: str) -> list[Record]:
    last_step: int | None = None
    last_cos: float | None = None
    out: list[Record] = []
    with open(path, "r") as f:
        for line in f:
            m = _STEP_RE.search(line)
            if m:
                last_step = int(m.group(1))
                last_cos = float(m.group(2))
                continue
            m = _COND_RE.search(line)
            if m and last_step is not None:
                out.append(Record(
                    step=last_step,
                    cfg=float(m.group(1)),
                    cfg_min=float(m.group(2)),
                    cfg_max=float(m.group(3)),
                    class_sig=float(m.group(4)),
                    class_min=float(m.group(5)),
                    class_max=float(m.group(6)),
                    pred_rms=float(m.group(7)),
                    cos_c_null=last_cos,
                ))
    return out


def _ema(xs: list[float], span: int) -> list[float]:
    if span <= 1 or not xs:
        return list(xs)
    alpha = 2.0 / (span + 1.0)
    out: list[float] = []
    s = xs[0]
    for v in xs:
        s = (1.0 - alpha) * s + alpha * v
        out.append(s)
    return out


def _plot_class_signal(records: list[Record], out_path: str, ema_span: int, splice_step: int | None) -> None:
    steps = [r.step for r in records]
    cs = [r.class_sig for r in records]
    cs_min = [r.class_min for r in records]
    cs_max = [r.class_max for r in records]
    cs_ema = _ema(cs, ema_span)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.fill_between(steps, cs_min, cs_max, alpha=0.15, color="tab:blue", label="min–max over 16 pairs")
    ax.plot(steps, cs, color="tab:blue", linewidth=0.7, alpha=0.45, label="raw mean")
    ax.plot(steps, cs_ema, color="tab:blue", linewidth=2.0, label=f"EMA (span={ema_span})")
    if splice_step is not None:
        ax.axvline(splice_step, color="grey", linestyle="--", linewidth=1, label=f"resume @ {splice_step}")
    ax.set_xlabel("opt step")
    ax.set_ylabel("class_signal  =  ‖pred(y=a) − pred(y=b)‖_RMS / ½(‖a‖+‖b‖)")
    ax.set_title("Class signal — pairwise class discriminability over training")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_combined(records: list[Record], out_path: str, ema_span: int, splice_step: int | None) -> None:
    steps = [r.step for r in records]
    cs = _ema([r.class_sig for r in records], ema_span)
    cf = _ema([r.cfg for r in records], ema_span)
    pr = _ema([r.pred_rms for r in records], ema_span)
    cos = _ema([r.cos_c_null for r in records if r.cos_c_null is not None], ema_span)
    cos_steps = [r.step for r in records if r.cos_c_null is not None]

    fig, ax1 = plt.subplots(figsize=(10, 4.8))
    ax1.plot(steps, cs, label="class_signal (a vs b)", color="tab:blue", linewidth=2.0)
    ax1.plot(steps, cf, label="cfg_signal (c vs null)", color="tab:orange", linewidth=2.0)
    ax1.plot(steps, pr, label="pred_rms ‖pred(y=k)‖", color="tab:gray", linewidth=1.2, linestyle=":")
    ax1.set_xlabel("opt step")
    ax1.set_ylabel("conditioning RMS-fraction")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    ax2 = ax1.twinx()
    ax2.plot(cos_steps, cos, label="mean cos(class, null)", color="tab:red", linewidth=1.6, alpha=0.85)
    ax2.set_ylabel("cos(class_emb, null_emb)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    if splice_step is not None:
        ax1.axvline(splice_step, color="grey", linestyle="--", linewidth=1)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best", fontsize=8)

    ax1.set_title(f"Conditioning telemetry — EMA(span={ema_span})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--log", action="append", required=True,
                   help="path or glob to a train_*.err log; pass multiple times for splicing")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--ema-span", type=int, default=20,
                   help="EMA span in records (one record per [cond] line); default 20")
    p.add_argument("--splice-step", type=int, default=None,
                   help="if set, draws a vertical line at this step (e.g. resume boundary)")
    p.add_argument("--csv", action="store_true",
                   help="also dump the parsed records as class_signal_log.csv")
    args = p.parse_args()

    paths: list[str] = []
    for spec in args.log:
        matches = sorted(glob.glob(spec))
        if not matches:
            raise SystemExit(f"no log file matched: {spec}")
        paths.extend(matches)

    records: list[Record] = []
    for pth in paths:
        recs = _parse_log(pth)
        print(f"[parse] {pth}: {len(recs)} cond records (steps {recs[0].step}..{recs[-1].step})" if recs else f"[parse] {pth}: 0 records")
        records.extend(recs)

    records.sort(key=lambda r: r.step)
    if not records:
        raise SystemExit("no records parsed")

    os.makedirs(args.out_dir, exist_ok=True)
    _plot_class_signal(
        records,
        os.path.join(args.out_dir, "class_signal.png"),
        ema_span=args.ema_span,
        splice_step=args.splice_step,
    )
    _plot_combined(
        records,
        os.path.join(args.out_dir, "conditioning_combined.png"),
        ema_span=args.ema_span,
        splice_step=args.splice_step,
    )

    if args.csv:
        import csv
        csv_path = os.path.join(args.out_dir, "class_signal_log.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "cfg_signal", "cfg_min", "cfg_max",
                        "class_signal", "class_min", "class_max",
                        "pred_rms", "cos_c_null"])
            for r in records:
                w.writerow([r.step, r.cfg, r.cfg_min, r.cfg_max,
                            r.class_sig, r.class_min, r.class_max,
                            r.pred_rms, "" if r.cos_c_null is None else r.cos_c_null])
        print(f"[csv ] wrote {csv_path}  ({len(records)} rows)")

    print(f"[done] wrote class_signal.png and conditioning_combined.png to {args.out_dir}")


if __name__ == "__main__":
    main()
