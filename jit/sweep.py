"""
jit/sweep.py — single-parameter hyperparameter sweep over the JiT 3DGS trainer.

For each value in --values, copies the base YAML, overrides one key, launches
jit/train_gsplat.py capped at --max_steps, then aggregates loss curves and
summary metrics across trials.

Trials run sequentially, each using all visible GPUs (same throughput as a
normal training run). Each trial starts from fresh init.

Example:
    source .3dgen/bin/activate
    python jit/sweep.py --param lr --values "1e-4,2e-4,5e-4" --max_steps 2000
"""
import argparse
import copy
import csv
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE_CONFIG = REPO_ROOT / "jit/configs/jit_train_gsplat.yaml"

# Columns recorded in loss_plots/loss_log.csv that are worth overlaying.
# (mse_by_t_bucket columns and grad_norm_<loss> are handled separately below.)
LOSS_COLUMNS = [
    "mse",
    "render_l1",
    "alpha_l1",
    "lpips",
    "aux",
    "repel",
    "class_null_cos",
    "grad_norm",
]


def _value_to_slug(value) -> str:
    s = str(value)
    return "".join(c if (c.isalnum() or c in "-._") else "_" for c in s)


def _parse_value(raw: str):
    """Parse a CLI value with YAML semantics so ints/floats/bools/lists work."""
    return yaml.safe_load(raw)


def _detect_num_gpus() -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "-L"], stderr=subprocess.DEVNULL
        ).decode()
        return max(1, len([l for l in out.splitlines() if l.strip()]))
    except Exception:
        return 1


def _build_trainer_cmd(
    trial_cfg_path: Path,
    results_dir: Path,
    num_gpus: int,
    mixed_precision: str,
) -> list[str]:
    """argv for jit/train_gsplat.py — required paths come from the sourced .env."""
    py_args = [
        "jit/train_gsplat.py",
        "--config", str(trial_cfg_path),
        "--obj_list", "$OBJ_LIST",
        "--gs_path", "$GS_DATA_PATH",
        "--mean_file", "$MEAN_FILE",
        "--std_file", "$STD_FILE",
        "--text_embed_path", "$TEXT_EMBED_PATH",
        "--sphere2plane_path", "$SPHERE2PLANE_PATH",
        "--ref_camera_tar", "$REF_CAMERA_TAR",
        "--mixed_precision", mixed_precision,
        "--results_dir", str(results_dir),
    ]
    if num_gpus <= 1:
        return ["python"] + py_args
    return [
        "accelerate", "launch",
        "--num_processes", str(num_gpus),
        "--num_machines", "1",
        "--multi_gpu",
        "--mixed_precision", mixed_precision,
        "--dynamo_backend", "no",
    ] + py_args


def _run_trial(
    cmd_argv: list[str],
    trial_dir: Path,
) -> tuple[int, float]:
    """Run cmd inside a bash subshell that sources .env; stdio → trial_dir."""
    # Use literal "$VAR" tokens in cmd_argv so the subshell expands them
    # after .env is sourced. This matches the behavior of train_gsplat.sh.
    quoted = " ".join(c if c.startswith("$") else shlex.quote(c) for c in cmd_argv)
    # Mirror train_gsplat.sh's fallback chain: some .env files only define
    # the DIT_GSPLAT_* variants. Validate each required var is set after
    # the chain so we fail fast with a clear message instead of letting
    # argparse complain about an empty value.
    fallback_chain = (
        "OBJ_LIST=${OBJ_LIST:-${DIT_GSPLAT_OBJ_LIST:-}}\n"
        "GS_DATA_PATH=${GS_DATA_PATH:-${DIT_GSPLAT_GS_PATH:-}}\n"
        "MEAN_FILE=${MEAN_FILE:-${DIT_GSPLAT_MEAN_FILE:-}}\n"
        "STD_FILE=${STD_FILE:-${DIT_GSPLAT_STD_FILE:-}}\n"
        "CLASS_MAP_PATH=${CLASS_MAP_PATH:-${DIT_GSPLAT_CLASS_MAP:-}}\n"
        "SPHERE2PLANE_PATH=${SPHERE2PLANE_PATH:-${DIT_GSPLAT_SPHERE2PLANE_PATH:-}}\n"
        "REF_CAMERA_TAR=${REF_CAMERA_TAR:-${DIT_GSPLAT_REF_CAMERA_TAR:-}}\n"
        "for v in OBJ_LIST GS_DATA_PATH MEAN_FILE STD_FILE CLASS_MAP_PATH "
        "SPHERE2PLANE_PATH REF_CAMERA_TAR; do\n"
        "  if [ -z \"${!v}\" ]; then\n"
        "    echo \"sweep.py: required path variable '$v' is empty after sourcing .env\" >&2\n"
        "    exit 2\n"
        "  fi\n"
        "done\n"
        "export OBJ_LIST GS_DATA_PATH MEAN_FILE STD_FILE CLASS_MAP_PATH "
        "SPHERE2PLANE_PATH REF_CAMERA_TAR\n"
    )
    stdout_path = trial_dir / "stdout.log"
    stderr_path = trial_dir / "stderr.log"
    # Mirror train_gsplat.sh: only pass --exclude_keys_file / --rank_transform_file
    # when the env var is set. Cache hash includes both (via excluded-keys
    # filter and rank-quantile bytes), so omitting them produces a fresh cache
    # and trains on a dataset that differs from your normal runs.
    extra_args = (
        "EXTRA_ARGS=()\n"
        "if [ -n \"$EXCLUDE_KEYS_FILE\" ]; then\n"
        "  EXTRA_ARGS+=(--exclude_keys_file \"$EXCLUDE_KEYS_FILE\")\n"
        "fi\n"
        "if [ -n \"$RANK_TRANSFORM_FILE\" ]; then\n"
        "  EXTRA_ARGS+=(--rank_transform_file \"$RANK_TRANSFORM_FILE\")\n"
        "fi\n"
    )
    # stdout → file only (mostly empty for this trainer).
    # stderr → tee'd so the trainer's Python logging is captured to file
    # AND forwarded to the parent terminal in real time. Without this the
    # user has no signal during a multi-hour trial.
    bash_script = (
        "set -a\n"
        f". {shlex.quote(str(REPO_ROOT / '.env'))}\n"
        "set +a\n"
        + fallback_chain
        + extra_args
        + "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True\n"
        + f"exec {quoted} \"${{EXTRA_ARGS[@]}}\" "
        + f">{shlex.quote(str(stdout_path))} "
        + f"2> >(tee {shlex.quote(str(stderr_path))} >&2)\n"
    )
    t0 = time.time()
    proc = subprocess.Popen(["bash", "-c", bash_script], cwd=REPO_ROOT)
    rc = proc.wait()
    return rc, time.time() - t0


def _read_loss_log(trial_dir: Path) -> dict[str, np.ndarray]:
    """Return {column: 1-D float array} from loss_plots/loss_log.csv.

    Missing values (empty / nan) are kept as np.nan so per-column dropna
    in plotting still works."""
    csv_path = trial_dir / "loss_plots" / "loss_log.csv"
    if not csv_path.exists():
        return {}
    try:
        with csv_path.open() as f:
            reader = csv.reader(f)
            header = next(reader)
            cols: dict[str, list[float]] = {h: [] for h in header}
            for row in reader:
                if not row:
                    continue
                # Pad short rows with empty cells
                if len(row) < len(header):
                    row = row + [""] * (len(header) - len(row))
                for h, v in zip(header, row):
                    try:
                        cols[h].append(float(v) if v not in ("", "nan", "NaN") else float("nan"))
                    except ValueError:
                        cols[h].append(float("nan"))
        return {h: np.asarray(v, dtype=float) for h, v in cols.items()}
    except Exception as e:
        print(f"  warning: failed to parse {csv_path}: {e}", file=sys.stderr)
        return {}


def _summary_for(log: dict[str, np.ndarray], metric: str, window: int) -> dict:
    out = {
        "n_logs": 0,
        "max_step": float("nan"),
        "mean": float("nan"),
        "final": float("nan"),
        "min": float("nan"),
        "min_at_step": float("nan"),
    }
    if not log or "step" not in log:
        return out
    steps = log["step"]
    out["n_logs"] = int(len(steps))
    if len(steps):
        out["max_step"] = float(steps[-1])
    if metric not in log:
        return out
    series = log[metric]
    mask = np.isfinite(series)
    if not mask.any():
        return out
    valid_series = series[mask]
    valid_steps = steps[mask]
    tail = valid_series[-window:] if window > 0 else valid_series
    out["mean"] = float(np.mean(tail))
    out["final"] = float(valid_series[-1])
    j = int(np.argmin(valid_series))
    out["min"] = float(valid_series[j])
    out["min_at_step"] = float(valid_steps[j])
    return out


def _plot_overlay(trials, column: str, out_path: Path) -> bool:
    fig, ax = plt.subplots(figsize=(9, 5))
    any_plotted = False
    for trial in trials:
        log = trial["log"]
        if column not in log or "step" not in log:
            continue
        steps = log["step"]
        values = log[column]
        mask = np.isfinite(steps) & np.isfinite(values)
        if not mask.any():
            continue
        ax.plot(steps[mask], values[mask], label=trial["label"], linewidth=1.6, alpha=0.9)
        any_plotted = True
    if not any_plotted:
        plt.close(fig)
        return False
    ax.set_xlabel("optimizer step")
    ax.set_ylabel(column)
    ax.set_title(f"{column} (sweep overlay)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    if column in ("mse", "render_l1", "alpha_l1", "lpips", "grad_norm") and any_plotted:
        # Loss curves usually benefit from log-y for early-step comparison —
        # but only if every plotted point is strictly positive.
        ymin, ymax = ax.get_ylim()
        if ymin > 0 and ymax > 0:
            try:
                ax.set_yscale("log")
            except ValueError:
                pass
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return True


def _plot_summary_bar(trials, metric: str, window: int, out_path: Path) -> None:
    labels = [t["label"] for t in trials]
    means = [t["summary"]["mean"] for t in trials]
    mins = [t["summary"]["min"] for t in trials]
    x = np.arange(len(labels))
    width = 0.4
    fig, ax = plt.subplots(figsize=(max(7, 1.5 * len(labels)), 5))
    bars_mean = ax.bar(x - width / 2, means, width, label=f"mean (last {window} logs)")
    bars_min = ax.bar(x + width / 2, mins, width, label="min")
    for bar, value in list(zip(bars_mean, means)) + list(zip(bars_min, mins)):
        if np.isfinite(value):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.4g}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} — summary across trials")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--param", required=True,
                        help="YAML key in the base config to sweep (e.g. lr, P_mean, batch_size).")
    parser.add_argument("--values", required=True,
                        help="Comma-separated list of values, parsed as YAML literals "
                             "(e.g. '1e-4,2e-4,5e-4' or 'true,false').")
    parser.add_argument("--max_steps", type=int, required=True,
                        help="Cap each trial at this many optimizer steps.")
    parser.add_argument("--base_config", default=str(DEFAULT_BASE_CONFIG),
                        help="Base YAML config (default: jit/configs/jit_train_gsplat.yaml).")
    parser.add_argument("--sweep_dir", default=None,
                        help="Output dir for the sweep. "
                             "Default: output/sweeps/sweep_<param>_<ts>/")
    parser.add_argument("--metric", default="mse",
                        help="Column in loss_log.csv used for the summary bar chart "
                             "(default: mse). All known columns get an overlay plot regardless.")
    parser.add_argument("--summary_window", type=int, default=200,
                        help="Number of trailing log rows averaged for the 'mean' summary stat.")
    parser.add_argument("--mixed_precision", default="bf16",
                        choices=["bf16", "fp16", "none"])
    parser.add_argument("--log_every", type=int, default=None,
                        help="Override log_every in the trial configs. Useful for short trials "
                             "where the default (200) yields too few log rows.")
    parser.add_argument("--val_every", type=int, default=0,
                        help="Override val_every in the trial configs (default: 0 = disable "
                             "validation for sweeps, since validation is expensive and not "
                             "the sweep signal).")
    parser.add_argument("--ckpt_every", type=int, default=None,
                        help="Override ckpt_every in the trial configs. Default leaves the base "
                             "config's value (which usually means no intermediate ckpts within "
                             "a short trial — only a final ckpt is written).")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print the planned per-trial commands without launching them.")
    parser.add_argument("--skip_existing", action="store_true",
                        help="If a trial dir already has a loss_log.csv with rows, skip launching "
                             "it (still aggregated into plots).")
    args = parser.parse_args()

    base_cfg_path = Path(args.base_config).resolve()
    base_cfg = yaml.safe_load(base_cfg_path.read_text()) or {}

    if args.param not in base_cfg:
        print(
            f"WARNING: '{args.param}' is not a key in {base_cfg_path.name}. "
            "Continuing — the trainer's argparse will reject unknown keys.",
            file=sys.stderr,
        )

    values = [_parse_value(v.strip()) for v in args.values.split(",") if v.strip()]
    if not values:
        print("ERROR: --values must contain at least one entry.", file=sys.stderr)
        sys.exit(2)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = (
        Path(args.sweep_dir)
        if args.sweep_dir
        else REPO_ROOT / "output" / "sweeps" / f"sweep_{_value_to_slug(args.param)}_{ts}"
    )
    sweep_dir = sweep_dir.resolve()
    sweep_dir.mkdir(parents=True, exist_ok=True)
    print(f"sweep dir: {sweep_dir}")

    num_gpus = _detect_num_gpus()
    print(f"detected GPUs: {num_gpus}")

    meta = {
        "param": args.param,
        "values": values,
        "max_steps": args.max_steps,
        "base_config": str(base_cfg_path),
        "metric": args.metric,
        "summary_window": args.summary_window,
        "mixed_precision": args.mixed_precision,
        "num_gpus": num_gpus,
        "timestamp": ts,
    }
    (sweep_dir / "sweep_meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))

    trials = []
    for idx, value in enumerate(values):
        slug = _value_to_slug(value)
        trial_dir = sweep_dir / f"trial_{idx:02d}_{slug}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        trial_cfg = copy.deepcopy(base_cfg)
        trial_cfg[args.param] = value
        trial_cfg["max_steps"] = args.max_steps
        # Each trial starts fresh
        trial_cfg.pop("resume", None)
        # Hot-reload overrides are noise for short trials
        trial_cfg.pop("overrides_yaml", None)
        # Apply per-sweep overrides
        if args.log_every is not None:
            trial_cfg["log_every"] = args.log_every
        if args.val_every is not None:
            trial_cfg["val_every"] = args.val_every
        if args.ckpt_every is not None:
            trial_cfg["ckpt_every"] = args.ckpt_every

        trial_cfg_path = trial_dir / "config.yaml"
        trial_cfg_path.write_text(yaml.safe_dump(trial_cfg, sort_keys=False))

        cmd = _build_trainer_cmd(
            trial_cfg_path, trial_dir, num_gpus, args.mixed_precision
        )

        label = f"{args.param}={value}"
        trial = {
            "idx": idx,
            "value": value,
            "label": label,
            "dir": trial_dir,
            "cfg": trial_cfg_path,
            "cmd": cmd,
        }

        existing_log = trial_dir / "loss_plots" / "loss_log.csv"
        if args.skip_existing and existing_log.exists():
            rows = max(0, sum(1 for _ in existing_log.open()) - 1)
            if rows > 0:
                print(f"[trial {idx}] {label} — skipping (loss_log.csv has {rows} rows)")
                trial["status"] = "skipped"
                trials.append(trial)
                continue

        print(f"\n[trial {idx}] {label} → {trial_dir}")
        print(f"  log:  {trial_dir / 'stderr.log'}")
        print(f"  tail: tail -f {trial_dir / 'stderr.log'}")
        print(f"  cmd:  {' '.join(cmd)}")
        if args.dry_run:
            trial["status"] = "dry-run"
            trials.append(trial)
            continue

        rc, elapsed = _run_trial(cmd, trial_dir)
        trial["status"] = "ok" if rc == 0 else f"failed(rc={rc})"
        trial["elapsed_s"] = elapsed
        print(f"  status: {trial['status']}  ({elapsed / 60:.1f} min)")
        if rc != 0:
            print(f"  see {trial_dir / 'stderr.log'}")
        trials.append(trial)

    if args.dry_run:
        print("\nDry run — no trials launched.")
        return

    # ---- Aggregate + visualize ----
    for t in trials:
        t["log"] = _read_loss_log(t["dir"])
        t["summary"] = _summary_for(t["log"], args.metric, args.summary_window)

    plot_dir = sweep_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for column in LOSS_COLUMNS:
        out_path = plot_dir / f"{column}.png"
        if _plot_overlay(trials, column, out_path):
            written.append(out_path)
    summary_path = plot_dir / f"summary_{args.metric}.png"
    _plot_summary_bar(trials, args.metric, args.summary_window, summary_path)
    written.append(summary_path)

    summary_csv = sweep_dir / "summary.csv"
    with summary_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "trial_idx", "value", "status", "elapsed_s", "max_step",
            f"mean_{args.metric}_last{args.summary_window}",
            f"final_{args.metric}",
            f"min_{args.metric}",
            "min_at_step",
            "trial_dir",
        ])
        for t in trials:
            s = t["summary"]
            w.writerow([
                t["idx"], t["value"], t.get("status", "?"),
                f"{t.get('elapsed_s', float('nan')):.1f}",
                s["max_step"], s["mean"], s["final"], s["min"], s["min_at_step"],
                str(t["dir"]),
            ])

    print("\n=== Sweep complete ===")
    print(f"sweep dir:   {sweep_dir}")
    print(f"summary csv: {summary_csv}")
    print(f"plots:       {plot_dir}/")
    print()
    header = (
        f"{'idx':>3}  {'value':<18}  {'status':<14}  "
        f"{'mean_'+args.metric:<16}  {'final_'+args.metric:<16}  "
        f"{'min_'+args.metric:<16}  {'max_step':>8}"
    )
    print(header)
    print("-" * len(header))
    for t in trials:
        s = t["summary"]
        max_step = int(s["max_step"]) if np.isfinite(s["max_step"]) else "-"
        print(
            f"{t['idx']:>3}  {str(t['value']):<18}  {str(t.get('status','?')):<14}  "
            f"{s['mean']:<16.6g}  {s['final']:<16.6g}  {s['min']:<16.6g}  {max_step:>8}"
        )


if __name__ == "__main__":
    main()
