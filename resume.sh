#!/usr/bin/env bash
# Resume the jit_classcond_hier1000_render5k_eff512 run from its latest checkpoint,
# EXTENDED to a 140,000-step horizon (was ~99,600) and pinned to a single A100.
#
# Reconstructs the original launch command (captured verbatim from the live process)
# and appends --resume plus the horizon/LR overrides below. The original warning still
# holds for the UNTOUCHED args: every absolute-step schedule (render_weight_schedule
# 5k→15k, enable_render_loss_after 5k, overrides.yaml gate) keys off the restored step,
# so don't shift those. They are all long-complete at the 53k checkpoint and unaffected
# by the horizon change.
#
# Horizon/LR overrides (CLI > the shared YAML, which is left untouched):
#   --epochs 300                 loop bound; 200 stopped at step 100k, need ≥280 to reach 140k
#   --max_steps 140000           precise hard stop at step 140k (300 epochs alone overshoot to 150k)
#   --lr_cosine_total_steps 140000   cosine decays to lr_min exactly at step 140k
#   --lr / --text_proj_lr 1.809815e-04   peak re-anchored to the ON-CURVE LR at step 53000 (no jump)
#   --lr_warmup_steps 53000      makes the cosine PEAK land at the resume step, so LR is
#                                value-continuous at restart, then decays to lr_min by 140k.
# These same args stay correct for any future auto-resume (latest ckpt ≥53k stays on the new
# cosine). Do NOT resume a checkpoint EARLIER than 53000 with them — steps <53000 would re-enter
# the linear warmup ramp. The LR re-anchor was computed for the 53000 checkpoint specifically.
#
# Usage:
#   ./resume.sh                # auto-pick latest complete checkpoint (GPU 0)
#   ./resume.sh path/to/N.pt   # resume from a specific checkpoint (must be ≥53000)
#   GPU=1 ./resume.sh          # pin to a different A100
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$REPO/output/jit_classcond_hier1000_render5k_eff512_20260601_040656"
PY="$REPO/.3dgen/bin/python"
cd "$REPO"

# ── 0) Refuse to start a second trainer on this run ──────────────────────────────
# Match on the unique run-dir basename: it appears in --results_dir whether the
# live process was launched with a relative ("output/<run>") or absolute path.
RUN_NAME="$(basename "$RESULTS_DIR")"
if pgrep -fa "train_gsplat.py" | grep -q -- "$RUN_NAME"; then
    echo "ERROR: a trainer is already running on this run. Aborting to avoid two" >&2
    echo "       processes writing the same dir / contending for the GPU:" >&2
    pgrep -fa "train_gsplat.py" | grep -- "$RUN_NAME" >&2
    exit 1
fi

# ── 1) Pick the resume checkpoint ────────────────────────────────────────────────
# Either the explicit arg, or the newest NNNNNNN.pt that passes a zip-integrity
# check (torch.save files are zip archives; a hard-crash mid-save leaves a truncated
# archive — skip those and fall back to the previous complete one).
CKPT="${1:-}"
if [[ -z "$CKPT" ]]; then
    # Zero-padded NNNNNNN.pt names sort by step under a plain reverse lexical sort.
    mapfile -t CANDS < <(ls -1 "$RESULTS_DIR"/[0-9]*.pt 2>/dev/null | sort -r)
    for c in "${CANDS[@]}"; do
        if "$PY" - "$c" <<'PYEOF'
import sys, zipfile
p = sys.argv[1]
try:
    z = zipfile.ZipFile(p)          # torch.save format = zip
    if z.testzip() is not None:     # CRC check every member
        sys.exit(2)
except Exception:
    sys.exit(2)
sys.exit(0)
PYEOF
        then
            CKPT="$c"
            break
        else
            echo "WARN: $c failed integrity check (truncated?), trying older one" >&2
        fi
    done
fi

if [[ -z "$CKPT" || ! -f "$CKPT" ]]; then
    echo "ERROR: no usable checkpoint found in $RESULTS_DIR" >&2
    exit 1
fi
echo "Resuming from: $CKPT"

# ── 2) Re-issue the original command + --resume + 140k-horizon overrides ─────────
# Pin to a single A100 (GPU 0 default; override with GPU=1 ./resume.sh). A plain
# python process only uses one device, but this makes it explicit and frees the other.
GPU="${GPU:-0}"
LOG="$RESULTS_DIR/train.log"
# Put the venv bin on PATH so torch's verify_ninja_availability() finds `ninja` when gsplat
# lazily loads its cached CUDA extension for the render loss. Launching $PY directly (no
# `activate`) otherwise omits .3dgen/bin from PATH → "Ninja is required to load C++ extensions"
# crash on the first render step, even though gsplat_cuda.so is already built+cached
# (~/.cache/torch_extensions/py310_cu128/gsplat_cuda/). No recompile/nvcc needed — ninja just
# no-op-verifies the up-to-date build. (The original 06-01 run worked because it ran activated.)
export PATH="$REPO/.3dgen/bin:$PATH"
CUDA_VISIBLE_DEVICES="$GPU" nohup "$PY" jit/train_gsplat.py \
    --config jit/configs/jit_train_gsplat.yaml \
    --model JiT-B/8 \
    --obj_list ${GS_PATH:-$REPO/data/gaussianverse}/all_obj_list_filtered.json \
    --gs_path ${GS_PATH:-$REPO/data/gaussianverse}/ \
    --mean_file $REPO/data/stats/all_mean_postfix.pt \
    --std_file $REPO/data/stats/all_std_postfix.pt \
    --class_map_path object_labels/hier_uniform_k1000/object_to_class.json \
    --sphere2plane_path ${GS_PATH:-$REPO/data/gaussianverse}/sphere2plane.npy \
    --ref_camera_tar $REPO/artifacts/ref_camera.tar.gz \
    --mixed_precision bf16 \
    --results_dir output/jit_classcond_hier1000_render5k_eff512_20260601_040656 \
    --exclude_keys_file $REPO/data/outlier_keys_8sigma.json \
    --rank_transform_file $REPO/data/stats/rank_quantiles_8ch_clipped.pt \
    --clip_thresholds_file $REPO/data/stats/clip_thresholds_opacity_scales.pt \
    --overrides_yaml jit/configs/overrides.yaml \
    --epochs 300 \
    --max_steps 140000 \
    --lr_cosine_total_steps 140000 \
    --lr 1.809815e-04 \
    --text_proj_lr 1.809815e-04 \
    --lr_warmup_steps 53000 \
    --resume "$CKPT" \
    >> "$LOG" 2>&1 &

PID=$!
disown "$PID" 2>/dev/null || true
echo "Launched trainer PID $PID on GPU $GPU (bf16) — horizon 140k, LR re-anchored at step 53000"
echo "Logs:  tail -f $LOG"
echo "Watch: grep -E 'Resumed at|Step ' $LOG | tail"
