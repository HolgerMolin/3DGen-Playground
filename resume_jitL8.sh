#!/usr/bin/env bash
# Resume the JiT-L/8 2-GPU scaling run (jit_L8_2gpu_sink_psz8_eff512_20260603_071312)
# from its latest valid checkpoint, back onto BOTH A100s.
#
# Self-contained + reproducible: points --config / --overrides_yaml at the FROZEN copies
# saved in the run dir, so later edits to the shared jit/configs/*.yaml can't change the
# hyperparameters this run resumes with. No LR re-anchor needed — the config's auto cosine
# horizon (lr_cosine_total_steps=0 -> ~99,200) recomputes identically and the restored step
# keeps the schedule continuous.
#
# Usage:
#   ./resume_jitL8.sh                # auto-pick latest complete checkpoint, both GPUs
#   ./resume_jitL8.sh path/to/N.pt   # resume from a specific checkpoint
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$REPO/output/jit_L8_2gpu_sink_psz8_eff512_20260603_071312"
PY="$REPO/.3dgen/bin/python"
ACCEL="$REPO/.3dgen/bin/accelerate"
cd "$REPO"

# ── 0) Refuse to start a second trainer on this run ──────────────────────────────
RUN_NAME="$(basename "$RESULTS_DIR")"
if pgrep -fa "train_gsplat.py" | grep -q -- "$RUN_NAME"; then
    echo "ERROR: a trainer is already running on this run. Aborting." >&2
    pgrep -fa "train_gsplat.py" | grep -- "$RUN_NAME" >&2
    exit 1
fi

# ── 1) Pick the resume checkpoint (explicit arg, or newest zip-valid NNNNNNN.pt) ──
CKPT="${1:-}"
if [[ -z "$CKPT" ]]; then
    mapfile -t CANDS < <(ls -1 "$RESULTS_DIR"/[0-9]*.pt 2>/dev/null | sort -r)
    for c in "${CANDS[@]}"; do
        if "$PY" - "$c" <<'PYEOF'
import sys, zipfile
p = sys.argv[1]
try:
    z = zipfile.ZipFile(p)          # torch.save format = zip
    if z.testzip() is not None:     # CRC every member
        sys.exit(2)
except Exception:
    sys.exit(2)
sys.exit(0)
PYEOF
        then CKPT="$c"; break
        else echo "WARN: $c failed integrity check (truncated?), trying older" >&2
        fi
    done
fi
if [[ -z "$CKPT" || ! -f "$CKPT" ]]; then
    echo "ERROR: no usable checkpoint found in $RESULTS_DIR" >&2; exit 1
fi
echo "Resuming from: $CKPT"

# venv bin on PATH so gsplat's cached CUDA extension loads (render is already engaged at
# this step) + reduce fragmentation OOM on the larger model.
export PATH="$REPO/.3dgen/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── 2) Re-issue the original 2-GPU launch + --resume, using the FROZEN config/overrides ──
nohup "$ACCEL" launch \
    --num_processes 2 \
    --num_machines 1 \
    --multi_gpu \
    --mixed_precision bf16 \
    --dynamo_backend no \
    jit/train_gsplat.py \
    --config "$RESULTS_DIR/config_frozen.yaml" \
    --model JiT-L/8 \
    --obj_list ${GS_PATH:-$REPO/data/gaussianverse}/all_obj_list_filtered.json \
    --gs_path ${GS_PATH:-$REPO/data/gaussianverse}/ \
    --mean_file $REPO/data/stats/all_mean_postfix.pt \
    --std_file $REPO/data/stats/all_std_postfix.pt \
    --class_map_path object_labels/hier_uniform_k1000/object_to_class.json \
    --sphere2plane_path ${GS_PATH:-$REPO/data/gaussianverse}/sphere2plane.npy \
    --ref_camera_tar $REPO/artifacts/ref_camera.tar.gz \
    --mixed_precision bf16 \
    --results_dir "output/$RUN_NAME" \
    --exclude_keys_file $REPO/data/outlier_keys_8sigma.json \
    --rank_transform_file $REPO/data/stats/rank_quantiles_8ch_clipped.pt \
    --clip_thresholds_file $REPO/data/stats/clip_thresholds_opacity_scales.pt \
    --overrides_yaml "$RESULTS_DIR/overrides_frozen.yaml" \
    --resume "$CKPT" \
    >> "$RESULTS_DIR/train.log" 2>&1 &

PID=$!
disown "$PID" 2>/dev/null || true
echo "Resumed $RUN_NAME  (accelerate PID $PID, 2-GPU DDP) from $(basename "$CKPT")"
echo "  Log:    tail -f $RESULTS_DIR/train.log"
echo "  Watch:  grep -E 'Resumed|Step ' $RESULTS_DIR/train.log | tail"
