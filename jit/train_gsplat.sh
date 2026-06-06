#!/bin/bash

# Capture caller-provided launch overrides BEFORE sourcing .env, because .env
# unconditionally assigns some of these (e.g. `NUM_GPUS=`) and would otherwise clobber
# an inline `NUM_GPUS=1 ./jit/train_gsplat.sh`. We re-apply them after sourcing so the
# command line always wins over .env.
_REQ_NUM_GPUS="${NUM_GPUS:-}"
_REQ_RESULTS_DIR="${RESULTS_DIR:-}"
_REQ_CLASS_MAP_PATH="${CLASS_MAP_PATH:-}"

source .env

# Re-apply caller overrides that .env may have reset to its own defaults.
[ -n "$_REQ_NUM_GPUS" ] && NUM_GPUS="$_REQ_NUM_GPUS"
[ -n "$_REQ_RESULTS_DIR" ] && RESULTS_DIR="$_REQ_RESULTS_DIR"
[ -n "$_REQ_CLASS_MAP_PATH" ] && CLASS_MAP_PATH="$_REQ_CLASS_MAP_PATH"

# Usage: ./jit/train_gsplat.sh [JiT-S/8|JiT-B/8|JiT-L/8|JiT-XL/8|...]
#
# Environment variables:
#   Path inputs:
#     OBJ_LIST, GS_DATA_PATH, MEAN_FILE, STD_FILE, TEXT_EMBED_PATH,
#     SPHERE2PLANE_PATH, REF_CAMERA_TAR, RESULTS_DIR, RESUME,
#     CLASS_MAP_PATH (optional; enables discrete class conditioning),
#     RANK_TRANSFORM_FILE (optional), VAL_PROMPTS_FILE (optional)
#   Launch overrides:
#     NUM_GPUS, NUM_MACHINES, MIXED_PRECISION, DYNAMO_BACKEND
#   Hyperparameters:
#     HUBER_DELTA — pseudo-Huber radius for the recon ground cost (unset = use config;
#       0 = force off / squared-L2; >0 robust)
#     LOG_RESIDUAL_MAD — truthy (1/true/yes) => log median/MAD/p90/p99 of the matched-residual
#       radius (train/recon_residual_*) to inform HUBER_DELTA. Pair with HUBER_DELTA=0 to measure
#       the natural (un-clipped) distribution. Unset/0 = off (zero overhead).
#     JIT_TRAIN_CONFIG — YAML for train_gsplat.py (default: jit/configs/jit_train_gsplat.yaml)
#     JIT_OVERRIDES_YAML — hot-reload overrides (default: jit/configs/overrides.yaml).
#       Set to empty to disable:  JIT_OVERRIDES_YAML= ./jit/train_gsplat.sh
# Positional $1 is an explicit model override. When unset, the YAML's
# `model:` field is authoritative (via --config). Avoid defaulting MODEL to
# JiT-XL/8 here — passing --model on the CLI would otherwise shadow the YAML.
MODEL_OVERRIDE=${1:-}
JIT_TRAIN_CONFIG=${JIT_TRAIN_CONFIG:-jit/configs/jit_train_gsplat.yaml}
# Unset → default path; explicitly empty → no --overrides_yaml
JIT_OVERRIDES_YAML="${JIT_OVERRIDES_YAML-jit/configs/overrides.yaml}"

# Resolve GPU count. Precedence: explicit NUM_GPUS > #GPUs pinned via CUDA_VISIBLE_DEVICES
# > all physical GPUs. nvidia-smi -L ignores CUDA_VISIBLE_DEVICES (always lists every
# physical GPU), so without this a `CUDA_VISIBLE_DEVICES=1 ./...` would auto-detect 2 and
# launch accelerate --multi_gpu on a device that isn't visible → "invalid device ordinal".
if [ -z "${NUM_GPUS:-}" ]; then
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        NUM_GPUS=$(printf '%s' "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c '[0-9]')
    else
        NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
    fi
fi
if [ "$NUM_GPUS" -le 0 ]; then
    NUM_GPUS=1
fi
NUM_MACHINES=${NUM_MACHINES:-1}
MIXED_PRECISION=${MIXED_PRECISION:-bf16}
DYNAMO_BACKEND=${DYNAMO_BACKEND:-no}

OBJ_LIST=${OBJ_LIST:-${DIT_GSPLAT_OBJ_LIST:-}}
GS_DATA_PATH=${GS_DATA_PATH:-${DIT_GSPLAT_GS_PATH:-}}
MEAN_FILE=${MEAN_FILE:-${DIT_GSPLAT_MEAN_FILE:-}}
STD_FILE=${STD_FILE:-${DIT_GSPLAT_STD_FILE:-}}
TEXT_EMBED_PATH=${TEXT_EMBED_PATH:-object_classification/text_tokens}
NULL_TEXT_TOKEN_PATH=${NULL_TEXT_TOKEN_PATH:-object_classification/null_text_token.npz}
SPHERE2PLANE_PATH=${SPHERE2PLANE_PATH:-${DIT_GSPLAT_SPHERE2PLANE_PATH:-}}
REF_CAMERA_TAR=${REF_CAMERA_TAR:-${DIT_GSPLAT_REF_CAMERA_TAR:-}}
RANK_TRANSFORM_FILE=${RANK_TRANSFORM_FILE:-}
CLIP_THRESHOLDS_FILE=${CLIP_THRESHOLDS_FILE:-}
VAL_PROMPTS_FILE=${VAL_PROMPTS_FILE:-}
RESUME=${RESUME:-}

# If RESUME not set via env, check the YAML config for a resume path
if [ -z "$RESUME" ] && [ -f "$JIT_TRAIN_CONFIG" ]; then
    YAML_RESUME=$(python3 -c "
import sys, yaml
cfg = yaml.safe_load(open('$JIT_TRAIN_CONFIG')) or {}
v = cfg.get('resume')
if v and str(v).lower() not in ('null', 'none', '~', ''):
    print(v)
" 2>/dev/null)
    if [ -n "$YAML_RESUME" ]; then
        RESUME="$YAML_RESUME"
    fi
fi

for path_var in OBJ_LIST GS_DATA_PATH MEAN_FILE STD_FILE TEXT_EMBED_PATH NULL_TEXT_TOKEN_PATH SPHERE2PLANE_PATH REF_CAMERA_TAR; do
    path_value=${!path_var}
    if [ -z "$path_value" ]; then
        echo "Missing required path variable: $path_var" >&2
        exit 1
    fi
    if [ ! -e "$path_value" ]; then
        echo "Configured path does not exist for $path_var: $path_value" >&2
        exit 1
    fi
done

if [ -n "$RESUME" ] && [ ! -e "$RESUME" ]; then
    echo "Configured resume checkpoint does not exist: $RESUME" >&2
    exit 1
fi

if [ -n "$RANK_TRANSFORM_FILE" ] && [ ! -e "$RANK_TRANSFORM_FILE" ]; then
    echo "Configured RANK_TRANSFORM_FILE does not exist: $RANK_TRANSFORM_FILE" >&2
    exit 1
fi

if [ -n "$CLIP_THRESHOLDS_FILE" ] && [ ! -e "$CLIP_THRESHOLDS_FILE" ]; then
    echo "Configured CLIP_THRESHOLDS_FILE does not exist: $CLIP_THRESHOLDS_FILE" >&2
    exit 1
fi

# Resolve the effective model for RESULTS_DIR: CLI positional > YAML > fallback.
# (The actual model argument to Python is handled below — this is display only.)
if [ -n "$MODEL_OVERRIDE" ]; then
    EFFECTIVE_MODEL="$MODEL_OVERRIDE"
else
    YAML_MODEL=$(python3 -c "
import sys, yaml
cfg = yaml.safe_load(open('$JIT_TRAIN_CONFIG')) or {}
v = cfg.get('model')
if v: print(v)
" 2>/dev/null)
    EFFECTIVE_MODEL="${YAML_MODEL:-JiT-XL/8}"
fi

RUN_TS=$(date +%Y%m%d_%H%M%S)
RUN_STEM="train_${RUN_TS}_$$"

# RESULTS_DIR is a BASE; every fresh launch gets a timestamp suffix so it lands in a
# UNIQUE dir and never clobbers a previous run's checkpoints/logs. When resuming, keep
# the dir as-is so we write back into the run being continued.
RESULTS_DIR="${RESULTS_DIR:-output/jit_${EFFECTIVE_MODEL}}"
if [ -z "$RESUME" ]; then
    RESULTS_DIR="${RESULTS_DIR%/}_${RUN_TS}"
fi

LOG_DIR="${RESULTS_DIR}"
mkdir -p "$LOG_DIR"
STDOUT_FILE="${LOG_DIR}/${RUN_STEM}.out"
STDERR_FILE="${LOG_DIR}/${RUN_STEM}.err"
PID_FILE="${LOG_DIR}/${RUN_STEM}.pid"

if [ "$NUM_GPUS" -le 1 ]; then
    CMD=(python)
else
    CMD=(
        accelerate launch
        --num_processes "$NUM_GPUS"
        --num_machines "$NUM_MACHINES"
        --multi_gpu
        --mixed_precision "$MIXED_PRECISION"
        --dynamo_backend "$DYNAMO_BACKEND"
    )
fi

PY_ARGS=(jit/train_gsplat.py)
if [ -f "$JIT_TRAIN_CONFIG" ]; then
    PY_ARGS+=(--config "$JIT_TRAIN_CONFIG")
else
    echo "JIT_TRAIN_CONFIG not found: $JIT_TRAIN_CONFIG (set JIT_TRAIN_CONFIG or add the file)" >&2
    exit 1
fi
# Only pass --model when the user explicitly overrode via positional arg.
# Otherwise the YAML's `model:` takes effect.
if [ -n "$MODEL_OVERRIDE" ]; then
    PY_ARGS+=(--model "$MODEL_OVERRIDE")
fi
PY_ARGS+=(
    --obj_list "$OBJ_LIST"
    --gs_path "$GS_DATA_PATH"
    --mean_file "$MEAN_FILE"
    --std_file "$STD_FILE"
    --text_embed_path "$TEXT_EMBED_PATH"
    --null_text_token_path "$NULL_TEXT_TOKEN_PATH"
    --sphere2plane_path "$SPHERE2PLANE_PATH"
    --ref_camera_tar "$REF_CAMERA_TAR"
    --mixed_precision "$MIXED_PRECISION"
    --results_dir "$RESULTS_DIR"
)
if [ -n "$EXCLUDE_KEYS_FILE" ]; then
    PY_ARGS+=(--exclude_keys_file "$EXCLUDE_KEYS_FILE")
fi
if [ -n "$VAL_PROMPTS_FILE" ]; then
    PY_ARGS+=(--val_prompts_file "$VAL_PROMPTS_FILE")
fi

OVERFIT=${OVERFIT:-0}

if [ -n "$RESUME" ]; then
    PY_ARGS+=(--resume "$RESUME")
fi

if [ -n "$RANK_TRANSFORM_FILE" ]; then
    PY_ARGS+=(--rank_transform_file "$RANK_TRANSFORM_FILE")
fi

if [ -n "$CLIP_THRESHOLDS_FILE" ]; then
    PY_ARGS+=(--clip_thresholds_file "$CLIP_THRESHOLDS_FILE")
fi

if [ "$OVERFIT" -gt 0 ] 2>/dev/null; then
    PY_ARGS+=(--overfit "$OVERFIT")
fi

# Pseudo-Huber radius for the within-patch ground cost. Inline HUBER_DELTA=<x> overrides
# the config's huber_delta (launch-time only; 0 = off / squared-L2). Lets you sweep delta
# per run without editing the YAML.
if [ -n "${HUBER_DELTA:-}" ]; then
    PY_ARGS+=(--huber_delta "$HUBER_DELTA")
fi

# Residual-radius diagnostics: when LOG_RESIDUAL_MAD is truthy, log median/MAD/p90/p99 of the
# matched-residual radius r=||pred-tgt|| (train/recon_residual_*) for the sinkhorn recon modes,
# in the same units as huber_delta. Pair with HUBER_DELTA=0 to read off a principled delta.
case "${LOG_RESIDUAL_MAD:-}" in
    ""|0|false|False|no|No) ;;
    *) PY_ARGS+=(--log_residual_mad) ;;
esac

# Discrete class conditioning. When CLASS_MAP_PATH is set, forward it so the model trains
# with a learnable per-class AdaLN embedding (LabelEmbedder) instead of CLIP text. Without
# it, train_gsplat.py defaults to text conditioning and never sees the classes.
if [ -n "${CLASS_MAP_PATH:-}" ]; then
    if [ ! -e "$CLASS_MAP_PATH" ]; then
        echo "Configured CLASS_MAP_PATH does not exist: $CLASS_MAP_PATH" >&2
        exit 1
    fi
    PY_ARGS+=(--class_map_path "$CLASS_MAP_PATH")
fi

if [ -n "$JIT_OVERRIDES_YAML" ]; then
    PY_ARGS+=(--overrides_yaml "$JIT_OVERRIDES_YAML")
fi

echo "Launching in background"
echo "config:  $JIT_TRAIN_CONFIG"
if [ -n "$JIT_OVERRIDES_YAML" ]; then
    echo "overrides: $JIT_OVERRIDES_YAML"
fi
echo "results: $RESULTS_DIR"
echo "stdout: $STDOUT_FILE"
echo "stderr: $STDERR_FILE"
echo "pid:    $PID_FILE"

nohup env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "${CMD[@]}" "${PY_ARGS[@]}" >"$STDOUT_FILE" 2>"$STDERR_FILE" < /dev/null &
PID=$!
echo "$PID" > "$PID_FILE"
disown "$PID" 2>/dev/null || true

echo "Started background job with PID $PID"
