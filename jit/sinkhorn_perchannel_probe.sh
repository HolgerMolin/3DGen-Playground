#!/bin/bash
# Per-CHANNEL Sinkhorn difficulty probe — which GT *channels* (and atlas *pixels*) does the
# trained sinkhorn_patch_hard recon loss stay worst on? Per-channel decomposition of
# sinkhorn_atlas_probe.sh, faithful to recon_loss=sinkhorn_patch_hard (HARD argmax assignment
# of the huber'd Sinkhorn plan), class-conditioned (LabelEmbedder) by default.
#
# Defaults match the classcond_hier1000 run's knobs: P=8, eps=0.05, iters=100, huber=2.0.
# Mirrors train_gsplat.sh path resolution so preprocessing is IDENTICAL to the training job.
# Runs in the FOREGROUND (single GPU), light enough to coexist with training.
#   RESUME=output/<run>/0103000.pt CUDA_VISIBLE_DEVICES=1 ./jit/sinkhorn_perchannel_probe.sh
#   ./jit/sinkhorn_perchannel_probe.sh --num_samples 128 --t_values 0.1,0.3,0.5,0.7,0.9
set -eo pipefail

source .env

if [ -f .3dgen/bin/activate ]; then
    # shellcheck disable=SC1091
    source .3dgen/bin/activate
fi

MODEL=${MODEL:-JiT-B/8}
WEIGHTS=${WEIGHTS:-ema}
T_VALUES=${T_VALUES:-0.1,0.3,0.5,0.7,0.9}
NUM_SAMPLES=${NUM_SAMPLES:-64}
BATCH_SIZE=${BATCH_SIZE:-4}
NUM_WORKERS=${NUM_WORKERS:-4}
MIXED_PRECISION=${MIXED_PRECISION:-bf16}
CKPT_SEARCH_ROOT=${CKPT_SEARCH_ROOT:-output}

# Sinkhorn / patch knobs — defaults match jit_train_gsplat.yaml (sinkhorn_patch_hard run).
PATCH_SIZE=${PATCH_SIZE:-8}
SINKHORN_EPS=${SINKHORN_EPS:-0.05}
SINKHORN_ITERS=${SINKHORN_ITERS:-100}
HUBER_DELTA=${HUBER_DELTA:-2.0}

# Path resolution identical to train_gsplat.sh (.env-driven).
OBJ_LIST=${OBJ_LIST:-${DIT_GSPLAT_OBJ_LIST:-}}
GS_DATA_PATH=${GS_DATA_PATH:-${DIT_GSPLAT_GS_PATH:-${GS_PATH:-}}}
MEAN_FILE=${MEAN_FILE:-${DIT_GSPLAT_MEAN_FILE:-}}
STD_FILE=${STD_FILE:-${DIT_GSPLAT_STD_FILE:-}}
SPHERE2PLANE_PATH=${SPHERE2PLANE_PATH:-${DIT_GSPLAT_SPHERE2PLANE_PATH:-}}
RANK_TRANSFORM_FILE=${RANK_TRANSFORM_FILE:-}
CLIP_THRESHOLDS_FILE=${CLIP_THRESHOLDS_FILE:-}
EXCLUDE_KEYS_FILE=${EXCLUDE_KEYS_FILE:-}
CLASS_MAP_PATH=${CLASS_MAP_PATH:-}
RESUME=${RESUME:-}

for path_var in OBJ_LIST GS_DATA_PATH MEAN_FILE STD_FILE SPHERE2PLANE_PATH CLASS_MAP_PATH; do
    path_value=${!path_var}
    if [ -z "$path_value" ]; then echo "Missing required path variable: $path_var" >&2; exit 1; fi
    if [ ! -e "$path_value" ]; then echo "Path does not exist for $path_var: $path_value" >&2; exit 1; fi
done

PY_ARGS=(
    jit/sinkhorn_perchannel_probe.py
    --model "$MODEL"
    --weights "$WEIGHTS"
    --obj_list "$OBJ_LIST"
    --gs_path "$GS_DATA_PATH"
    --mean_file "$MEAN_FILE"
    --std_file "$STD_FILE"
    --class_map_path "$CLASS_MAP_PATH"
    --sphere2plane_path "$SPHERE2PLANE_PATH"
    --sh_degree0_only
    --patch_size "$PATCH_SIZE"
    --sinkhorn_eps "$SINKHORN_EPS"
    --sinkhorn_iters "$SINKHORN_ITERS"
    --huber_delta "$HUBER_DELTA"
    --t_values "$T_VALUES"
    --num_samples "$NUM_SAMPLES"
    --batch_size "$BATCH_SIZE"
    --num_workers "$NUM_WORKERS"
    --mixed_precision "$MIXED_PRECISION"
    --ckpt_search_root "$CKPT_SEARCH_ROOT"
)
[ -n "$RANK_TRANSFORM_FILE" ]  && PY_ARGS+=(--rank_transform_file "$RANK_TRANSFORM_FILE")
[ -n "$CLIP_THRESHOLDS_FILE" ] && PY_ARGS+=(--clip_thresholds_file "$CLIP_THRESHOLDS_FILE")
[ -n "$EXCLUDE_KEYS_FILE" ]    && PY_ARGS+=(--exclude_keys_file "$EXCLUDE_KEYS_FILE")
[ -n "$RESUME" ]               && PY_ARGS+=(--resume "$RESUME")

echo "Running per-channel sinkhorn probe (foreground): model=$MODEL weights=$WEIGHTS t=$T_VALUES N=$NUM_SAMPLES bs=$BATCH_SIZE P=$PATCH_SIZE eps=$SINKHORN_EPS iters=$SINKHORN_ITERS huber=$HUBER_DELTA"
exec python "${PY_ARGS[@]}" "$@"
