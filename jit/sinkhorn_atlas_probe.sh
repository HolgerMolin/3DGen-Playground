#!/bin/bash
# Sinkhorn atlas probe — visualize where the *assignment* loss the model was trained
# on (recon_loss=sinkhorn_patch) stays high, plus how far each GT cell's OT-matched
# prediction sits within its patch. Faithful counterpart to mse_atlas_probe.sh (index
# MSE punishes within-patch permutation, which sinkhorn forgives — see the .py header).
#
# Mirrors train_gsplat.sh path resolution so preprocessing (clip + rank-transform
# pairing, mean/std, sphere2plane) is IDENTICAL to the running training job.
# Runs in the FOREGROUND (single GPU), intentionally light to coexist with training.
#   RESUME=output/<run>/0001000.pt MODEL=JiT-L/8 ./jit/sinkhorn_atlas_probe.sh
#   ./jit/sinkhorn_atlas_probe.sh --num_samples 32 --t_values 0.2,0.5,0.8
# No `set -u`: .env may reference unbound vars (matches train_gsplat.sh).
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
BATCH_SIZE=${BATCH_SIZE:-8}
NUM_WORKERS=${NUM_WORKERS:-4}
MIXED_PRECISION=${MIXED_PRECISION:-bf16}
CKPT_SEARCH_ROOT=${CKPT_SEARCH_ROOT:-output}

# Sinkhorn / patch knobs — defaults match jit_train_gsplat.yaml (sinkhorn_patch run).
PATCH_SIZE=${PATCH_SIZE:-4}
SINKHORN_EPS=${SINKHORN_EPS:-0.05}
SINKHORN_ITERS=${SINKHORN_ITERS:-50}

# Path resolution identical to train_gsplat.sh (.env-driven). GS_DATA_PATH also falls
# back to GS_PATH (the canonical .env var) since DIT_GSPLAT_GS_PATH is not always set.
OBJ_LIST=${OBJ_LIST:-${DIT_GSPLAT_OBJ_LIST:-}}
GS_DATA_PATH=${GS_DATA_PATH:-${DIT_GSPLAT_GS_PATH:-${GS_PATH:-}}}
MEAN_FILE=${MEAN_FILE:-${DIT_GSPLAT_MEAN_FILE:-}}
STD_FILE=${STD_FILE:-${DIT_GSPLAT_STD_FILE:-}}
TEXT_EMBED_PATH=${TEXT_EMBED_PATH:-object_classification/text_tokens}
NULL_TEXT_TOKEN_PATH=${NULL_TEXT_TOKEN_PATH:-object_classification/null_text_token.npz}
SPHERE2PLANE_PATH=${SPHERE2PLANE_PATH:-${DIT_GSPLAT_SPHERE2PLANE_PATH:-}}
RANK_TRANSFORM_FILE=${RANK_TRANSFORM_FILE:-}
CLIP_THRESHOLDS_FILE=${CLIP_THRESHOLDS_FILE:-}
EXCLUDE_KEYS_FILE=${EXCLUDE_KEYS_FILE:-}
RESUME=${RESUME:-}

for path_var in OBJ_LIST GS_DATA_PATH MEAN_FILE STD_FILE TEXT_EMBED_PATH NULL_TEXT_TOKEN_PATH SPHERE2PLANE_PATH; do
    path_value=${!path_var}
    if [ -z "$path_value" ]; then echo "Missing required path variable: $path_var" >&2; exit 1; fi
    if [ ! -e "$path_value" ]; then echo "Path does not exist for $path_var: $path_value" >&2; exit 1; fi
done

PY_ARGS=(
    jit/sinkhorn_atlas_probe.py
    --model "$MODEL"
    --weights "$WEIGHTS"
    --obj_list "$OBJ_LIST"
    --gs_path "$GS_DATA_PATH"
    --mean_file "$MEAN_FILE"
    --std_file "$STD_FILE"
    --text_embed_path "$TEXT_EMBED_PATH"
    --null_text_token_path "$NULL_TEXT_TOKEN_PATH"
    --sphere2plane_path "$SPHERE2PLANE_PATH"
    --sh_degree0_only
    --patch_size "$PATCH_SIZE"
    --sinkhorn_eps "$SINKHORN_EPS"
    --sinkhorn_iters "$SINKHORN_ITERS"
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

echo "Running sinkhorn atlas probe (foreground): model=$MODEL weights=$WEIGHTS t=$T_VALUES N=$NUM_SAMPLES bs=$BATCH_SIZE P=$PATCH_SIZE eps=$SINKHORN_EPS iters=$SINKHORN_ITERS"
exec python "${PY_ARGS[@]}" "$@"
