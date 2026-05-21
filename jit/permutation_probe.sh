#!/bin/bash
# Permutation probe — how much does the model rely on the canonical atlas order?
# For each sample it scrambles the 128x128 Gaussian atlas with a fresh random
# permutation, forwards the (same-noise) permuted x_t, and reports three MSEs per
# flow-matching t: canonical (reference), perm-vs-permuted-GT, perm-vs-canonical-GT.
# Two granularities: 'full' (all 16,384 splats) and 'patch' (8x8 patch blocks).
#
# Mirrors train_gsplat.sh / mse_atlas_probe.sh path resolution so preprocessing
# (clip + rank-transform pairing, mean/std, sphere2plane) is IDENTICAL to training.
#
# Runs in the FOREGROUND (single GPU) and is light enough to coexist with a
# training run. Override anything via env or extra flags, e.g.:
#   ./jit/permutation_probe.sh --num_samples 128 --granularities full
#   RESUME=output/<run>/0080000.pt WEIGHTS=model ./jit/permutation_probe.sh
# No `set -u`: .env may reference unbound vars (e.g. LD_LIBRARY_PATH).
set -eo pipefail

source .env

# Use the project venv (see CLAUDE.md). Activate if present.
if [ -f .3dgen/bin/activate ]; then
    # shellcheck disable=SC1091
    source .3dgen/bin/activate
fi

MODEL=${MODEL:-JiT-B/8}
WEIGHTS=${WEIGHTS:-ema}
T_VALUES=${T_VALUES:-0.1,0.3,0.5,0.7,0.9}
GRANULARITIES=${GRANULARITIES:-full,patch}
NUM_SAMPLES=${NUM_SAMPLES:-64}
BATCH_SIZE=${BATCH_SIZE:-16}
NUM_WORKERS=${NUM_WORKERS:-4}
MIXED_PRECISION=${MIXED_PRECISION:-bf16}
MAX_VRAM_GB=${MAX_VRAM_GB:-10}
CKPT_SEARCH_ROOT=${CKPT_SEARCH_ROOT:-output}

# Path resolution identical to train_gsplat.sh (.env-driven).
OBJ_LIST=${OBJ_LIST:-${DIT_GSPLAT_OBJ_LIST:-}}
GS_DATA_PATH=${GS_DATA_PATH:-${DIT_GSPLAT_GS_PATH:-}}
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
    jit/permutation_probe.py
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
    --t_values "$T_VALUES"
    --granularities "$GRANULARITIES"
    --num_samples "$NUM_SAMPLES"
    --batch_size "$BATCH_SIZE"
    --num_workers "$NUM_WORKERS"
    --mixed_precision "$MIXED_PRECISION"
    --max_vram_gb "$MAX_VRAM_GB"
    --ckpt_search_root "$CKPT_SEARCH_ROOT"
)
[ -n "$RANK_TRANSFORM_FILE" ]  && PY_ARGS+=(--rank_transform_file "$RANK_TRANSFORM_FILE")
[ -n "$CLIP_THRESHOLDS_FILE" ] && PY_ARGS+=(--clip_thresholds_file "$CLIP_THRESHOLDS_FILE")
[ -n "$EXCLUDE_KEYS_FILE" ]    && PY_ARGS+=(--exclude_keys_file "$EXCLUDE_KEYS_FILE")
[ -n "$RESUME" ]               && PY_ARGS+=(--resume "$RESUME")

# Single visible GPU keeps the probe out of the trainer's way; override with
# CUDA_VISIBLE_DEVICES=<idx> ./jit/permutation_probe.sh
echo "Running permutation probe (foreground): model=$MODEL weights=$WEIGHTS t=$T_VALUES gran=$GRANULARITIES N=$NUM_SAMPLES bs=$BATCH_SIZE"
exec python "${PY_ARGS[@]}" "$@"
