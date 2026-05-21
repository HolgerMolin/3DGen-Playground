#!/bin/bash
# Atlas layout-quality probe — plane-neighbor vs true-3D-neighbor roughness.
# Data-only (no model/checkpoint). Mirrors train_gsplat.sh path resolution so
# preprocessing matches the training data. Foreground, single GPU.
#   ./jit/atlas_layout_quality_probe.sh [--k 8 --num_samples 128 ...]
# No `set -u`: .env may reference unbound vars (matches train_gsplat.sh).
set -eo pipefail

source .env
if [ -f .3dgen/bin/activate ]; then
    # shellcheck disable=SC1091
    source .3dgen/bin/activate
fi

NUM_SAMPLES=${NUM_SAMPLES:-64}
BATCH_SIZE=${BATCH_SIZE:-8}
NUM_WORKERS=${NUM_WORKERS:-4}
K=${K:-4}

OBJ_LIST=${OBJ_LIST:-${DIT_GSPLAT_OBJ_LIST:-}}
GS_DATA_PATH=${GS_DATA_PATH:-${DIT_GSPLAT_GS_PATH:-}}
MEAN_FILE=${MEAN_FILE:-${DIT_GSPLAT_MEAN_FILE:-}}
STD_FILE=${STD_FILE:-${DIT_GSPLAT_STD_FILE:-}}
TEXT_EMBED_PATH=${TEXT_EMBED_PATH:-object_classification/text_tokens}
SPHERE2PLANE_PATH=${SPHERE2PLANE_PATH:-${DIT_GSPLAT_SPHERE2PLANE_PATH:-}}
RANK_TRANSFORM_FILE=${RANK_TRANSFORM_FILE:-}
CLIP_THRESHOLDS_FILE=${CLIP_THRESHOLDS_FILE:-}
EXCLUDE_KEYS_FILE=${EXCLUDE_KEYS_FILE:-}

for path_var in OBJ_LIST GS_DATA_PATH MEAN_FILE STD_FILE TEXT_EMBED_PATH SPHERE2PLANE_PATH; do
    path_value=${!path_var}
    if [ -z "$path_value" ]; then echo "Missing required path variable: $path_var" >&2; exit 1; fi
    if [ ! -e "$path_value" ]; then echo "Path does not exist for $path_var: $path_value" >&2; exit 1; fi
done

PY_ARGS=(
    jit/atlas_layout_quality_probe.py
    --obj_list "$OBJ_LIST"
    --gs_path "$GS_DATA_PATH"
    --mean_file "$MEAN_FILE"
    --std_file "$STD_FILE"
    --text_embed_path "$TEXT_EMBED_PATH"
    --sphere2plane_path "$SPHERE2PLANE_PATH"
    --sh_degree0_only
    --num_samples "$NUM_SAMPLES"
    --batch_size "$BATCH_SIZE"
    --num_workers "$NUM_WORKERS"
    --k "$K"
)
[ -n "$RANK_TRANSFORM_FILE" ]  && PY_ARGS+=(--rank_transform_file "$RANK_TRANSFORM_FILE")
[ -n "$CLIP_THRESHOLDS_FILE" ] && PY_ARGS+=(--clip_thresholds_file "$CLIP_THRESHOLDS_FILE")
[ -n "$EXCLUDE_KEYS_FILE" ]    && PY_ARGS+=(--exclude_keys_file "$EXCLUDE_KEYS_FILE")

echo "Running atlas layout-quality probe (foreground, data-only): N=$NUM_SAMPLES bs=$BATCH_SIZE k=$K"
exec python "${PY_ARGS[@]}" "$@"
