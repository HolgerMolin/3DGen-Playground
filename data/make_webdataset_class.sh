#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

#############################################
# Load environment variables
#############################################
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/../.env}"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  . "$ENV_FILE"
  set +a
fi

GS_DATA_PATH="${GS_DATA_PATH:-${GS_PATH:-${DIT_GS_PATH:-}}}"
MEAN_FILE="${MEAN_FILE:-${GS_MEAN_FILE:-${DIT_GS_MEAN_FILE:-}}}"
STD_FILE="${STD_FILE:-${GS_STD_FILE:-${DIT_GS_STD_FILE:-}}}"
CLASS_MAP_PATH="${CLASS_MAP_PATH:-${DIT_CLASS_MAP:-${DIT_GSPLAT_CLASS_MAP:-}}}"
SPHERE2PLANE_PATH="${SPHERE2PLANE_PATH:-${DIT_SPHERE2PLANE_PATH:-${DIT_GSPLAT_SPHERE2PLANE_PATH:-}}}"
CLASS_WEBDATASET_DIR="${CLASS_WEBDATASET_DIR:-${WEBDATASET_CLASS_DIR:-}}"
OBJ_LIST_PATH="${OBJ_LIST_PATH:-${ALL_OBJ_JSON:-}}"
SHARD_SIZE="${SHARD_SIZE:-1000}"
NUM_WORKERS="${NUM_WORKERS:-0}"
FEATURE_MODE="${FEATURE_MODE:-full}"
STORAGE_DTYPE="${STORAGE_DTYPE:-bfloat16}"
SHARD_PREFIX="${SHARD_PREFIX:-gaussianverse-class}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

if [[ -z "$GS_DATA_PATH" || -z "$CLASS_MAP_PATH" || -z "$SPHERE2PLANE_PATH" || -z "$CLASS_WEBDATASET_DIR" || -z "$OBJ_LIST_PATH" ]]; then
  echo "Missing required configuration. Set GS_DATA_PATH/GS_PATH, CLASS_MAP_PATH, SPHERE2PLANE_PATH, CLASS_WEBDATASET_DIR or WEBDATASET_CLASS_DIR, and OBJ_LIST_PATH or ALL_OBJ_JSON." >&2
  exit 1
fi

read -r -a OBJ_LIST_ARGS <<< "$OBJ_LIST_PATH"

PY_ARGS=(
  --gs_path "$GS_DATA_PATH"
  --class_map "$CLASS_MAP_PATH"
  --sphere2plane_path "$SPHERE2PLANE_PATH"
  --output_dir "$CLASS_WEBDATASET_DIR"
  --obj_list "${OBJ_LIST_ARGS[@]}"
  --shard_size "$SHARD_SIZE"
  --num_workers "$NUM_WORKERS"
  --feature_mode "$FEATURE_MODE"
  --storage_dtype "$STORAGE_DTYPE"
  --shard_prefix "$SHARD_PREFIX"
  --log_level "$LOG_LEVEL"
)

if [[ -n "$MEAN_FILE" ]]; then
  PY_ARGS+=(--mean_file "$MEAN_FILE")
fi

if [[ -n "$STD_FILE" ]]; then
  PY_ARGS+=(--std_file "$STD_FILE")
fi

if [[ -n "${MAX_SHARDS:-}" ]]; then
  PY_ARGS+=(--max_shards "$MAX_SHARDS")
fi

if [[ "${INCLUDE_FULL_PLANE:-0}" == "1" || "${INCLUDE_FULL_PLANE:-false}" == "true" ]]; then
  PY_ARGS+=(--include_full_plane)
fi

cd "$SCRIPT_DIR"
python make_webdataset_class.py "${PY_ARGS[@]}" "$@"

#### DEBUG ONLY ####
# MAX_SHARDS=5 SHARD_SIZE=20 NUM_WORKERS=4 bash make_webdataset_class.sh
