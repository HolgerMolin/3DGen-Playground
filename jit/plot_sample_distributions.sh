#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .3dgen/bin/activate
set -a; source .env; set +a
python jit/plot_sample_distributions.py \
    --obj_list "$ALL_OBJ_JSON" \
    --gs_path "$GS_PATH" \
    --mean_file "$ALL_MEAN_FILE" \
    --std_file "$ALL_STD_FILE" \
    --sphere2plane_path "$SPHERE2PLANE_PATH" \
    --num_samples "${NUM_SAMPLES:-32}" \
    --bins "${BINS:-128}" \
    --output_dir "${OUTPUT_DIR:-output/distribution_plots}" \
    "$@"
