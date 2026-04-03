#!/bin/bash
source .env

cd "$REPO_ROOT"

CHECKPOINT_PATH="${UNET_INF_CHECKPOINT:-output/unet_results_gsplat}"

python unet/inf.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --results_dir output/unet_inference_gsplat \
    --state_key auto \
    --sampler ddpm \
    --class_map "$DIT_GSPLAT_CLASS_MAP" \
    --mean_file "$DIT_GSPLAT_MEAN_FILE" \
    --std_file "$DIT_GSPLAT_STD_FILE" \
    --sphere2plane_path "$DIT_GSPLAT_SPHERE2PLANE_PATH" \
    --ref_camera_tar "$DIT_GSPLAT_REF_CAMERA_TAR" \
    --render_size 512 \
    --num_inference_steps 250 \
    --num_render_views 4 \
    --seed 0 \
    --mixed_precision none \
    --class_label 0
