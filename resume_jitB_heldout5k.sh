#!/usr/bin/env bash
# Resume the JiT-B held-out overfitting run from its latest checkpoint, into the SAME run dir.
# Identical config to launch_jitB_heldout5k_2gpu.sh; only adds --resume.
# Default run dir: the existing jitB_heldout5k_overfit_*; override: RUN_DIR=output/<dir> bash <this>
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACCEL="$REPO/.3dgen/bin/accelerate"
cd "$REPO"

RUN_DIR="${RUN_DIR:-$(ls -dt "$REPO"/output/jitB_heldout5k_overfit_* | head -1)}"
CKPT="$(ls -t "$RUN_DIR"/0*.pt | head -1)"
RUN="$(basename "$RUN_DIR")"
[ -f "$RUN_DIR/overrides_frozen.yaml" ] || cp jit/configs/overrides.yaml "$RUN_DIR/overrides_frozen.yaml"

export PATH="$REPO/.3dgen/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Resuming $RUN from $CKPT"
nohup "$ACCEL" launch \
    --num_processes 2 --num_machines 1 --multi_gpu --mixed_precision bf16 --dynamo_backend no \
    jit/train_gsplat.py \
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
    --results_dir "output/$RUN" \
    --resume "$CKPT" \
    --exclude_keys_file $REPO/data/exclude_outliers_plus_heldout5k.json \
    --rank_transform_file $REPO/data/stats/rank_quantiles_8ch_clipped.pt \
    --clip_thresholds_file $REPO/data/stats/clip_thresholds_opacity_scales.pt \
    --overrides_yaml "$RUN_DIR/overrides_frozen.yaml" \
    --huber_delta 1.7 --lr 4e-4 --batch_size 256 --gradient_accumulation_steps 1 \
    --heldout_obj_list $REPO/data/heldout5k_obj_list.json \
    --trainprobe_obj_list $REPO/data/trainprobe5k_obj_list.json \
    --heldout_every 2000 --heldout_max_samples 2048 \
    >> "$RUN_DIR/train.log" 2>&1 &

PID=$!
disown "$PID" 2>/dev/null || true
echo "Resumed $RUN (PID $PID) from $(basename "$CKPT")"
echo "  Log:     tail -f $RUN_DIR/train.log"
echo "  Overfit: grep '\[heldout\]' $RUN_DIR/train.log"
