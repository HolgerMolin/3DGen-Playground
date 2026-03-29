#!/bin/bash
source .env

# Number of GPUs (default: all visible)
NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}
NUM_GPUS=${NUM_GPUS:-1}
NUM_MACHINES=${NUM_MACHINES:-1}
DYNAMO_BACKEND=${DYNAMO_BACKEND:-no}
TRAIN_RENDER_LOG_EVERY=${TRAIN_RENDER_LOG_EVERY:-1000}
TRAIN_RENDER_LOG_NUM_CAM=${TRAIN_RENDER_LOG_NUM_CAM:-1}
RESULTS_DIR=${RESULTS_DIR:-$DIT_RESULTS_DIR}
CKPT_EVERY=${CKPT_EVERY:-10000}
MIXED_PRECISION=${MIXED_PRECISION:-bf16}
OBJ_LIST=${OBJ_LIST:-$DIT_OBJ_LIST}
GS_DATA_PATH=${GS_DATA_PATH:-$DIT_GS_PATH}
MEAN_FILE=${MEAN_FILE:-$DIT_GS_MEAN_FILE}
STD_FILE=${STD_FILE:-$DIT_GS_STD_FILE}
CLASS_MAP_PATH=${CLASS_MAP_PATH:-$DIT_CLASS_MAP}
SPHERE2PLANE_PATH=${SPHERE2PLANE_PATH:-$DIT_SPHERE2PLANE_PATH}
REF_CAMERA_TAR=${REF_CAMERA_TAR:-$DIT_REF_CAMERA_TAR}

# Single-GPU: runs with plain python
# Multi-GPU:  runs with accelerate launch
if [ "$NUM_GPUS" -le 1 ]; then
    CMD="python"
else
    CMD="accelerate launch --num_processes $NUM_GPUS --num_machines $NUM_MACHINES --multi_gpu --mixed_precision $MIXED_PRECISION --dynamo_backend $DYNAMO_BACKEND"
fi

$CMD jit/train.py \
    --model JiT-B/16 \
    --obj_list $OBJ_LIST \
    --gs_path $GS_DATA_PATH \
    --mean_file $MEAN_FILE \
    --std_file $STD_FILE \
    --class_map $CLASS_MAP_PATH \
    --sphere2plane_path $SPHERE2PLANE_PATH \
    --sh_degree0_only \
    --predict_xstart \
    --enable_render_loss_after 2000 \
    --render_loss_weight 0.1 \
    --lpips_loss_weight 0.1 \
    --lpips_net vgg \
    --render_loss_num_cam 1 \
    --train_render_size 256 \
    --ref_camera_tar $REF_CAMERA_TAR \
    --epochs 100 \
    --batch_size 32 \
    --lr 1e-4 \
    --ema_decay 0.9999 \
    --mixed_precision $MIXED_PRECISION \
    --gradient_accumulation_steps 1 \
    --no-gradient_checkpointing \
    --num_workers 4 \
    --seed 0 \
    --log_every 100 \
    --ckpt_every $CKPT_EVERY \
    --train_render_log_every $TRAIN_RENDER_LOG_EVERY \
    --train_render_log_num_cam $TRAIN_RENDER_LOG_NUM_CAM \
    --val_every 1000 \
    --results_dir $RESULTS_DIR
