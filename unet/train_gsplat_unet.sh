#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate 3dgen
source .env

# Usage: ./unet/train_gsplat_unet.sh [UNet-S|UNet-B|UNet-L]
MODEL=${1:-${MODEL:-UNet-B}}

NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}
NUM_GPUS=${NUM_GPUS:-1}
NUM_MACHINES=${NUM_MACHINES:-1}
DYNAMO_BACKEND=${DYNAMO_BACKEND:-no}
CLASS_EMBED_DIM=${CLASS_EMBED_DIM:-256}
VAL_SAMPLER=${VAL_SAMPLER:-dpm}
DPM_SOLVER_ORDER=${DPM_SOLVER_ORDER:-2}
DPM_ALGORITHM_TYPE=${DPM_ALGORITHM_TYPE:-dpmsolver++}
DPM_SOLVER_TYPE=${DPM_SOLVER_TYPE:-midpoint}
DPM_TIMESTEP_SPACING=${DPM_TIMESTEP_SPACING:-trailing}
DPM_USE_KARRAS_SIGMAS=${DPM_USE_KARRAS_SIGMAS:-false}
TRAIN_RENDER_LOG_EVERY=${TRAIN_RENDER_LOG_EVERY:-100}
TRAIN_RENDER_LOG_NUM_CAM=${TRAIN_RENDER_LOG_NUM_CAM:-1}
RESULTS_DIR=${RESULTS_DIR:-${UNET_GSPLAT_RESULTS_DIR:-output/unet_results_gsplat}}
CKPT_EVERY=${CKPT_EVERY:-10000}
MIXED_PRECISION=${MIXED_PRECISION:-bf16}
OBJ_LIST=${OBJ_LIST:-$DIT_GSPLAT_OBJ_LIST}
GS_DATA_PATH=${GS_DATA_PATH:-$DIT_GSPLAT_GS_PATH}
MEAN_FILE=${MEAN_FILE:-$DIT_GSPLAT_MEAN_FILE}
STD_FILE=${STD_FILE:-$DIT_GSPLAT_STD_FILE}
CLASS_MAP_PATH=${CLASS_MAP_PATH:-$DIT_GSPLAT_CLASS_MAP}
SPHERE2PLANE_PATH=${SPHERE2PLANE_PATH:-$DIT_GSPLAT_SPHERE2PLANE_PATH}
REF_CAMERA_TAR=${REF_CAMERA_TAR:-$DIT_GSPLAT_REF_CAMERA_TAR}

LOG_DIR="${RESULTS_DIR}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"

if [ "$NUM_GPUS" -le 1 ]; then
    CMD="python"
else
    CMD="accelerate launch --num_processes $NUM_GPUS --num_machines $NUM_MACHINES --multi_gpu --mixed_precision $MIXED_PRECISION --dynamo_backend $DYNAMO_BACKEND"
fi

echo "Logging to $LOG_FILE"
$CMD unet/train_gsplat_unet.py \
    --model $MODEL \
    --class_embed_dim $CLASS_EMBED_DIM \
    --obj_list $OBJ_LIST \
    --gs_path $GS_DATA_PATH \
    --mean_file $MEAN_FILE \
    --std_file $STD_FILE \
    --class_map $CLASS_MAP_PATH \
    --sphere2plane_path $SPHERE2PLANE_PATH \
    --sh_degree0_only \
    --predict_xstart \
    --noise_schedule squaredcos_cap_v2 \
    --snr_gamma 5.0 \
    --enable_render_loss_after -1 \
    --render_loss_weight 1.0 \
    --lpips_loss_weight 1.0 \
    --lpips_net vgg \
    --render_loss_num_cam 1 \
    --train_render_size 256 \
    --ref_camera_tar $REF_CAMERA_TAR \
    --epochs 5 \
    --batch_size 4 \
    --lr 3e-5 \
    --weight_decay 1e-2 \
    --lr_warmup_steps 500 \
    --lr_total_steps 100000 \
    --lr_final_ratio 0.01 \
    --ema_decay 0.9999 \
    --mixed_precision $MIXED_PRECISION \
    --gradient_accumulation_steps 1 \
    --num_workers 4 \
    --seed 0 \
    --log_every 100 \
    --ckpt_every $CKPT_EVERY \
    --val_sampler $VAL_SAMPLER \
    --dpm_solver_order $DPM_SOLVER_ORDER \
    --dpm_algorithm_type $DPM_ALGORITHM_TYPE \
    --dpm_solver_type $DPM_SOLVER_TYPE \
    --dpm_timestep_spacing $DPM_TIMESTEP_SPACING \
    $( [ "$DPM_USE_KARRAS_SIGMAS" = "true" ] && printf '%s' "--dpm_use_karras_sigmas" ) \
    --train_render_log_every $TRAIN_RENDER_LOG_EVERY \
    --train_render_log_num_cam $TRAIN_RENDER_LOG_NUM_CAM \
    --val_every 1000 \
    --results_dir $RESULTS_DIR \
    2>&1 | tee "$LOG_FILE"
