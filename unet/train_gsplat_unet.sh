#!/bin/bash

source .env

# Usage: ./unet/train_gsplat_unet.sh [UNet-S|UNet-B|UNet-L]
#
# Environment variables are used only for path-like inputs:
#   OBJ_LIST, GS_DATA_PATH, MEAN_FILE, STD_FILE, CLASS_MAP_PATH,
#   SPHERE2PLANE_PATH, REF_CAMERA_TAR, RESULTS_DIR
MODEL=${1:-UNet-B}

NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$NUM_GPUS" -le 0 ]; then
    NUM_GPUS=1
fi

OBJ_LIST=${OBJ_LIST:-${DIT_GSPLAT_OBJ_LIST:-}}
GS_DATA_PATH=${GS_DATA_PATH:-${DIT_GSPLAT_GS_PATH:-}}
MEAN_FILE=${MEAN_FILE:-${DIT_GSPLAT_MEAN_FILE:-}}
STD_FILE=${STD_FILE:-${DIT_GSPLAT_STD_FILE:-}}
CLASS_MAP_PATH=${CLASS_MAP_PATH:-${DIT_GSPLAT_CLASS_MAP:-}}
SPHERE2PLANE_PATH=${SPHERE2PLANE_PATH:-${DIT_GSPLAT_SPHERE2PLANE_PATH:-}}
REF_CAMERA_TAR=${REF_CAMERA_TAR:-${DIT_GSPLAT_REF_CAMERA_TAR:-}}

for path_var in OBJ_LIST GS_DATA_PATH MEAN_FILE STD_FILE CLASS_MAP_PATH SPHERE2PLANE_PATH REF_CAMERA_TAR; do
    path_value=${!path_var}
    if [ -z "$path_value" ]; then
        echo "Missing required path variable: $path_var" >&2
        exit 1
    fi
    if [ ! -e "$path_value" ]; then
        echo "Configured path does not exist for $path_var: $path_value" >&2
        exit 1
    fi
done

RESULTS_DIR=${RESULTS_DIR:-${UNET_GSPLAT_RESULTS_DIR:-output/unet_results_gsplat}}
RUN_TS=$(date +%Y%m%d_%H%M%S)
RUN_STEM="train_${RUN_TS}_$$"

LOG_DIR="${RESULTS_DIR}"
mkdir -p "$LOG_DIR"
STDOUT_FILE="${LOG_DIR}/${RUN_STEM}.out"
STDERR_FILE="${LOG_DIR}/${RUN_STEM}.err"
PID_FILE="${LOG_DIR}/${RUN_STEM}.pid"

if [ "$NUM_GPUS" -le 1 ]; then
    CMD=(python)
else
    CMD=(
        accelerate launch
        --num_processes "$NUM_GPUS"
        --num_machines 1
        --multi_gpu
        --mixed_precision bf16
        --dynamo_backend no
    )
fi

PY_ARGS=(
    unet/train_gsplat_unet.py
    --model "$MODEL"
    --class_embed_dim 768
    --spatial_fold_factor 2
    --obj_list "$OBJ_LIST"
    --gs_path "$GS_DATA_PATH"
    --mean_file "$MEAN_FILE"
    --std_file "$STD_FILE"
    --class_map "$CLASS_MAP_PATH"
    --sphere2plane_path "$SPHERE2PLANE_PATH"
    --sh_degree0_only
    --predict_xstart
    --noise_schedule squaredcos_cap_v2
    --snr_gamma 5.0
    --no-allow_x0_min_snr_weighting
    --enable_render_loss_after 0
    --render_loss_weight 10.0
    --alpha_mask_loss_weight 10.0
    --lpips_loss_weight 10.0
    --lpips_net vgg
    --render_loss_num_cam 2
    --train_render_size 512
    --ref_camera_tar "$REF_CAMERA_TAR"
    --epochs 1000
    --batch_size 8
    --lr 5e-5
    --weight_decay 1e-2
    --lr_warmup_steps 1000
    --lr_total_steps 200000
    --lr_final_ratio 0.01
    --ema_decay 0.9999
    --mixed_precision bf16
    --gradient_accumulation_steps 1
    --num_workers 2
    --seed 0
    --log_every 100
    --ckpt_every 2000
    --fixed_check_every 100
    --fixed_check_timesteps 900,975,999
    --fixed_check_num_samples 2
    --fixed_check_num_cam 1
    --fixed_check_seed 1234
    --val_sampler dpm
    --dpm_solver_order 2
    --dpm_algorithm_type dpmsolver++
    --dpm_solver_type midpoint
    --dpm_timestep_spacing trailing
    --no-dpm_use_karras_sigmas
    --train_render_log_every 100
    --train_render_log_num_cam 1
    --val_every 100
    --results_dir "$RESULTS_DIR"
    # To resume, uncomment and edit the checkpoint path below.
    --resume "output/unet_results_gsplat/0020000.pt"
)

echo "Launching in background"
echo "results: $RESULTS_DIR"
echo "stdout: $STDOUT_FILE"
echo "stderr: $STDERR_FILE"
echo "pid:    $PID_FILE"

nohup "${CMD[@]}" "${PY_ARGS[@]}" >"$STDOUT_FILE" 2>"$STDERR_FILE" < /dev/null &
PID=$!
echo "$PID" > "$PID_FILE"
disown "$PID" 2>/dev/null || true

echo "Started background job with PID $PID"
