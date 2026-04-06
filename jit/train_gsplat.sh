#!/bin/bash

source .env

# Usage: ./jit/train_gsplat.sh [JiT-S/8|JiT-B/8|JiT-L/8|JiT-XL/8|...]
#
# Environment variables:
#   Path inputs:
#     OBJ_LIST, GS_DATA_PATH, MEAN_FILE, STD_FILE, CLASS_MAP_PATH,
#     SPHERE2PLANE_PATH, REF_CAMERA_TAR, RESULTS_DIR, RESUME
#   Launch overrides:
#     NUM_GPUS, NUM_MACHINES, MIXED_PRECISION, DYNAMO_BACKEND
MODEL=${1:-JiT-XL/8}

NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}
if [ "$NUM_GPUS" -le 0 ]; then
    NUM_GPUS=1
fi
NUM_MACHINES=${NUM_MACHINES:-1}
MIXED_PRECISION=${MIXED_PRECISION:-bf16}
DYNAMO_BACKEND=${DYNAMO_BACKEND:-no}

OBJ_LIST=${OBJ_LIST:-${DIT_GSPLAT_OBJ_LIST:-}}
GS_DATA_PATH=${GS_DATA_PATH:-${DIT_GSPLAT_GS_PATH:-}}
MEAN_FILE=${MEAN_FILE:-${DIT_GSPLAT_MEAN_FILE:-}}
STD_FILE=${STD_FILE:-${DIT_GSPLAT_STD_FILE:-}}
CLASS_MAP_PATH=${CLASS_MAP_PATH:-${DIT_GSPLAT_CLASS_MAP:-}}
SPHERE2PLANE_PATH=${SPHERE2PLANE_PATH:-${DIT_GSPLAT_SPHERE2PLANE_PATH:-}}
REF_CAMERA_TAR=${REF_CAMERA_TAR:-${DIT_GSPLAT_REF_CAMERA_TAR:-}}
RESUME=${RESUME:-}

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

if [ -n "$RESUME" ] && [ ! -e "$RESUME" ]; then
    echo "Configured resume checkpoint does not exist: $RESUME" >&2
    exit 1
fi

RESULTS_DIR="output/jit_${MODEL}_results_gsplat"
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
        --num_machines "$NUM_MACHINES"
        --multi_gpu
        --mixed_precision "$MIXED_PRECISION"
        --dynamo_backend "$DYNAMO_BACKEND"
    )
fi

PY_ARGS=(
    jit/train_gsplat.py
    --model "$MODEL"
    --obj_list "$OBJ_LIST"
    --gs_path "$GS_DATA_PATH"
    --mean_file "$MEAN_FILE"
    --std_file "$STD_FILE"
    --class_map "$CLASS_MAP_PATH"
    --sphere2plane_path "$SPHERE2PLANE_PATH"
    --sh_degree0_only
    --predict_xstart
    --noise_schedule squaredcos_cap_v2
    --enable_render_loss_after 5000
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
    --ema_decay 0.9999
    --mixed_precision "$MIXED_PRECISION"
    --gradient_accumulation_steps 4
    --num_workers 2
    --seed 0
    --log_every 100
    --ckpt_every 5000
    --train_render_log_every 100
    --train_render_log_num_cam 1
    --val_every 100
    --val_sampler heun
    --val_sampling_steps 50
    --results_dir "$RESULTS_DIR"
)

if [ -n "$RESUME" ]; then
    PY_ARGS+=(--resume "$RESUME")
fi

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
