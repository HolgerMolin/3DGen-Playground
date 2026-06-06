#!/usr/bin/env bash
# 2026-06-03 OVERNIGHT SCALING RUN: JiT-L/8 across BOTH A100s (DDP), from scratch.
#
# Goal: scale the established sinkhorn_patch_hard + render@5k recipe from JiT-B/8 to
# JiT-L/8 (depth 12->24, hidden 768->1024, heads 12->16; ~130M -> ~460M params),
# holding the EFFECTIVE batch at 512 so every absolute-step schedule stays valid.
#
# Changes vs the JiT-B baseline (all in jit/configs/jit_train_gsplat.yaml):
#   model              JiT-B/8 -> JiT-L/8
#   batch_size         256     -> 64    (per-GPU micro-batch)
#   grad_accum         2       -> 4     (64 x 4 x 2 GPUs = EFFECTIVE 512, unchanged)
#   lr / text_proj_lr  4e-4    -> 3e-4  (muP-ish width scaling: 768/1024)
#   lr_cosine_total    99600   -> 0     (AUTO; resolves to ~99,600 at eff-512)
# UNCHANGED (matches baseline): chamfer_patch_size=8, recon_loss=sinkhorn_patch_hard, eps=0.05, iters=100,
#   cw=0.07, huber_delta=2.0 (config default; NOT 1.7 -> no --huber_delta flag), the
#   mse_hybrid cold-start bootstrap (w=1.0, warmup 200), permute_atlas=patch,
#   chamfer_rev_weight=12, render@5k (schedule 5k->15k, cutoff t>=0.8), class-cond
#   hier_uniform_k1000, bf16, compile, per-sample zoom, clipped rank+clip tables.
#
# Multi-GPU: accelerate launch --multi_gpu --num_processes 2 (NO CUDA_VISIBLE_DEVICES
# pin). compile happens in-script (config compile=true), so accelerate uses
# --dynamo_backend no to avoid double-compiling.
#
# Usage:  ./launch_jitL8_2gpu_sink_psz16.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$REPO/.3dgen/bin/python"
ACCEL="$REPO/.3dgen/bin/accelerate"
cd "$REPO"

TS="$(date +%Y%m%d_%H%M%S)"
RUN="jit_L8_2gpu_sink_psz8_eff512_${TS}"
RESULTS_DIR="$REPO/output/$RUN"
mkdir -p "$RESULTS_DIR"

# Freeze a PRIVATE copy of overrides.yaml so any later live edits to the shared
# jit/configs/overrides.yaml can't leak into this run's hot-reload.
cp jit/configs/overrides.yaml "$RESULTS_DIR/overrides_frozen.yaml"

# venv bin on PATH so gsplat's cached CUDA extension loads when render engages at
# step 5000 (launching directly otherwise omits .3dgen/bin -> "Ninja is required").
export PATH="$REPO/.3dgen/bin:$PATH"
# Reduce fragmentation OOM on the larger model.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nohup "$ACCEL" launch \
    --num_processes 2 \
    --num_machines 1 \
    --multi_gpu \
    --mixed_precision bf16 \
    --dynamo_backend no \
    jit/train_gsplat.py \
    --config jit/configs/jit_train_gsplat.yaml \
    --model JiT-L/8 \
    --obj_list ${GS_PATH:-$REPO/data/gaussianverse}/all_obj_list_filtered.json \
    --gs_path ${GS_PATH:-$REPO/data/gaussianverse}/ \
    --mean_file $REPO/data/stats/all_mean_postfix.pt \
    --std_file $REPO/data/stats/all_std_postfix.pt \
    --class_map_path object_labels/hier_uniform_k1000/object_to_class.json \
    --sphere2plane_path ${GS_PATH:-$REPO/data/gaussianverse}/sphere2plane.npy \
    --ref_camera_tar $REPO/artifacts/ref_camera.tar.gz \
    --mixed_precision bf16 \
    --results_dir "output/$RUN" \
    --exclude_keys_file $REPO/data/outlier_keys_8sigma.json \
    --rank_transform_file $REPO/data/stats/rank_quantiles_8ch_clipped.pt \
    --clip_thresholds_file $REPO/data/stats/clip_thresholds_opacity_scales.pt \
    --overrides_yaml "$RESULTS_DIR/overrides_frozen.yaml" \
    >> "$RESULTS_DIR/train.log" 2>&1 &

PID=$!
disown "$PID" 2>/dev/null || true
echo "Launched $RUN  (accelerate PID $PID, 2-GPU DDP, bf16, from scratch)"
echo "  Model:  JiT-L/8   eff batch 512 (64 x ga4 x 2 GPUs)   recon=sinkhorn_patch_hard psz8   lr 3e-4"
echo "  Log:    tail -f $RESULTS_DIR/train.log"
echo "  Verify: grep -E 'num_processes|EFFECTIVE|COLD-START|patch' $RESULTS_DIR/train.log | head"
