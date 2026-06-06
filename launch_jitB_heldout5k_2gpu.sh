#!/usr/bin/env bash
# JiT-B/8 overfitting-detection run: BASIC experimental setup (= huber_d1.7) but with a
# 5,000-object HELD-OUT split so train-vs-held-out loss is logged during training.
#
#   model JiT-B/8 | recon=sinkhorn_patch_hard P=8 eps0.05 iters100 cw0.07 | huber_delta=1.7
#   class-cond hier_uniform_k1000 | P_mean=0 P_std=1.5 | render@5k (5k->15k, cutoff t>=0.8)
#   2-GPU DDP, bf16, eff batch = 256 x ga1 x 2 = 512 | lr 4e-4 (JiT-B base; YAML 3e-4 is muP-for-L)
#
# Held-out monitor (NEW, gated flags): every 2000 steps, eval recon + render losses on the 5k
# held-out set AND a disjoint 5k in-training probe -> logs heldout/*, trainprobe/*, overfit/*_gap
# to wandb + train.log. Overfitting = a widening heldout - trainprobe gap.
#
# Split files (built from filtered list, labeled, minus 8sigma outliers):
#   data/exclude_outliers_plus_heldout5k.json  (726 outliers + 5000 held-out -> training never sees them)
#   data/heldout5k_obj_list.json   (5000 held-out)        data/trainprobe5k_obj_list.json (5000 in-train probe)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACCEL="$REPO/.3dgen/bin/accelerate"
cd "$REPO"

TS="$(date +%Y%m%d_%H%M%S)"
RUN="jitB_heldout5k_overfit_${TS}"
RESULTS_DIR="$REPO/output/$RUN"
mkdir -p "$RESULTS_DIR"

# Freeze a private overrides.yaml so the shared file's hot-reload edits can't leak in.
cp jit/configs/overrides.yaml "$RESULTS_DIR/overrides_frozen.yaml"

# venv bin on PATH so gsplat's cached CUDA extension loads when render engages at step 5000.
export PATH="$REPO/.3dgen/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nohup "$ACCEL" launch \
    --num_processes 2 \
    --num_machines 1 \
    --multi_gpu \
    --mixed_precision bf16 \
    --dynamo_backend no \
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
    --exclude_keys_file $REPO/data/exclude_outliers_plus_heldout5k.json \
    --rank_transform_file $REPO/data/stats/rank_quantiles_8ch_clipped.pt \
    --clip_thresholds_file $REPO/data/stats/clip_thresholds_opacity_scales.pt \
    --overrides_yaml "$RESULTS_DIR/overrides_frozen.yaml" \
    --huber_delta 1.7 \
    --lr 4e-4 \
    --batch_size 256 \
    --gradient_accumulation_steps 1 \
    --heldout_obj_list $REPO/data/heldout5k_obj_list.json \
    --trainprobe_obj_list $REPO/data/trainprobe5k_obj_list.json \
    --heldout_every 2000 \
    --heldout_max_samples 2048 \
    >> "$RESULTS_DIR/train.log" 2>&1 &

PID=$!
disown "$PID" 2>/dev/null || true
echo "Launched $RUN  (accelerate PID $PID, 2-GPU DDP, bf16, JiT-B/8, from scratch)"
echo "  eff batch 512 (256 x ga1 x 2, ~57GB/GPU)  lr 4e-4  huber 1.7  class-cond  render@5k"
echo "  Held-out: 5000 excluded + 5000 train-probe; monitor every 2000 steps"
echo "  Log:        tail -f $RESULTS_DIR/train.log"
echo "  Overfit:    grep '\[heldout\]' $RESULTS_DIR/train.log"
echo "  Verify now: grep -E 'num_processes|Dataset size|overfitting monitor ON|render_weight_schedule' $RESULTS_DIR/train.log | head"
