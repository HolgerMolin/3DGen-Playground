#!/usr/bin/env bash
# v2 RECOVERY relaunch. Preconditions (already true): huber relaunch's 0005000.pt exists,
# huber stopped, GPU 1 free. Runs sequentially on GPU 1, JiT-B/8, --max_steps 5000:
#   Run 1  MSE: --recon_loss mse + neutralize the 3 chamfer/sinkhorn knobs the YAML sets
#               (else train_gsplat.py rejects them under mse):
#                 --permute_atlas none  --chamfer_rev_weight 1.0  --mse_hybrid_weight 0.0
#   Run 2  huber-sinkhorn @ recon patch 16: --huber_delta 1.7 --chamfer_patch_size 16
#               (recon_loss=sinkhorn_patch_hard + permute=patch from YAML, auto-aligns to 16;
#                YAML chamfer_rev_weight/mse_hybrid stay valid for sinkhorn = identical to ref).
# Everything else identical to the huber reference. Each 5k ckpt is pure-recon (render
# weight 0 at step 5000). Frozen private overrides per run.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; PY="$REPO/.3dgen/bin/python"; cd "$REPO"
export PATH="$REPO/.3dgen/bin:$PATH"
GPU="${GPU:-1}"; MAX_STEPS="${MAX_STEPS:-5000}"
log(){ echo "[queue2 $(date +%H:%M:%S)] $*"; }

# defensive: never co-run with the huber relaunch (would OOM GPU 1)
if ps -eo cmd | grep train_gsplat.py | grep -q "huber_d1.7_msehybrid_w1.0_wu200_20260603_015346"; then
  log "ABORT: huber relaunch still running — free GPU $GPU first"; exit 1
fi

run_one(){
  local name="$1"; shift
  local RD="$REPO/output/$name"; mkdir -p "$RD"; cp jit/configs/overrides.yaml "$RD/overrides_frozen.yaml"
  log "launching $name :: recon-args: $*"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" jit/train_gsplat.py \
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
      --results_dir "output/$name" \
      --exclude_keys_file $REPO/data/outlier_keys_8sigma.json \
      --rank_transform_file $REPO/data/stats/rank_quantiles_8ch_clipped.pt \
      --clip_thresholds_file $REPO/data/stats/clip_thresholds_opacity_scales.pt \
      --overrides_yaml "$RD/overrides_frozen.yaml" \
      --max_steps "$MAX_STEPS" \
      "$@" \
      > "$RD/train.log" 2>&1
  log "$name exited (rc=$?); 5k ckpt: $(ls "$RD"/0005000.pt 2>/dev/null || echo MISSING)"
}

run_one "mse_match5k_b8_$(date +%Y%m%d_%H%M%S)" \
        --recon_loss mse --permute_atlas none --chamfer_rev_weight 1.0 --mse_hybrid_weight 0.0
run_one "huber_d1.7_psz16_match5k_$(date +%Y%m%d_%H%M%S)" \
        --huber_delta 1.7 --chamfer_patch_size 16
log "QUEUE2 COMPLETE."
