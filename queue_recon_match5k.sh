#!/usr/bin/env bash
# Sequential 5k-matched study on GPU 1 (model patch = JiT-B/8 for ALL runs). Steps:
#   1) wait for the running huber relaunch to save 0005000.pt (the sinkhorn-P8 5k reference)
#   2) [STOP_HUBER=1] stop the huber relaunch to free GPU 1
#   3) Run 1 — MSE:            recon_loss=mse, permute_atlas=none (REQUIRED for mse)
#   4) Run 2 — huber-sinkhorn @ recon patch 16: identical to the huber relaunch but
#              --chamfer_patch_size 16 (16x16=256 Gaussians/patch; permute_atlas=patch
#              auto-aligns to 16). recon stays sinkhorn_patch_hard, huber_delta=1.7.
# Both: JiT-B/8 (model patch 8), --max_steps 5000 (saves 0005000.pt then exits), every
# other hyperparameter identical to the huber relaunch (class-cond, render@5k → weight 0
# at step 5000 so the 5k ckpt is pure-recon, P_mean=0, lr 4e-4, bf16, MSE-warmup w1.0/wu200
# from the YAML). Frozen private overrides so the live main run's edits can't leak in.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; PY="$REPO/.3dgen/bin/python"; cd "$REPO"
export PATH="$REPO/.3dgen/bin:$PATH"             # gsplat cached-ext load if render touches step 5000

GPU="${GPU:-1}"
HUBER_DIR="${HUBER_DIR:-output/huber_d1.7_msehybrid_w1.0_wu200_20260603_015346}"
STOP_HUBER="${STOP_HUBER:-1}"
MAX_STEPS="${MAX_STEPS:-5000}"
log(){ echo "[queue $(date +%H:%M:%S)] $*"; }

# 1) wait for the huber 5k checkpoint --------------------------------------------------
log "waiting for $HUBER_DIR/0005000.pt ..."
until [ -f "$HUBER_DIR/0005000.pt" ]; do sleep 30; done
log "huber 0005000.pt present."

# 2) stop the huber relaunch to free GPU 1 ---------------------------------------------
if [ "$STOP_HUBER" = 1 ]; then
  hp="$(cat "$HUBER_DIR"/*.pid 2>/dev/null | head -1 || true)"
  if [ -n "${hp:-}" ] && kill -0 "$hp" 2>/dev/null; then
    pg="$(ps -o pgid= -p "$hp" | tr -d ' ')"
    log "stopping huber pid=$hp pgid=$pg (SIGTERM)"
    kill -TERM -"$pg" 2>/dev/null || true
    for _ in $(seq 1 10); do kill -0 "$hp" 2>/dev/null || break; sleep 3; done
    kill -KILL -"$pg" 2>/dev/null || true
    sleep 5
  fi
  log "huber stopped; GPU $GPU free."
fi

# 3+4) sequential runs (JiT-B/8; extra recon args passed per run) ----------------------
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

# Run 1: MSE (index-aligned). permute MUST be none under mse.
run_one "mse_match5k_b8_$(date +%Y%m%d_%H%M%S)" \
        --recon_loss mse --permute_atlas none

# Run 2: huber-sinkhorn, recon patch 16 (recon_loss=sinkhorn_patch_hard + permute=patch
# come from the YAML, same as the huber relaunch; permute auto-aligns to chamfer_patch_size).
run_one "huber_d1.7_psz16_match5k_$(date +%Y%m%d_%H%M%S)" \
        --huber_delta 1.7 --chamfer_patch_size 16

log "QUEUE COMPLETE."
