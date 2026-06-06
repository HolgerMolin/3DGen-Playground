#!/usr/bin/env bash
# Controlled single-variable A/B: MSE vs Huber-Sinkhorn recon, EVERYTHING else identical.
# Two fresh-from-scratch JiT-B/8 runs, ONE PER GPU, launched in PARALLEL (background).
#   GPU 0  Run A  MSE            : --recon_loss mse + the 3 neutralizing flags the YAML needs
#                                  (permute_atlas none / chamfer_rev_weight 1.0 / mse_hybrid_weight 0.0)
#   GPU 1  Run B  Huber-Sinkhorn : config as-is (sinkhorn_patch_hard, permute_atlas patch,
#                                  huber_delta 2.0 = "linear for extreme values", cold-start bootstrap)
# Both: eff batch 512 (--batch_size 256 --gradient_accumulation_steps 2; ga4 is tuned for JiT-L),
# --max_steps 40000, seed 0 (from YAML, identical init), bf16. Render engages at step 5000 and
# ramps [5000,0,0,0]->[15000,5.5,2.75,5.5] from the YAML ("current values"). NO --resume (fresh).
# Frozen private overrides per run so the live main run's overrides.yaml edits can't leak in.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; PY="$REPO/.3dgen/bin/python"; cd "$REPO"
export PATH="$REPO/.3dgen/bin:$PATH"          # gsplat cached-ext load at first render (step 5000)
TS="$(date +%Y%m%d_%H%M%S)"
MAX_STEPS="${MAX_STEPS:-40000}"
BATCH="${BATCH:-256}"; GA="${GA:-2}"          # eff = BATCH*GA = 512; fallback BATCH=128 GA=4 if OOM
log(){ echo "[cmp $(date +%H:%M:%S)] $*"; }

launch_one(){
  local name="$1" gpu="$2"; shift 2
  local RD="$REPO/output/$name"; mkdir -p "$RD"
  cp jit/configs/overrides.yaml "$RD/overrides_frozen.yaml"
  log "launching $name on GPU $gpu :: recon-args: ${*:-<config defaults>}"
  nohup env CUDA_VISIBLE_DEVICES="$gpu" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "$PY" jit/train_gsplat.py \
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
      --batch_size "$BATCH" --gradient_accumulation_steps "$GA" \
      --max_steps "$MAX_STEPS" \
      "$@" \
      >> "$RD/train.log" 2>&1 < /dev/null &
  local pid=$!
  echo "$pid" > "$RD/pid"; disown "$pid" 2>/dev/null || true
  log "  $name PID=$pid  log: tail -f $RD/train.log"
}

# Run A: MSE (GPU 0) — neutralize the sinkhorn-only YAML knobs the arg-validator rejects under MSE
launch_one "cmp_mse_b8_render5k_$TS"      0 \
        --recon_loss mse --permute_atlas none --chamfer_rev_weight 1.0 --mse_hybrid_weight 0.0

# Run B: Huber-Sinkhorn (GPU 1) — config defaults (sinkhorn_patch_hard + huber_delta 2.0 + permute patch)
launch_one "cmp_sinkhuber_b8_render5k_$TS" 1

log "BOTH LAUNCHED (eff batch ${BATCH}x${GA}=$((BATCH*GA)), max_steps=$MAX_STEPS)."
log "MSE      : output/cmp_mse_b8_render5k_$TS"
log "SinkHuber: output/cmp_sinkhuber_b8_render5k_$TS"
