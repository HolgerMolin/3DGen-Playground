#!/usr/bin/env bash
# Relaunch an IDENTICAL run to huber_d1.7_20260602_175026 (killed at step 20650),
# FROM SCRATCH, with the ONLY change being the updated MSE-warmup in your config:
#     jit/configs/jit_train_gsplat.yaml
#       mse_hybrid_weight:        1.5 -> 1.0
#       mse_hybrid_warmup_steps: 1000 -> 200
# Those two are intentionally NOT passed on the CLI, so the YAML (your edit) governs
# them. Everything else is copied verbatim from huber_d1.7's checkpoint args:
#   JiT-B/8 | recon_loss=sinkhorn_patch_hard P=8 eps=0.05 iters=100 cw=0.07
#   huber_delta=1.7 (CLI; YAML default is 2.0) | class-cond hier_uniform_k1000
#   P_mean=0 P_std=1.5 | render@5k (schedule 5k->15k, cutoff t>=0.8) | bf16 compile
#   epochs=200, lr=4e-4, cos=99600 (all from the YAML, same as huber_d1.7)
#
# Pinned to GPU 1 (free; the main 103k run is on GPU 0).  Override: GPU=N bash <this>
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$REPO/.3dgen/bin/python"
cd "$REPO"

GPU="${GPU:-1}"
TS="$(date +%Y%m%d_%H%M%S)"
RUN="huber_d1.7_msehybrid_w1.0_wu200_${TS}"
RESULTS_DIR="$REPO/output/$RUN"
mkdir -p "$RESULTS_DIR"

# Freeze a PRIVATE copy of overrides.yaml so the live main run's ongoing hot-reload
# edits to the shared jit/configs/overrides.yaml can't leak into this experiment.
# (Its current contents already match the canonical config, so this stays
# identical-to-huber_d1.7 — it just decouples the two runs.)
cp jit/configs/overrides.yaml "$RESULTS_DIR/overrides_frozen.yaml"

# venv bin on PATH so gsplat's cached CUDA extension loads when render engages at
# step 5000 (launching $PY directly otherwise omits .3dgen/bin -> "Ninja is required").
export PATH="$REPO/.3dgen/bin:$PATH"

CUDA_VISIBLE_DEVICES="$GPU" nohup "$PY" jit/train_gsplat.py \
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
    --exclude_keys_file $REPO/data/outlier_keys_8sigma.json \
    --rank_transform_file $REPO/data/stats/rank_quantiles_8ch_clipped.pt \
    --clip_thresholds_file $REPO/data/stats/clip_thresholds_opacity_scales.pt \
    --overrides_yaml "$RESULTS_DIR/overrides_frozen.yaml" \
    --huber_delta 1.7 \
    >> "$RESULTS_DIR/train.log" 2>&1 &

PID=$!
disown "$PID" 2>/dev/null || true
echo "Launched $RUN  (PID $PID, GPU $GPU, bf16, from scratch, huber_delta=1.7)"
echo "  MSE warmup from YAML: weight=1.0  warmup_steps=200"
echo "  Log:    tail -f $RESULTS_DIR/train.log"
echo "  Verify: grep 'COLD-START' $RESULTS_DIR/train.log   # should read 'λ decays 1.0 -> 0 over 200 steps'"
