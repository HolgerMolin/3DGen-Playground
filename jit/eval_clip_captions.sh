#!/usr/bin/env bash
# CLIP text-image alignment of the BEST text-conditioned checkpoint on a baseline
# caption set (default: data/baseline_captions_500.json, 500 single-object captions).
#
# For each caption it samples N_SEEDS objects (deterministic per-(prompt,seed) noise),
# renders each from N_VIEWS cameras, and scores
#     mean_v cos( CLIP_img(view_v), CLIP_txt(caption) )   [ViT-L/14 joint space]
# averaged over seeds, per CFG scale. Wraps jit/eval_clip_alignment.py (same sample ->
# render -> CLIP pipeline as the training-time validation grids).
#
# Usage:
#   bash jit/eval_clip_captions.sh
#   GPU=0 N_SEEDS=4 CFG_SCALES=1,2,3,4,6 MAX_PROMPTS=100 SHUFFLE_SEED=0 \
#       CAPTIONS_FILE=data/baseline_captions_500.json bash jit/eval_clip_captions.sh
#
# Regenerate / resize the baseline caption set with:
#   .3dgen/bin/python data/gen_baseline_captions.py --n 500 --seed 0 \
#       --out data/baseline_captions_500.json
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$REPO/.3dgen/bin/python"
cd "$REPO"
export PATH="$REPO/.3dgen/bin:$PATH"          # cached gsplat CUDA ext needs .3dgen/bin on PATH
# Source .env with nounset OFF: .env references vars like $LD_LIBRARY_PATH that may be
# unset, which would abort under `set -u`. allexport (set -a) exports what it defines.
if [ -f "$REPO/.env" ]; then set -a; set +u; . "$REPO/.env"; set +a; set -u; fi

# ---- knobs (env-overridable) --------------------------------------------------------
GPU="${GPU:-0}"                               # GPU 0 has the most free memory right now
# Best text-conditioned checkpoint = flagship sinkhorn+render run @ 99k (EMA weights used).
CKPT="${CKPT:-$REPO/output/jit_final_sinkhorn_render_bs512_69k_20260529_071914/0099195.pt}"
CAPTIONS_FILE="${CAPTIONS_FILE:-$REPO/data/baseline_captions_500.json}"
N_SEEDS="${N_SEEDS:-2}"                        # samples per caption (avg'd); raise for tighter SEM
N_VIEWS="${N_VIEWS:-4}"
CFG_SCALES="${CFG_SCALES:-1,2,3,4,6}"
MAX_PROMPTS="${MAX_PROMPTS:-}"                 # cap caption count (subsample); empty = all 500
SHUFFLE_SEED="${SHUFFLE_SEED:-}"              # deterministic shuffle before MAX_PROMPTS
SUB_BATCH="${SUB_BATCH:-16}"                   # prompts generated per sampling call. Small ->
                                               # co-resident-safe; large (250+) -> saturates a
                                               # free GPU (diffusion sampling is launch-bound).
BF16="${BF16:-0}"                              # 1 -> bf16-autocast sampling (~2x; trained bf16)
MAX_GPU_GB="${MAX_GPU_GB:-12}"                 # hard cap on this process's GPU mem; eval OOMs
                                               # itself (not the training run) if exceeded.
                                               # Keep below the GPU's FREE memory. "" = uncapped.
OUT_DIR="${OUT_DIR:-}"                         # default: <ckpt_dir>/clip_eval_step<step>

# ---- guards -------------------------------------------------------------------------
[ -f "$CKPT" ] || { echo "ERROR: checkpoint not found: $CKPT" >&2; exit 1; }
[ -f "$CAPTIONS_FILE" ] || { echo "ERROR: captions not found: $CAPTIONS_FILE (run data/gen_baseline_captions.py)" >&2; exit 1; }

echo "[eval] GPU=$GPU ckpt=$CKPT"
echo "[eval] captions=$CAPTIONS_FILE  n_seeds=$N_SEEDS n_views=$N_VIEWS cfg=$CFG_SCALES${MAX_PROMPTS:+ max=$MAX_PROMPTS}${SHUFFLE_SEED:+ shuf=$SHUFFLE_SEED} mem_cap=${MAX_GPU_GB:-none}GiB"

CUDA_VISIBLE_DEVICES="$GPU" "$PY" jit/eval_clip_alignment.py \
    --ckpt "$CKPT" \
    --prompts_file "$CAPTIONS_FILE" \
    --n_seeds "$N_SEEDS" \
    --n_views "$N_VIEWS" \
    --cfg_scales "$CFG_SCALES" \
    --sub_batch "$SUB_BATCH" \
    ${MAX_GPU_GB:+--max_gpu_gb "$MAX_GPU_GB"} \
    $([ "$BF16" = 1 ] && echo --bf16) \
    ${MAX_PROMPTS:+--max_prompts "$MAX_PROMPTS"} \
    ${SHUFFLE_SEED:+--shuffle_seed "$SHUFFLE_SEED"} \
    ${OUT_DIR:+--out_dir "$OUT_DIR"}
