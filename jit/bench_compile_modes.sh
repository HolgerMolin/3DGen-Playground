#!/bin/bash
# Benchmark torch.compile modes for JiT training throughput.
#
# Runs three arms of --max_steps 1000 each, single-GPU, no wandb, fresh process
# per arm. Parses Steps/sec from the log between step 500 and 1000 (skipping
# warmup + autotune) and prints a comparison table.
#
# Usage:   ./jit/bench_compile_modes.sh
# Output:  printed to stdout; full per-arm logs under output/bench_compile_<ts>/<mode>/
#
# Expected wall-clock: ~10 min per arm × 3 arms = ~30 min total. The first
# ~30–90s of each arm is compile/autotune time (excluded from sps measurement).

set -o pipefail

cd "$(dirname "$0")/.."   # repo root
# .env references LD_LIBRARY_PATH (and friends) that may be unset; source it
# without `-u` strictness so unset upstream vars don't abort the bench.
set +u
source .env
set +u

# Mirror the fallback chain in jit/train_gsplat.sh: prefer OBJ_LIST etc., fall
# back to DIT_GSPLAT_* set by .env. Without this, .env-only setups pass empty
# strings into the python script and it dies with FileNotFoundError.
OBJ_LIST=${OBJ_LIST:-${DIT_GSPLAT_OBJ_LIST:-}}
GS_DATA_PATH=${GS_DATA_PATH:-${DIT_GSPLAT_GS_PATH:-}}
MEAN_FILE=${MEAN_FILE:-${DIT_GSPLAT_MEAN_FILE:-}}
STD_FILE=${STD_FILE:-${DIT_GSPLAT_STD_FILE:-}}
CLASS_MAP_PATH=${CLASS_MAP_PATH:-${DIT_GSPLAT_CLASS_MAP:-}}
SPHERE2PLANE_PATH=${SPHERE2PLANE_PATH:-${DIT_GSPLAT_SPHERE2PLANE_PATH:-}}
REF_CAMERA_TAR=${REF_CAMERA_TAR:-${DIT_GSPLAT_REF_CAMERA_TAR:-}}
# Optional inputs (forwarded only when set). Both participate in the
# Class3DGenDataset cache hash, so omitting them when .env sets them
# silently builds a NEW lazy cache instead of attaching to the existing one.
EXCLUDE_KEYS_FILE=${EXCLUDE_KEYS_FILE:-}
RANK_TRANSFORM_FILE=${RANK_TRANSFORM_FILE:-}

# Fail early if any required path is empty/missing.
for v in OBJ_LIST GS_DATA_PATH MEAN_FILE STD_FILE CLASS_MAP_PATH SPHERE2PLANE_PATH REF_CAMERA_TAR; do
    val=${!v}
    if [ -z "$val" ]; then
        echo "Missing required path var: $v (check .env)" >&2
        exit 1
    fi
    if [ ! -e "$val" ]; then
        echo "Path for $v does not exist: $val" >&2
        exit 1
    fi
done
# Validate optional paths if provided.
for v in EXCLUDE_KEYS_FILE RANK_TRANSFORM_FILE; do
    val=${!v}
    if [ -n "$val" ] && [ ! -e "$val" ]; then
        echo "Optional path $v is set but does not exist: $val" >&2
        exit 1
    fi
done

MODES=("default" "max-autotune-no-cudagraphs" "reduce-overhead")
MAX_STEPS=1000

BENCH_ROOT="output/bench_compile_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BENCH_ROOT"
echo "Bench root: $BENCH_ROOT"
echo

# Activate venv once for the parent shell; subprocesses inherit.
source .3dgen/bin/activate

for MODE in "${MODES[@]}"; do
    ARM_DIR="$BENCH_ROOT/$MODE"
    mkdir -p "$ARM_DIR"
    LOG="$ARM_DIR/run.err"

    echo "==================================================================="
    echo "Bench arm: $MODE  →  $ARM_DIR"
    echo "==================================================================="

    # For reduce-overhead, also surface recompile/graph-break events so we can
    # tell if CUDA Graphs are sticking. No-op for the other modes.
    EXTRA_ENV=()
    if [ "$MODE" = "reduce-overhead" ]; then
        EXTRA_ENV+=(TORCH_LOGS=recompiles,graph_breaks)
    fi

    # Forward optional cache-key-affecting args only when their .env vars are set.
    OPTIONAL_ARGS=()
    if [ -n "$EXCLUDE_KEYS_FILE" ]; then
        OPTIONAL_ARGS+=(--exclude_keys_file "$EXCLUDE_KEYS_FILE")
    fi
    if [ -n "$RANK_TRANSFORM_FILE" ]; then
        OPTIONAL_ARGS+=(--rank_transform_file "$RANK_TRANSFORM_FILE")
    fi

    START=$(date +%s)
    env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "${EXTRA_ENV[@]}" \
        python jit/train_gsplat.py \
        --config jit/configs/jit_train_gsplat.yaml \
        --obj_list "$OBJ_LIST" \
        --gs_path "$GS_DATA_PATH" \
        --mean_file "$MEAN_FILE" \
        --std_file "$STD_FILE" \
        --class_map "$CLASS_MAP_PATH" \
        --sphere2plane_path "$SPHERE2PLANE_PATH" \
        --ref_camera_tar "$REF_CAMERA_TAR" \
        --mixed_precision bf16 \
        --results_dir "$ARM_DIR" \
        --compile_mode "$MODE" \
        --max_steps "$MAX_STEPS" \
        --no-wandb \
        "${OPTIONAL_ARGS[@]}" \
        2> "$LOG" \
        || echo "  (arm $MODE exited non-zero; continuing — partial log retained)"
    ELAPSED=$(( $(date +%s) - START ))
    echo "  wall-clock: ${ELAPSED}s"
    echo
done

echo "==================================================================="
echo "Results  (steady-state Steps/sec, averaged over step 500–${MAX_STEPS})"
echo "==================================================================="
printf "%-32s %12s %12s %12s\n" "compile_mode" "mean_sps" "n_samples" "speedup"
printf "%-32s %12s %12s %12s\n" "------------" "--------" "---------" "-------"

# First pass: collect mean sps per arm
declare -A SPS_BY_MODE
for MODE in "${MODES[@]}"; do
    LOG="$BENCH_ROOT/$MODE/run.err"
    if [ ! -f "$LOG" ]; then
        SPS_BY_MODE[$MODE]="MISSING"
        continue
    fi
    # Extract "<step> <sps>" pairs, then filter to step >= 500 and average.
    READ=$(sed -n 's/.*Step\s\+\([0-9]\+\).*Steps\/sec:\s*\([0-9.]\+\).*/\1 \2/p' "$LOG" \
        | awk '$1>=500 {sum+=$2; n++} END {if(n>0) printf "%.3f %d", sum/n, n; else print "NA 0"}')
    SPS_BY_MODE[$MODE]="$READ"
done

# Baseline for speedup column
BASELINE_SPS=$(echo "${SPS_BY_MODE[default]}" | awk '{print $1}')

for MODE in "${MODES[@]}"; do
    READ=${SPS_BY_MODE[$MODE]}
    MEAN=$(echo "$READ" | awk '{print $1}')
    N=$(echo "$READ" | awk '{print $2}')
    if [ "$MEAN" = "NA" ] || [ "$MEAN" = "MISSING" ]; then
        printf "%-32s %12s %12s %12s\n" "$MODE" "$MEAN" "-" "-"
        continue
    fi
    if [ -n "$BASELINE_SPS" ] && [ "$BASELINE_SPS" != "NA" ] && [ "$BASELINE_SPS" != "MISSING" ]; then
        SPEEDUP=$(awk -v a="$MEAN" -v b="$BASELINE_SPS" 'BEGIN {printf "%+.1f%%", (a/b - 1) * 100}')
    else
        SPEEDUP="-"
    fi
    printf "%-32s %12s %12s %12s\n" "$MODE" "$MEAN" "$N" "$SPEEDUP"
done

echo
echo "Notes:"
echo "  - Per-arm log: $BENCH_ROOT/<mode>/run.err"
echo "  - For reduce-overhead, grep '$BENCH_ROOT/reduce-overhead/run.err' for"
echo "    'Recompile' or 'graph break'. If those appear past step ~50 the"
echo "    CUDA Graph isn't sticking and the win will be small or negative."
echo "  - n_samples is the number of log windows averaged (log_every=250 →"
echo "    expect 3 windows: 500, 750, 1000)."
