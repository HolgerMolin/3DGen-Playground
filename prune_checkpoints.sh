#!/usr/bin/env bash
# Per-run checkpoint retention. For each run dir given as an arg, KEEP exactly:
#   (1) the latest step-numbered checkpoint  (resume point), and
#   (2) the last recon-only checkpoint = highest-step ckpt saved BEFORE render engaged
#       (render onset = first step where "Render_L1: <x>" with x>0 appears in the run's log).
#       If render never engaged, (2) == (1), so only the latest is kept.
# DELETE every other step-numbered "<digits>.pt". NEVER deletes non-step .pt (ema_*, best, latest).
# Dry-run with DRY_RUN=1. Designed to be run by parallel agents on disjoint dir slices.
set -uo pipefail
DRY_RUN="${DRY_RUN:-0}"
total_freed=0; total_del=0
for d in "$@"; do
  [ -d "$d" ] || { echo "[skip missing] $d"; continue; }
  mapfile -t ckpts < <(ls "$d"/*.pt 2>/dev/null | grep -E '/[0-9]+\.pt$' | sort -t/ -k2 -n)
  if [ "${#ckpts[@]}" -le 1 ]; then echo "[$d] ${#ckpts[@]} step-ckpt(s), nothing to prune"; continue; fi
  onset=$(grep -hoE "Step +[0-9]+ \|.*Render_L1: [0-9.]+" "$d"/*.log "$d"/*.err 2>/dev/null \
          | gawk 'match($0,/Step +([0-9]+)/,s) && match($0,/Render_L1: ([0-9.]+)/,r){ if(r[1]+0>0){print s[1]; exit} }')
  [ -z "$onset" ] && onset=999999999
  latest_f="${ckpts[-1]}"
  recon_f=""
  for f in "${ckpts[@]}"; do s=$(basename "$f" .pt); s=$((10#$s)); [ "$s" -lt "$onset" ] && recon_f="$f"; done
  [ -z "$recon_f" ] && recon_f="$latest_f"
  keep="|$latest_f|$recon_f|"
  freed=0; ndel=0
  for f in "${ckpts[@]}"; do
    case "$keep" in *"|$f|"*) continue;; esac
    sz=$(stat -c%s "$f" 2>/dev/null || echo 0)
    if [ "$DRY_RUN" = 1 ]; then echo "  WOULD rm $f ($((sz/1000000))MB)"; else rm -f "$f"; fi
    freed=$((freed+sz)); ndel=$((ndel+1))
  done
  total_freed=$((total_freed+freed)); total_del=$((total_del+ndel))
  printf "[%s] onset=%s keep={%s, %s} deleted=%d freed=%dMB\n" \
    "$d" "$onset" "$(basename "$recon_f")" "$(basename "$latest_f")" "$ndel" "$((freed/1000000))"
done
printf "TOTAL: deleted=%d files, freed=%dMB\n" "$total_del" "$((total_freed/1000000))"
