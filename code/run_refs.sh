#!/usr/bin/env bash
# Measure the reference maps on the within-lesion test, starting when a GPU is free enough.
#   bash run_refs.sh                 # wait up to 3 h for a quiet card, then run anyway
#   bash run_refs.sh --now           # start immediately on the least-loaded card
#   bash run_refs.sh --wait-hours 8  # be more patient
set -uo pipefail
cd "$(dirname "$0")"

NEED_FREE_MIB=${NEED_FREE_MIB:-8000}   # our job peaks well under this; the margin is courtesy
MAX_UTIL=${MAX_UTIL:-40}               # percent; above this we would be halving someone's run
POLL_SECONDS=${POLL_SECONDS:-300}
WAIT_HOURS=3
START_NOW=0
while [ $# -gt 0 ]; do
  case "$1" in
    --now) START_NOW=1; shift ;;
    --wait-hours) WAIT_HOURS="$2"; shift 2 ;;
    *) echo "unknown option: $1"; exit 2 ;;
  esac
done

mkdir -p logs
exec > >(tee -a logs/refs.log) 2>&1
echo "=== $(date) : waiting for a card (need ${NEED_FREE_MIB} MiB free, util <= ${MAX_UTIL}%) ==="

# A card is "quiet" when it has room and is not already saturated. Memory alone is not enough:
pick_quiet() {
  nvidia-smi --query-gpu=index,memory.total,memory.used,utilization.gpu \
             --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', *' -v need="$NEED_FREE_MIB" -v maxu="$MAX_UTIL" \
        '{ free = $2 - $3; if (free >= need && $4 <= maxu) { print $1; exit } }'
}

pick_least_loaded() {
  nvidia-smi --query-gpu=index,memory.total,memory.used \
             --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', *' '{ free = $2 - $3; if (free > best) { best = free; idx = $1 } } END { print idx }'
}

DEADLINE=$(( $(date +%s) + WAIT_HOURS * 3600 ))
GPU=""
if [ "$START_NOW" = "1" ]; then
  GPU=$(pick_least_loaded)
  echo "--now given; taking the least-loaded card: GPU $GPU"
else
  while :; do
    GPU=$(pick_quiet)
    if [ -n "$GPU" ]; then
      echo "$(date +%H:%M) : GPU $GPU is quiet; starting"
      break
    fi
    if [ "$(date +%s)" -ge "$DEADLINE" ]; then
      GPU=$(pick_least_loaded)
      echo "$(date +%H:%M) : waited ${WAIT_HOURS} h without a quiet card; taking GPU $GPU anyway"
      break
    fi
    sleep "$POLL_SECONDS"
  done
fi

if [ -z "$GPU" ]; then
  echo "no GPU reported by nvidia-smi; stopping"
  exit 1
fi

# env.sh assigns its own defaults to these, so an override the caller exported is saved across
_keep_data="${HAM_DATA_ROOT:-}"
_keep_mask="${HAM_MASK_DIR:-}"
source env.sh
[ -n "$_keep_data" ] && export HAM_DATA_ROOT="$_keep_data"
[ -n "$_keep_mask" ] && export HAM_MASK_DIR="$_keep_mask"
echo "  HAM_DATA_ROOT = $HAM_DATA_ROOT"
echo "  HAM_MASK_DIR  = $HAM_MASK_DIR"
if [ ! -d "$HAM_DATA_ROOT" ]; then
  echo "HAM_DATA_ROOT does not exist; export it before running this and try again"
  exit 1
fi
if [ ! -d "$HAM_MASK_DIR" ]; then
  echo "HAM_MASK_DIR does not exist; export it before running this and try again"
  exit 1
fi

O="${HAM_OUTPUT_ROOT:-outputs}/headswap"
for F in 0 1 2 3 4; do
  if [ -f "$O/refs_f${F}.npz" ]; then
    echo "=== fold $F already measured; skipping ==="
    continue
  fi
  echo "=== $(date +%H:%M) : fold $F on GPU $GPU ==="
  HAM_GPU="$GPU" python -m improve.within_refs --fold "$F" || {
    echo "fold $F failed; stopping so the failure is not buried"
    exit 1
  }
done

echo "=== $(date) : done ==="
echo
echo "The centered Gaussian is the check: its two values must match what head_swap.py reported"
echo "for the same fold (+0.2038 whole-image and +0.0165 within on fold 0). Two independent"
echo "implementations agreeing on them is what makes the Grad-CAM row worth reading."
