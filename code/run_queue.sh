#!/usr/bin/env bash
# The overnight queue: the reference maps that need no training, on every fold.
#   HAM_GPU=3 bash run_queue.sh
#   HAM_GPU=3 bash run_queue.sh --folds "1 2" --ig-steps 16
set -uo pipefail
cd "$(dirname "$0")"

FOLDS="0 1 2 3 4"
IG_STEPS=32
while [ $# -gt 0 ]; do
  case "$1" in
    --folds) FOLDS="$2"; shift 2 ;;
    --ig-steps) IG_STEPS="$2"; shift 2 ;;
    *) echo "unknown option: $1"; exit 2 ;;
  esac
done

mkdir -p logs
exec > >(tee -a logs/queue.log) 2>&1

# env.sh supplies its own defaults for the data paths and they do not exist on this machine, so
_keep_data="${HAM_DATA_ROOT:-}"
_keep_mask="${HAM_MASK_DIR:-}"
_keep_gpu="${HAM_GPU:-}"
if [ -z "$_keep_gpu" ]; then
  echo "set HAM_GPU to the card to use, e.g. HAM_GPU=3 bash $0"
  exit 1
fi
source env.sh
[ -n "$_keep_data" ] && export HAM_DATA_ROOT="$_keep_data"
[ -n "$_keep_mask" ] && export HAM_MASK_DIR="$_keep_mask"
export HAM_GPU="$_keep_gpu"

# HAM_OUTPUT_ROOT is deliberately not preserved: these are reference maps for the reported
case "${HAM_OUTPUT_ROOT:-}" in
  *grid12*)
    echo "HAM_OUTPUT_ROOT points at the grid-12 tree; these rows belong in the main tree."
    echo "unset HAM_OUTPUT_ROOT, source env.sh again, and rerun."
    exit 1 ;;
esac

for v in HAM_DATA_ROOT HAM_MASK_DIR HAM_SPLIT_ROOT HAM_OUTPUT_ROOT; do
  eval "p=\${$v:-}"
  echo "  $v = $p"
  if [ ! -d "$p" ]; then
    echo "$v does not exist; export it before running this and try again"
    exit 1
  fi
done

echo "=== $(date) : folds $FOLDS on GPU $HAM_GPU, IG $IG_STEPS steps ==="
O="$HAM_OUTPUT_ROOT/headswap"
for F in $FOLDS; do
  if [ -f "$O/refs2_f${F}.npz" ]; then
    echo "=== fold $F already measured; skipping ==="
    continue
  fi
  echo "=== $(date +%H:%M) : fold $F ==="
  python -m improve.within_refs2 --fold "$F" --ig-steps "$IG_STEPS" || {
    echo "fold $F failed; stopping so the failure is not buried"
    exit 1
  }
done

echo "=== $(date) : done ==="
echo
echo "Read the two checks before anything else. On fold 0 the centered Gaussian must come out at"
echo "+0.2038 whole-image and +0.0165 within, which head_swap.py and within_refs.py both report,"
echo "and the lesion-mask row must have an IoU of 1.0000 with a within value near zero. If either"
echo "is wrong the Grad-CAM depths and the Integrated Gradients row mean nothing."
