#!/usr/bin/env bash
# Reproduce key results on a second backbone via improve/bb_run.py (originals untouched).
#   STEP=flat HAM_GPU=0 nohup bash run_backbone.sh > logs/batch_B1.log 2>&1 &
set -u
cd "$(dirname "$0")"
_REQ_GPU="${HAM_GPU:-}"
. ./env.sh > /dev/null
export HAM_GPU="${_REQ_GPU:-${HAM_GPU:-0}}"
export HAM_MASK_DIR="${HAM_MASK_DIR:-$PWD/masks_ham10000}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BB="${BB:-convnext_tiny}"; OUT="$PWD/outputs_${BB}"
STEP="${STEP:-flat}"; FOLDS="${FOLDS:-0 1 2 3 4}"; SEED="${SEED:-42}"
mkdir -p "$OUT/residual" logs/bb logs/batch_status logs/batch_summary
cp -n "$PWD/outputs/residual/cv_folds.csv" "$OUT/residual/cv_folds.csv" 2>/dev/null   # use the same folds
R() { python -m improve.bb_run --backbone "$BB" --out "$OUT" -- "$@"; }
ST="logs/batch_status/bb_${STEP}.txt"; T0=$(date +%s); DONE=0; TOTAL=$(echo $FOLDS | wc -w)
status () { { echo "batch=bb_$STEP ($BB)  GPU=$HAM_GPU  updated $(date '+%m-%d %H:%M')"
              echo "progress: $DONE / $TOTAL folds   elapsed $(( ($(date +%s)-T0)/60 ))min"; echo "current: $1"
              echo "outputs: $OUT (separate from the existing outputs)"; } > "$ST"; }
echo "===== second backbone [$BB] step=$STEP GPU=$HAM_GPU starting $(date '+%m-%d %H:%M') ====="
for K in $FOLDS; do
  case "$STEP" in
    flat)
      [ -f "$OUT/residual/flat_f${K}_s${SEED}_val_predictions.npz" ] && { DONE=$((DONE+1)); continue; }
      status "1-stage fold $K"
      R improve.residual --model flat --fold $K --seed $SEED > logs/bb/flat_f$K.log 2>&1 \
        || { echo "--- [flat f$K] failed"; tail -8 logs/bb/flat_f$K.log; continue; }
      echo "--- [flat f$K] $(grep -o 'val acc=[0-9.]*' logs/bb/flat_f$K.log)" ;;
  esac
  DONE=$((DONE+1)); status "preparing next fold"
done
status "done"; echo " finished $(date '+%m-%d %H:%M')"
