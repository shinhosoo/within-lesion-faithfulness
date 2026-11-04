#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
_REQ_GPU="${HAM_GPU:-}"
. ./env.sh > /dev/null
export HAM_GPU="${_REQ_GPU:-${HAM_GPU:-1}}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODELS="${MODELS:-flat}"
FOLDS="${FOLDS:-0 1 2 3 4}"
SEED="${SEED:-42}"
mkdir -p logs/residual outputs/residual
echo "===== classifier CV   GPU=$HAM_GPU   models=$MODELS   folds=$FOLDS   starting $(date '+%H:%M:%S') ====="
for M in $MODELS; do
  for K in $FOLDS; do
    TAG="${M}_f${K}_s${SEED}"; LOG="logs/residual/${TAG}.log"
    if [ -f "outputs/residual/${TAG}_val_predictions.npz" ]; then echo "--- [$TAG] already done, skipped"; continue; fi
    echo "--- [$TAG]  $(date '+%H:%M:%S')"
    python -m improve.residual --model "$M" --fold "$K" --seed "$SEED" > "$LOG" 2>&1 \
      || { echo "  failed:"; tail -20 "$LOG" | sed 's/^/    /'; exit 1; }
    grep -o "\[residual.*val acc=[0-9.]*" "$LOG" | sed 's/^/    /'
  done
done
echo " finished $(date '+%H:%M:%S')"
