#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
_REQ_GPU="${HAM_GPU:-}"; . ./env.sh > /dev/null
export HAM_GPU="${_REQ_GPU:-${HAM_GPU:-0}}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BASE_ROOT="${HAM_OUTPUT_ROOT:-$PWD/outputs}"
NESTED="${NESTED_ROOT:-$PWD/outputs_nested}"
INNER="${INNER:-0.15}"; FOLDS="${FOLDS:-0 1 2 3 4}"
if [ "$(cd "$BASE_ROOT" 2>/dev/null && pwd)" = "$(cd "$NESTED" 2>/dev/null && pwd)" ]; then
    echo "refusing to run: nested root == reported root ($BASE_ROOT)"; exit 1; fi
PY="${HAM_PYTHON:-python}"
PKG="experiments"; [ -d improve ] && [ ! -d experiments ] && PKG="improve"
export HAM_OUTPUT_ROOT="$NESTED" HAM_INNER_VAL="$INNER"
mkdir -p "$NESTED/residual" logs/nested logs/batch_status
if [ -f "$BASE_ROOT/residual/cv_folds.csv" ] && [ ! -f "$NESTED/residual/cv_folds.csv" ]; then
    cp "$BASE_ROOT/residual/cv_folds.csv" "$NESTED/residual/cv_folds.csv"
    echo "copied cv_folds.csv (folds held identical)"; fi
ST="logs/batch_status/nested.txt"; T0=$(date +%s); DONE=0
NB=$(echo $FOLDS|wc -w); TOTAL=$NB
status () { { echo "batch=nested GPU=$HAM_GPU inner=$INNER $(date '+%m-%d %H:%M')"
  echo "progress: $DONE / $TOTAL  elapsed $(( ($(date +%s)-T0)/60 )) min"
  echo "current: $1"; echo "output: $NESTED (reported run in $BASE_ROOT untouched)"; } > "$ST"; }
echo "===== nested CV folds=$FOLDS inner=$INNER pkg=$PKG GPU=$HAM_GPU start $(date '+%m-%d %H:%M') ====="
# Table 3 varies the epoch-selection rule of the explanation head and nothing else, so it reads
# the same frozen classifier as the main protocol. The checkpoints are copied here rather than
# retrained: retraining them under HAM_INNER_VAL would change the classifier as well, which is a
# different experiment from the one the table reports.
for K in $FOLDS; do
  SRC="$BASE_ROOT/residual/flat_f${K}_s42.pth"; DST="$NESTED/residual/flat_f${K}_s42.pth"
  if [ -f "$DST" ]; then echo "--- [bb f$K] already in place"; DONE=$((DONE+1)); continue; fi
  if [ ! -f "$SRC" ]; then echo "--- [bb f$K] MISSING $SRC; train the main protocol first"; continue; fi
  status "classifier fold $K"
  cp "$SRC" "$DST" && DONE=$((DONE+1)) && echo "--- [bb f$K] classifier copied from the reported run"
done
status "done"; echo "===== end $(date '+%m-%d %H:%M') ====="
echo "this script only stages the frozen classifiers. The explanation heads of Table 3 are"
echo "then run against this root:  HAM_OUTPUT_ROOT=$NESTED HAM_INNER_VAL=$INNER $PY -m $PKG.head_swap --fold K --head all"
echo "HAM_INNER_VAL applies to the head alone, because the classifier is not retrained here."
