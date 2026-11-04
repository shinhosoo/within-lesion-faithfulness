#!/usr/bin/env bash
# Measure Supplementary Table S16's four baselines over the three explanation heads,
# with the folds split across two GPUs.
#
# Nothing is written beside the runs: the results go to OUT and the logs to LOGS,
# both outside the project, and perturb_heads.py refuses to write under the output
# root, the splits or the data in any case.
#
#   bash run_s16_heads.sh                  # all five folds, both GPUs
#   HEADS=sup bash run_s16_heads.sh        # one head
#   FOLDS_A=0 FOLDS_B=1 bash run_s16_heads.sh   # a short trial on two folds
#
# This script does not detach itself. For a long run, from outside:
#   mkdir -p ~/s16_heads_logs
#   nohup bash run_s16_heads.sh > ~/s16_heads_logs/run.out 2>&1 &
#
# Run it from the project root, beside SkinCancer/ and improve/.

set -u

OUT="${OUT:-$HOME/s16_heads}"
LOGS="${LOGS:-$HOME/s16_heads_logs}"
GPU_A="${GPU_A:-0}"
GPU_B="${GPU_B:-1}"
FOLDS_A="${FOLDS_A:-0,1,2}"
FOLDS_B="${FOLDS_B:-3,4}"
HEADS="${HEADS:-unsup,nomacro,sup}"
BASELINES="${BASELINES:-blur20,blur5,mean,noise}"
PY="${PY:-python}"

for v in HAM_OUTPUT_ROOT HAM_SPLIT_ROOT HAM_DATA_ROOT HAM_MASK_DIR; do
  if [ -z "${!v:-}" ]; then
    echo "$v is not set. The published runs need all four, and HAM_MASK_DIR especially:"
    echo "the within-lesion curves have nothing to rank without the lesion masks."
    exit 1
  fi
done

mkdir -p "$OUT" "$LOGS"
echo "out  -> $OUT"
echo "logs -> $LOGS"
echo "gpu $GPU_A takes folds $FOLDS_A, gpu $GPU_B takes folds $FOLDS_B"
echo "heads $HEADS, baselines $BASELINES"
echo

a="$LOGS/s16heads.gpu$GPU_A.log"
b="$LOGS/s16heads.gpu$GPU_B.log"

HAM_GPU="$GPU_A" PYTHONUNBUFFERED=1 $PY perturb_heads.py \
  --folds "$FOLDS_A" --heads "$HEADS" --baselines "$BASELINES" --out "$OUT" >"$a" 2>&1 &
pa=$!
HAM_GPU="$GPU_B" PYTHONUNBUFFERED=1 $PY perturb_heads.py \
  --folds "$FOLDS_B" --heads "$HEADS" --baselines "$BASELINES" --out "$OUT" >"$b" 2>&1 &
pb=$!
wait $pa; ra=$?
wait $pb; rb=$?
echo "   gpu $GPU_A exit=$ra  $a"
echo "   gpu $GPU_B exit=$rb  $b"
echo

# The published baseline has to reproduce the archive. If it did not, the script
# stopped, and that is the first thing to see.
grep -hE "REFUSED|max \|difference\||failed|Traceback" "$a" "$b" | sed 's/^/   /' | head -20
echo "   npz written: $(ls "$OUT"/*.npz 2>/dev/null | wc -l)"
echo

echo "== collected over every fold present"
$PY perturb_heads.py --collect "$OUT"
