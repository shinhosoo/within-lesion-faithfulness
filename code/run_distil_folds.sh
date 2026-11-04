#!/usr/bin/env bash
# Attribution distillation on the remaining folds, at the grid the fold-0 results used.
#   HAM_GPU=2 bash run_distil_folds.sh
#   HAM_GPU=2 bash run_distil_folds.sh --grid 12 --folds "1 2"
set -uo pipefail
cd "$(dirname "$0")"

GRID=6
FOLDS="1 2 3 4"
LAM_ATTR=1
LAM_MASK=0.2
while [ $# -gt 0 ]; do
  case "$1" in
    --grid) GRID="$2"; shift 2 ;;
    --folds) FOLDS="$2"; shift 2 ;;
    *) echo "unknown option: $1"; exit 2 ;;
  esac
done

mkdir -p logs
exec > >(tee -a logs/distil_folds.log) 2>&1

# The caller's data paths have to survive env.sh, which supplies its own defaults for them, and
# HAM_GPU is preserved for a sharper reason: env.sh prints a value for it, and whether that is a
_keep_data="${HAM_DATA_ROOT:-}"
_keep_mask="${HAM_MASK_DIR:-}"
_keep_out="${HAM_OUTPUT_ROOT:-}"
_keep_gpu="${HAM_GPU:-}"
# Checked before the source, not after: env.sh supplies a value for HAM_GPU, so afterwards a
if [ -z "$_keep_gpu" ]; then
  echo "set HAM_GPU to the card to use, e.g. HAM_GPU=2 bash $0"
  exit 1
fi
source env.sh
[ -n "$_keep_data" ] && export HAM_DATA_ROOT="$_keep_data"
[ -n "$_keep_mask" ] && export HAM_MASK_DIR="$_keep_mask"
[ -n "$_keep_out" ] && export HAM_OUTPUT_ROOT="$_keep_out"
[ -n "$_keep_gpu" ] && export HAM_GPU="$_keep_gpu"

for v in HAM_DATA_ROOT HAM_MASK_DIR HAM_SPLIT_ROOT HAM_OUTPUT_ROOT; do
  eval "p=\${$v:-}"
  echo "  $v = $p"
  if [ ! -d "$p" ]; then
    echo "$v does not exist; export it before running this and try again"
    exit 1
  fi
done

# A grid-6 result written into the grid-12 tree, or the reverse, cannot be told apart afterwards:
case "$HAM_OUTPUT_ROOT" in
  *grid12*)
    if [ "$GRID" != "12" ]; then
      echo "HAM_OUTPUT_ROOT points at the grid-12 tree but --grid is $GRID."
      echo "Results would be indistinguishable from the other grid. Fix one of them."
      exit 1
    fi ;;
  *)
    if [ "$GRID" = "12" ]; then
      echo "--grid 12 with HAM_OUTPUT_ROOT = $HAM_OUTPUT_ROOT would overwrite the grid-6"
      echo "targets and heads, which share the same filenames. Point it at a separate tree."
      exit 1
    fi ;;
esac

echo "=== $(date) : grid $GRID, folds $FOLDS, GPU $HAM_GPU, out $HAM_OUTPUT_ROOT ==="

D="$HAM_OUTPUT_ROOT/distil"
for F in $FOLDS; do
  # A and B differ only in whether a mask is used, so each needs its own target cache; B's
  # carries the _free suffix. Both stages are skipped when their output is already there.
  for FREE in "" "--mask-free"; do
    SUF=""
    [ -n "$FREE" ] && SUF="_free"
    if [ -f "$D/targets/f${F}_train${SUF}.npz" ] && [ -f "$D/targets/f${F}_val${SUF}.npz" ]; then
      echo "=== $(date +%H:%M) : fold $F targets${SUF} present; skipping ==="
    else
      echo "=== $(date +%H:%M) : fold $F targets${SUF} ==="
      python -m improve.distil --stage targets --fold "$F" --grid "$GRID" $FREE || {
        echo "fold $F targets${SUF} failed; stopping so the failure is not buried"
        exit 1
      }
    fi
  done

  for FREE in "" "--mask-free"; do
    if [ -n "$FREE" ]; then
      TAG="mlp_attr${LAM_ATTR}_mask0_free_f${F}"
      ARGS="--lam-attr $LAM_ATTR --lam-mask 0 --mask-free"
    else
      TAG="mlp_attr${LAM_ATTR}_mask${LAM_MASK}_f${F}"
      ARGS="--lam-attr $LAM_ATTR --lam-mask $LAM_MASK"
    fi
    if [ -f "$D/ds_${TAG}.npz" ]; then
      echo "=== $(date +%H:%M) : $TAG present; skipping ==="
      continue
    fi
    echo "=== $(date +%H:%M) : fold $F train $TAG ==="
    python -m improve.distil --stage train --fold "$F" --grid "$GRID" $ARGS || {
      echo "fold $F train $TAG failed; stopping"
      exit 1
    }
  done
done

echo "=== $(date) : done ==="
echo
echo "The number to read is the within-lesion value of the mask-free rows (B). On fold 0 it was"
echo "-0.0037 while B's whole-image score reached +0.2089, against the centered Gaussian's"
echo "+0.2038 on the same fold. If that pattern holds across folds, the whole-image test can be"
echo "driven to a learning-free prior's level with no annotation while the within-lesion test"
echo "stays at zero, which is what makes the new column worth reporting."
