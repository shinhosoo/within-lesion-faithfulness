#!/usr/bin/env bash
# Four saliency mechanisms on one frozen backbone, measured against the same lesion masks.
#   bash run_head_swap.sh check     # steps 1 and 2 only
#   bash run_head_swap.sh           # steps 1, 2 and 3
#   bash run_head_swap.sh 0 1       # only these folds, after check has passed
set -euo pipefail
cd "$(dirname "$0")"

# env.sh assigns its own defaults to the HAM_* paths, so sourcing it discards an override the
_keep_data="${HAM_DATA_ROOT:-}"
_keep_mask="${HAM_MASK_DIR:-}"
source env.sh
[ -n "$_keep_data" ] && export HAM_DATA_ROOT="$_keep_data"
[ -n "$_keep_mask" ] && export HAM_MASK_DIR="$_keep_mask"
if [ -n "$_keep_data" ] || [ -n "$_keep_mask" ]; then
  echo "  [override] HAM_DATA_ROOT = $HAM_DATA_ROOT"
  echo "  [override] HAM_MASK_DIR  = $HAM_MASK_DIR"
fi

echo "=============================================================="
echo " 1. what each head costs            (no GPU, no data needed)"
echo "=============================================================="
python -m improve.head_swap --count

echo
echo "=============================================================="
echo " 2. smoke test                      (one fold, a few images)"
echo "=============================================================="
python -m improve.head_swap --fold 0 --head all --limit 70 --epochs 1 --quiet

if [ "${1:-}" = "check" ]; then
  echo
  echo "Both checks passed. The smoke test wrote its own separately named cache and result,"
  echo "so nothing it left is in the way. Run"
  echo "  bash run_head_swap.sh"
  echo "to do the folds."
  exit 0
fi

FOLDS=("$@")
if [ ${#FOLDS[@]} -eq 0 ]; then FOLDS=(0 1 2 3 4); fi

echo
echo "=============================================================="
echo " 3. folds ${FOLDS[*]}"
echo "=============================================================="
# No --rebuild-cache: a cache already on disk is reused after its row count is checked against
# the split, so an interrupted run resumes instead of re-encoding the backbone.
for F in "${FOLDS[@]}"; do
  echo
  echo "--- fold $F ---"
  python -m improve.head_swap --fold "$F" --head all --quiet
done

echo
echo "=============================================================="
echo " the table"
echo "=============================================================="
python -m improve.head_swap --summary

echo
echo
echo "The caches under \$HAM_OUTPUT_ROOT/headswap/cache are large and are only needed to"
echo "re-train a head. The hs_*.npz results do not depend on them and can be kept alone:"
echo "  rm -rf \"\$HAM_OUTPUT_ROOT/headswap/cache\""
