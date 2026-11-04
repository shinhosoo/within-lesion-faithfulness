#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
_REQ_GPU="${HAM_GPU:-}"
. ./env.sh > /dev/null
export HAM_GPU="${_REQ_GPU:-0}"
mkdir -p logs/hier
echo "===== GPU0 parallel experiment   GPU=$HAM_GPU   starting $(date '+%H:%M:%S') ====="

echo; echo "##### 1-stage seeds 43, 44 #####"
for SD in 43 44; do
  echo "--- flat s$SD  $(date '+%H:%M:%S')"
  python -m improve.flat_seed --seed $SD > "logs/hier/flat_s${SD}.log" 2>&1 \
    || { echo "  failed:"; tail -20 "logs/hier/flat_s${SD}.log"; exit 1; }
  grep "acc=" "logs/hier/flat_s${SD}.log" | sed 's/^/    /'
done

echo; echo " finished $(date '+%H:%M:%S')"
