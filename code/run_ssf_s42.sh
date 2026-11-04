#!/usr/bin/env bash
# The seed-42 pooled reading of ssf, which is the one value the Table S10 correction lacks.
#   nohup bash run_ssf_s42.sh > logs/ssf_s42.log 2>&1 &
set -u

STEM=ssf
NET=outputs/headswap
OUTD=outputs/headswap_s42_pooled
CK=outputs/headswap/ckpt_ssf_before
HOLD=outputs/headswap_hold_ssf
mkdir -p logs "$OUTD" "$CK" "$HOLD"

[ -f env.sh ] && . ./env.sh
for v in HAM_DATA_ROOT HAM_SPLIT_ROOT HAM_MASK_DIR; do
    [ -z "${!v:-}" ] && { echo "ABORT: $v is not set."; exit 1; }
done
if pgrep -f 'improve\.(head_swap|cond_head|deep_head)' >/dev/null 2>&1; then
    echo "ABORT: another run is in progress; it writes the same filenames."
    exit 1
fi

echo "  preflight"
n=$(ls "$NET"/pre_retrain/hs_${STEM}_f[0-4].npz 2>/dev/null | wc -l)
echo "    published archive in pre_retrain/      ${n}/5"
[ "$n" -eq 5 ] || { echo "  ABORT: the row this is checked against is not there."; exit 1; }
n=$(ls "$NET"/head_${STEM}_f[0-4].pth 2>/dev/null | wc -l)
if [ "$n" -eq 5 ]; then
    cp -p "$NET"/head_${STEM}_f[0-4].pth "$CK"/
    echo "    current checkpoints (seed 44) kept in ${CK}/   $(ls "$CK" | wc -l)/5"
fi
moved=0
for pat in "hs_${STEM}_f" "hs_${STEM}_mapsoftmax_f"; do
    for f in 0 1 2 3 4; do
        [ -e "${NET}/${pat}${f}.npz" ] && { mv "${NET}/${pat}${f}.npz" "$HOLD"/; moved=$((moved+1)); }
    done
done
echo "    ${moved} archive(s) held in ${HOLD}/"

restore () {
    local rc=$?
    ls "$HOLD"/*.npz >/dev/null 2>&1 && mv "$HOLD"/*.npz "$NET"/ 2>/dev/null
    rmdir "$HOLD" 2>/dev/null
    ls "$CK"/*.pth >/dev/null 2>&1 && cp -p "$CK"/head_${STEM}_f[0-4].pth "$NET"/ 2>/dev/null
    echo "  [restore] archives and the seed-44 checkpoints are back in ${NET}/"
    exit $rc
}
trap restore EXIT INT TERM
echo

echo "  A. training ${STEM} at seed 42, five folds across four GPUs"
for f in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$f python -m improve.head_swap --head "$STEM" --fold "$f" --seed 42 \
        > "logs/${STEM}_s42_f${f}_train.log" 2>&1 &
done
wait
CUDA_VISIBLE_DEVICES=0 python -m improve.head_swap --head "$STEM" --fold 4 --seed 42 \
    > "logs/${STEM}_s42_f4_train.log" 2>&1
n=$(ls "$NET"/hs_${STEM}_f[0-4].npz 2>/dev/null | wc -l)
echo "    ${n}/5 archives written (rectified reading, straight from training)"
[ "$n" -eq 5 ] || { echo "  ABORT: training did not produce five archives."; exit 1; }

echo
echo "  B. does the retrained head reproduce the published row?"
python - <<'PY'
import glob
import numpy as np
def rd(pat):
    C = [[] for _ in range(5)]
    for p in sorted(glob.glob(pat)):
        with np.load(p, allow_pickle=True) as z:
            R = z["metrics"][:, 0, :]
            v = np.asarray(z["faith_head"], dtype=float).ravel()
            for k, x in enumerate((np.nanmean(R[:, 0]), np.nanmean(R[:, 1]),
                                   np.nanmean(R[:, 2]), v[2], v[5])):
                C[k].append(float(x))
    return C
def m(c):
    return sum(c) / len(c)
A = rd("outputs/headswap/pre_retrain/hs_ssf_f*.npz")
B = rd("outputs/headswap/hs_ssf_f*.npz")
names = ("pointing", "energy", "IoU", "Ins-Del", "within")
print(f"    {'measure':<10}{'published':>12}{'retrained':>12}{'diff':>10}")
worst = 0.0
for k, nm in enumerate(names):
    d = m(B[k]) - m(A[k])
    worst = max(worst, abs(d))
    print(f"    {nm:<10}{m(A[k]):>+12.4f}{m(B[k]):>+12.4f}{d:>+10.4f}")
print()
if worst < 0.005:
    print(f"    reproduces to within {worst:.4f} on every column, so the retrained checkpoint")
    print(f"    stands in for the published one and its pooled reading is a seed-42 value.")
else:
    print(f"    does NOT reproduce: the largest difference is {worst:.4f}. The pooled value")
    print(f"    below is then a property of THIS run, and the table should use the seed-44")
    print(f"    measurement with the seed stated instead.")
PY

echo
echo "  C. the pooled reading of the same checkpoints"
for f in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$f python -m improve.head_swap --head "$STEM" --fold "$f" \
        --measure-only --map softmax > "logs/${STEM}_s42_f${f}_sm.log" 2>&1 &
done
wait
CUDA_VISIBLE_DEVICES=0 python -m improve.head_swap --head "$STEM" --fold 4 \
    --measure-only --map softmax > "logs/${STEM}_s42_f4_sm.log" 2>&1
mkdir -p "$OUTD"
mv "$NET"/hs_${STEM}_mapsoftmax_f[0-4].npz "$OUTD"/ \
    && echo "    5 archives -> ${OUTD}/"
# the rectified archives of this run are kept too, under a name that says what they are
mkdir -p outputs/headswap_s42_rect
mv "$NET"/hs_${STEM}_f[0-4].npz outputs/headswap_s42_rect/ 2>/dev/null

trap - EXIT INT TERM
ls "$HOLD"/*.npz >/dev/null 2>&1 && mv "$HOLD"/*.npz "$NET"/
rmdir "$HOLD" 2>/dev/null
ls "$CK"/*.pth >/dev/null 2>&1 && cp -p "$CK"/head_${STEM}_f[0-4].pth "$NET"/
echo "  D. restored"

python - <<'PY'
import glob
import numpy as np
def sd(v):
    m = sum(v) / len(v)
    return (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5
def row(pat):
    C = [[] for _ in range(5)]
    for p in sorted(glob.glob(pat)):
        with np.load(p, allow_pickle=True) as z:
            R = z["metrics"][:, 0, :]
            v = np.asarray(z["faith_head"], dtype=float).ravel()
            for k, x in enumerate((np.nanmean(R[:, 0]), np.nanmean(R[:, 1]),
                                   np.nanmean(R[:, 2]), v[2], v[5])):
                C[k].append(float(x))
    if not C[0]:
        return None
    return "  ".join(f"{np.mean(c):+.3f} ± {sd(c):.3f}" if k >= 3
                     else f"{np.mean(c):.3f} ± {sd(c):.3f}" for k, c in enumerate(C))
print()
for tag, pat in (("ssf seed 42, pooled  ", "outputs/headswap_s42_pooled/hs_ssf_mapsoftmax_f*.npz"),
                 ("ssf seed 44, pooled  ", "outputs/headswap_s44/hs_ssf_mapsoftmax_f*.npz"),
                 ("ssf seed 42, rectified", "outputs/headswap/pre_retrain/hs_ssf_f*.npz")):
    r = row(pat)
    print(f"  {tag}  {r if r else 'missing'}")
print()
print("  The first line is what replaces Table S10's 'Scale and shift, per channel' row if")
print("  section B reproduced; the third is what that row currently prints and belongs in a")
print("  footnote as the rectified reading of the same head.")
PY
echo
echo "  ALL DONE  $(date)"
