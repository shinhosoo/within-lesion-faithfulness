#!/usr/bin/env bash
# The third readout seed of the lowrank head, to close the question seeds 42 and 44 opened.
#   nohup bash run_lowrank_seed43.sh > logs/seed43.log 2>&1 &
set -u
cd "$(dirname "$(readlink -f "$0")")" || exit 1

SEED=43
STEM=lowrank
S43=outputs/headswap_s43
HOLD=outputs/headswap_s42_hold
CKPT=outputs/headswap/ckpt_s44
NET=outputs/headswap
mkdir -p logs "$S43"

[ -f env.sh ] && . ./env.sh
for v in HAM_DATA_ROOT HAM_SPLIT_ROOT HAM_MASK_DIR; do
    [ -z "${!v:-}" ] && { echo "ABORT: $v is not set."; exit 1; }
done

# ---- preflight: every file this script will move or overwrite has a surviving copy ----
echo "  preflight"
fail=0
n=$(ls "$CKPT"/head_${STEM}_f[0-4].pth 2>/dev/null | wc -l)
echo "    seed-44 checkpoints in ckpt_s44/        ${n}/5"
[ "$n" -eq 5 ] || { echo "      training overwrites head_${STEM}_f*.pth and these are the only"
                    echo "      copy of seed 44; refusing."; fail=1; }
n=$(ls "$NET"/pre_retrain2/hs_${STEM}_mapsoftmax_f[0-4].npz 2>/dev/null | wc -l)
echo "    seed-42 backup in pre_retrain2/         ${n}/5"
[ "$n" -eq 5 ] || { echo "      the measuring pass writes over the published archives and this"
                    echo "      is the backstop; refusing."; fail=1; }
n=$(ls "$NET"/hs_${STEM}_mapsoftmax_f[0-4].npz 2>/dev/null | wc -l)
echo "    seed-42 live archives                   ${n}/5"
[ "$n" -eq 5 ] || { echo "      not all five are in place; sort that out before running."; fail=1; }
n=$(ls "$NET"/hs_${STEM}_f[0-4].npz 2>/dev/null | wc -l)
echo "    hs_${STEM}_f*.npz path (must be empty)  ${n}/5 present"
[ "$n" -eq 0 ] || { echo "      training writes here and would overwrite ${n} file(s); move them"
                    echo "      somewhere named first."; fail=1; }
[ "$fail" -eq 0 ] || { echo "  ABORT: preflight failed; nothing was touched."; exit 1; }
echo

run_fold () {                                   # seed fold gpu  -> trains, no --map
    local S=$1 F=$2 G=$3
    echo "    [train] s${S} f${F} gpu${G}  $(date +%H:%M:%S)"
    CUDA_VISIBLE_DEVICES="$G" python -m improve.head_swap \
        --head "$STEM" --fold "$F" --seed "$S" \
        > "logs/${STEM}_s${S}_f${F}_train.log" 2>&1 \
        || { echo "    [FAIL] s${S} f${F}; see logs/${STEM}_s${S}_f${F}_train.log"; return 1; }
    echo "    [ok]    s${S} f${F}  $(date +%H:%M:%S)"
}

# ---- A: train, four folds at a time; each fold is its own filename, so this is safe ----
echo "  A. training seed ${SEED}, five folds"
for f in 0 1 2 3; do run_fold "$SEED" "$f" "$f" & done
wait
run_fold "$SEED" 4 0 || exit 1
n=$(ls "$NET"/hs_${STEM}_f[0-4].npz 2>/dev/null | wc -l)
[ "$n" -eq 5 ] || { echo "  ABORT: training produced ${n}/5 archives."; exit 1; }
mv "$NET"/hs_${STEM}_f[0-4].npz "$S43"/            # the rectified reading of seed 43
echo "    5 archives (default reading) -> ${S43}/"
echo

# ---- B: hold the published archives, with a trap so they come back whatever happens ----
mkdir -p "$HOLD"
restore () {
    local rc=$?
    if ls "$HOLD"/hs_${STEM}_mapsoftmax_f[0-4].npz >/dev/null 2>&1; then
        mv "$HOLD"/hs_${STEM}_mapsoftmax_f[0-4].npz "$NET"/ 2>/dev/null
        echo "  [trap] published seed-42 archives restored to ${NET}/"
    fi
    rmdir "$HOLD" 2>/dev/null
    if ls "$CKPT"/head_${STEM}_f[0-4].pth >/dev/null 2>&1; then
        cp -p "$CKPT"/head_${STEM}_f[0-4].pth "$NET"/ 2>/dev/null
        echo "  [trap] seed-44 checkpoints restored to ${NET}/"
    fi
    exit $rc
}
trap restore EXIT INT TERM
mv "$NET"/hs_${STEM}_mapsoftmax_f[0-4].npz "$HOLD"/
echo "  B. published seed-42 archives held in ${HOLD}/ (trap will put them back)"
echo

# ---- C: measure the seed-43 checkpoints in the pooled reading ----
echo "  C. measuring seed ${SEED} in the pooled reading (no training; ~45s per fold)"
for f in 0 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=0 python -m improve.head_swap \
        --head "$STEM" --fold "$f" --measure-only --map softmax \
        > "logs/${STEM}_s${SEED}_f${f}_sm.log" 2>&1 \
        || { echo "    [FAIL] measure f${f}"; break; }
    mv "$NET"/hs_${STEM}_mapsoftmax_f${f}.npz "$S43"/ \
        || { echo "    [FAIL] collect f${f}"; break; }
    echo "    [ok] f${f}"
done
echo

# ---- the trap restores B and the checkpoints; report after that ----
trap - EXIT INT TERM
restore_rc=0
mv "$HOLD"/hs_${STEM}_mapsoftmax_f[0-4].npz "$NET"/ || restore_rc=1
rmdir "$HOLD" 2>/dev/null
cp -p "$CKPT"/head_${STEM}_f[0-4].pth "$NET"/ || restore_rc=1
echo "  D. seed-42 archives and seed-44 checkpoints restored (rc=${restore_rc})"
echo

python - <<'PY'
import glob
import numpy as np

G = 0.0120


def read(pat):
    P, I, W = [], [], []
    for p in sorted(glob.glob(pat)):
        with np.load(p, allow_pickle=True) as z:
            R = z["metrics"][:, 0, :]
            P.append(float(np.nanmean(R[:, 0])))
            I.append(float(np.nanmean(R[:, 2])))
            W.append(float(np.asarray(z["faith_head"], dtype=float).ravel()[5]))
    return P, I, W


SETS = [("seed 42", "outputs/headswap/hs_lowrank_mapsoftmax_f*.npz"),
        ("seed 43", "outputs/headswap_s43/hs_lowrank_mapsoftmax_f*.npz"),
        ("seed 44", "outputs/headswap_s44/hs_lowrank_mapsoftmax_f*.npz")]

print(f"  lowrank, pooled reading, by readout seed   (centered Gaussian within = {G:+.4f})")
print(f"  {'seed':<9}{'folds':>6}{'pointing':>11}{'IoU':>9}{'within':>11}{'sd':>9}"
      f"{'folds above':>13}")
means = {}
for tag, pat in SETS:
    P, I, W = read(pat)
    if not W:
        print(f"  {tag:<9}{'--':>6}   missing")
        continue
    above = sum(1 for w in W if w > G)
    means[tag] = (np.mean(P), np.mean(I), np.mean(W), np.std(W))
    print(f"  {tag:<9}{len(W):>6}{np.mean(P):>11.4f}{np.mean(I):>9.4f}"
          f"{np.mean(W):>+11.4f}{np.std(W):>9.4f}{above:>9}/{len(W)}")
    print(f"  {'':9}{'':6}  per-fold within: " + "  ".join(f"{w:+.4f}" for w in W))

if len(means) == 3:
    W = np.array([means[t][2] for t in ("seed 42", "seed 43", "seed 44")])
    P = np.array([means[t][0] for t in ("seed 42", "seed 43", "seed 44")])
    print()
    print(f"  across the three seeds: within {W.mean():+.4f} +/- {W.std(ddof=1):.4f}, "
          f"range {W.min():+.4f} to {W.max():+.4f}")
    print(f"                          pointing {P.mean():.4f} +/- {P.std(ddof=1):.4f}, "
          f"range {P.min():.4f} to {P.max():.4f}")
    if len(set(np.sign(W - G))) > 1:
        print(f"  The seeds fall on both sides of the learning-free reference, so no single")
        print(f"  number for this row is true of the mechanism. Report the seeds, not a mean:")
        print(f"  averaging {W.min():+.4f} and {W.max():+.4f} gives {W.mean():+.4f}, which")
        print(f"  describes neither and erases the result.")
        rho = np.corrcoef(P, W)[0, 1]
        print(f"  Over the three seeds, overlap and the within-lesion value move in opposite")
        print(f"  directions (pointing vs within, r = {rho:+.3f} over 3 points -- reported as a")
        print(f"  direction, not a coefficient, since three points cannot support one).")
    else:
        print(f"  All three seeds fall on the same side of the reference.")
PY
echo
echo "  ALL DONE  $(date)"
