#!/usr/bin/env bash
# A second readout seed for the remaining stage-2 mechanisms, to bound what the lowrank result
#   nohup bash run_other_seeds.sh > logs/other_seeds.log 2>&1 &
set -u

SEED=44
TRAIN_STEMS=(cam mlp cond)        # checkpoints are seed 42; need training
MEASURE_ONLY=(ssf)                # checkpoints are already seed 44
NET=outputs/headswap
OUTD=outputs/headswap_s44
B42=outputs/headswap/ckpt_s42
HOLD=outputs/headswap_hold_s42
mkdir -p logs "$OUTD" "$B42" "$HOLD"

[ -f env.sh ] && . ./env.sh
for v in HAM_DATA_ROOT HAM_SPLIT_ROOT HAM_MASK_DIR; do
    [ -z "${!v:-}" ] && { echo "ABORT: $v is not set."; exit 1; }
done

if pgrep -f 'improve.head_swap' >/dev/null 2>&1; then
    echo "ABORT: a head_swap run is still going. This script writes the same filenames and"
    echo "       would race it. Wait for it to finish."
    pgrep -af 'improve.head_swap' | sed 's/^/       /'
    exit 1
fi

# ---------- back up everything this script can overwrite, and verify ----------
echo "  backing up what this run would overwrite"
fail=0
for s in "${TRAIN_STEMS[@]}"; do
    n=$(ls "$NET"/head_${s}_f[0-4].pth 2>/dev/null | wc -l)
    if [ "$n" -ne 5 ]; then
        echo "    ${s}: ${n}/5 checkpoints present -- refusing"; fail=1; continue
    fi
    cp -p "$NET"/head_${s}_f[0-4].pth "$B42"/ || fail=1
    m=$(ls "$B42"/head_${s}_f[0-4].pth 2>/dev/null | wc -l)
    echo "    ${s}: checkpoints ${m}/5 copied to ${B42}/"
    [ "$m" -eq 5 ] || fail=1
done
[ "$fail" -eq 0 ] || { echo "  ABORT: backup incomplete; nothing was changed."; exit 1; }
echo

# Hold aside every archive a run below would land on, so the published values cannot be lost.
hold_for () {                    # stem -> moves its hs_<stem>_f*.npz and _mapsoftmax_f*.npz away
    local s=$1 moved=0
    for pat in "hs_${s}_f" "hs_${s}_mapsoftmax_f"; do
        for f in 0 1 2 3 4; do
            if [ -e "${NET}/${pat}${f}.npz" ]; then
                mv "${NET}/${pat}${f}.npz" "$HOLD"/ && moved=$((moved + 1))
            fi
        done
    done
    echo "    ${s}: ${moved} archive(s) held in ${HOLD}/"
}

restore_all () {
    local rc=$?
    local n=0
    if ls "$HOLD"/*.npz >/dev/null 2>&1; then
        n=$(ls "$HOLD"/*.npz | wc -l)
        mv "$HOLD"/*.npz "$NET"/ 2>/dev/null
        echo "  [trap] ${n} published archive(s) restored to ${NET}/"
    fi
    rmdir "$HOLD" 2>/dev/null
    for s in "${TRAIN_STEMS[@]}"; do
        if ls "$B42"/head_${s}_f[0-4].pth >/dev/null 2>&1; then
            cp -p "$B42"/head_${s}_f[0-4].pth "$NET"/ 2>/dev/null
        fi
    done
    echo "  [trap] seed-42 checkpoints restored from ${B42}/"
    exit $rc
}
trap restore_all EXIT INT TERM

echo "  holding published archives aside"
for s in "${TRAIN_STEMS[@]}" "${MEASURE_ONLY[@]}"; do hold_for "$s"; done
echo

collect () {                     # stem fold suffix -> move the result into OUTD
    local s=$1 f=$2 sfx=$3
    local src="${NET}/hs_${s}${sfx}_f${f}.npz"
    [ -e "$src" ] || { echo "    [FAIL] ${s}${sfx} f${f}: no archive written"; return 1; }
    mv "$src" "${OUTD}/hs_${s}${sfx}_f${f}.npz" || return 1
}

# ---------- A: train seed 44 for the three mechanisms, four folds at a time ----------
echo "  A. training seed ${SEED}: ${TRAIN_STEMS[*]}"
JOBS=()
for s in "${TRAIN_STEMS[@]}"; do for f in 0 1 2 3 4; do JOBS+=("${s}:${f}"); done; done
i=0
while [ $i -lt ${#JOBS[@]} ]; do
    pids=()
    for g in 0 1 2 3; do
        [ $i -lt ${#JOBS[@]} ] || break
        s=${JOBS[$i]%%:*}; f=${JOBS[$i]##*:}; i=$((i + 1))
        echo "    [train] ${s} f${f} gpu${g}  $(date +%H:%M:%S)"
        (
          CUDA_VISIBLE_DEVICES="$g" python -m improve.head_swap \
              --head "$s" --fold "$f" --seed "$SEED" \
              > "logs/${s}_s${SEED}_f${f}_train.log" 2>&1 \
            && echo "    [ok]    ${s} f${f}  $(date +%H:%M:%S)" \
            || echo "    [FAIL]  ${s} f${f}; see logs/${s}_s${SEED}_f${f}_train.log"
        ) &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p"; done
done
echo "    collecting the default-reading archives"
for s in "${TRAIN_STEMS[@]}"; do for f in 0 1 2 3 4; do collect "$s" "$f" ""; done; done
echo

# ---------- B: measure every stem in the pooled reading, four GPUs at a time ----------
echo "  B. measuring in the pooled reading (no training, ~45s per fold, four at a time)"
JOBS=()
for s in "${TRAIN_STEMS[@]}" "${MEASURE_ONLY[@]}"; do
    for f in 0 1 2 3 4; do JOBS+=("${s}:${f}"); done
done
i=0
while [ $i -lt ${#JOBS[@]} ]; do
    pids=()
    for g in 0 1 2 3; do
        [ $i -lt ${#JOBS[@]} ] || break
        s=${JOBS[$i]%%:*}; f=${JOBS[$i]##*:}; i=$((i + 1))
        (
          CUDA_VISIBLE_DEVICES="$g" python -m improve.head_swap \
              --head "$s" --fold "$f" --measure-only --map softmax \
              > "logs/${s}_s${SEED}_f${f}_sm.log" 2>&1 \
            && { mv "${NET}/hs_${s}_mapsoftmax_f${f}.npz" \
                    "${OUTD}/hs_${s}_mapsoftmax_f${f}.npz" \
                 && echo "    [ok]   ${s} f${f} gpu${g}" \
                 || echo "    [FAIL] ${s} f${f}: no archive to collect"; } \
            || echo "    [FAIL] ${s} f${f} measure; see logs/${s}_s${SEED}_f${f}_sm.log"
        ) &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p"; done
done
echo

# ---------- restore, then report ----------
trap - EXIT INT TERM
rc=0
if ls "$HOLD"/*.npz >/dev/null 2>&1; then mv "$HOLD"/*.npz "$NET"/ || rc=1; fi
rmdir "$HOLD" 2>/dev/null
for s in "${TRAIN_STEMS[@]}"; do cp -p "$B42"/head_${s}_f[0-4].pth "$NET"/ || rc=1; done
echo "  C. published archives and seed-42 checkpoints restored (rc=${rc})"
echo

python - <<'PY'
import glob
import numpy as np

G = 0.0120
# The seed-42 value Table S10 prints, for the comparison this script exists to make.
PUB = {"ssf": (0.276, 0.200, -0.0680), "lowrank": (0.515, 0.308, -0.0840),
       "cam": (0.277, 0.245, -0.0400), "mlp": (0.407, 0.268, -0.1180),
       "cond": (0.265, 0.174, -0.0260)}


def read(pat):
    P, I, W = [], [], []
    for p in sorted(glob.glob(pat)):
        with np.load(p, allow_pickle=True) as z:
            R = z["metrics"][:, 0, :]
            P.append(float(np.nanmean(R[:, 0])))
            I.append(float(np.nanmean(R[:, 2])))
            W.append(float(np.asarray(z["faith_head"], dtype=float).ravel()[5]))
    return P, I, W


print(f"  stage-2 mechanisms, pooled reading, seed 42 (published) vs seed 44")
print(f"  centered Gaussian within = {G:+.4f}\n")
print(f"  {'mechanism':<10}{'seed':>6}{'pointing':>10}{'IoU':>8}{'within':>10}{'sd':>8}"
      f"{'above':>8}")
crossing, stable = [], []
for stem in ("ssf", "lowrank", "cam", "mlp", "cond"):
    p42, i42, w42 = PUB[stem]
    print(f"  {stem:<10}{'42':>6}{p42:>10.3f}{i42:>8.3f}{w42:>+10.4f}{'--':>8}"
          f"{'0/5':>8}   (Table S10)")
    for tag, pat in ((f"44", f"outputs/headswap_s44/hs_{stem}_mapsoftmax_f*.npz"),):
        P, I, W = read(pat)
        if not W:
            print(f"  {'':10}{tag:>6}{'--':>10}{'--':>8}{'--':>10}{'--':>8}{'--':>8}   missing")
            continue
        above = sum(1 for w in W if w > G)
        print(f"  {'':10}{tag:>6}{np.mean(P):>10.3f}{np.mean(I):>8.3f}"
              f"{np.mean(W):>+10.4f}{np.std(W):>8.4f}{above:>6}/{len(W)}")
        print(f"  {'':16}  per-fold within: " + "  ".join(f"{w:+.4f}" for w in W))
        d_over = np.mean(P) - p42
        d_with = np.mean(W) - w42
        (crossing if above > 0 else stable).append(
            (stem, d_over, d_with, above, len(W)))
print()
if crossing:
    print("  crossed the learning-free reference at seed 44:")
    for s, do, dw, a, n in crossing:
        arrow = "opposite directions" if do * dw < 0 else "the same direction"
        print(f"    {s:<9} overlap {do:+.3f}, within {dw:+.4f} -> {arrow}, {a}/{n} folds above")
if stable:
    print("  stayed below it:")
    for s, do, dw, a, n in stable:
        print(f"    {s:<9} overlap {do:+.3f}, within {dw:+.4f}, 0/{n} folds above")
print()
print("  Read this against s3_mlp, which has three seeds, a pointing span of 0.082 and a")
print("  within span of 0.005, and 0 of 15 folds above the reference. If the mechanisms that")
print("  cross are the ones whose overlap falls, the two quantities are dissociated by a")
print("  controlled intervention rather than by an absent correlation.")
PY
echo
echo "  ALL DONE  $(date)"
