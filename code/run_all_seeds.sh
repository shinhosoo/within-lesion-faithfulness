#!/usr/bin/env bash
# A third readout seed for cam, mlp and ssf, and all three seeds for cond.
#   nohup bash run_all_seeds.sh > logs/all_seeds.log 2>&1 &
set -u

HS_STEMS=(cam mlp ssf)          # head_swap.py builds these; one new seed each
HS_SEED=43
COND_SEEDS=(42 43 44)           # cond_head.py; seed 42 also checks the printed row
NET=outputs/headswap
CK=outputs/headswap/ckpt_orig
HOLD=outputs/headswap_hold_all
GPUS=(0 1 2 3)
mkdir -p logs "$CK" "$HOLD"

[ -f env.sh ] && . ./env.sh
for v in HAM_DATA_ROOT HAM_SPLIT_ROOT HAM_MASK_DIR; do
    [ -z "${!v:-}" ] && { echo "ABORT: $v is not set."; exit 1; }
done
if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
   | grep -q . && pgrep -f 'improve\.(head_swap|cond_head)' >/dev/null 2>&1; then
    echo "ABORT: a head_swap or cond_head run is still going; it writes the same filenames."
    pgrep -af 'improve\.(head_swap|cond_head)' | sed 's/^/  /'
    exit 1
fi

# ---------------------------------------------------------------- preflight
echo "  preflight: backing up every file a run below would overwrite"
fail=0
for s in "${HS_STEMS[@]}" cond; do
    n=$(ls "$NET"/head_${s}_f[0-4].pth 2>/dev/null | wc -l)
    if [ "$n" -ne 5 ]; then
        echo "    ${s}: ${n}/5 checkpoints -- refusing"; fail=1; continue
    fi
    cp -p "$NET"/head_${s}_f[0-4].pth "$CK"/ || fail=1
    m=$(ls "$CK"/head_${s}_f[0-4].pth 2>/dev/null | wc -l)
    echo "    ${s}: ${m}/5 checkpoints -> ${CK}/"
    [ "$m" -eq 5 ] || fail=1
done
[ "$fail" -eq 0 ] || { echo "  ABORT: backup incomplete; nothing changed."; exit 1; }

moved=0
for pat in hs_cam_f hs_cam_mapsoftmax_f hs_mlp_f hs_mlp_mapsoftmax_f \
           hs_ssf_f hs_ssf_mapsoftmax_f hs_cond_f; do
    for f in 0 1 2 3 4; do
        [ -e "${NET}/${pat}${f}.npz" ] && { mv "${NET}/${pat}${f}.npz" "$HOLD"/; \
                                            moved=$((moved + 1)); }
    done
done
echo "    ${moved} archive(s) held in ${HOLD}/"

restore () {
    local rc=$?
    local n=0
    if ls "$HOLD"/*.npz >/dev/null 2>&1; then
        n=$(ls "$HOLD"/*.npz | wc -l)
        mv "$HOLD"/*.npz "$NET"/ 2>/dev/null
    fi
    rmdir "$HOLD" 2>/dev/null
    cp -p "$CK"/head_*_f[0-4].pth "$NET"/ 2>/dev/null
    echo "  [restore] ${n} archive(s) and the original checkpoints are back in ${NET}/"
    exit $rc
}
trap restore EXIT INT TERM
echo

# ---------------------------------------------------------------- the wave runner
run_waves () {
    local -n JOBS=$1
    local label=$2
    local i=0 w=0
    local total=${#JOBS[@]}
    while [ $i -lt $total ]; do
        w=$((w + 1))
        local pids=()
        for g in "${GPUS[@]}"; do
            [ $i -lt $total ] || break
            IFS='|' read -r cmd lg src dst <<< "${JOBS[$i]}"
            i=$((i + 1))
            (
              CUDA_VISIBLE_DEVICES="$g" eval "$cmd" > "$lg" 2>&1 \
                && { [ -e "$src" ] \
                     && mv "$src" "$dst" \
                     && echo "    [ok]   ${label} $(basename "$dst") gpu${g}" \
                     || echo "    [FAIL] ${label} $(basename "$dst"): nothing written"; } \
                || echo "    [FAIL] ${label} $(basename "$dst"): see ${lg}"
            ) &
            pids+=($!)
        done
        echo "    -- ${label} wave ${w}, ${#pids[@]} job(s), $(date +%H:%M:%S)"
        for p in "${pids[@]}"; do wait "$p"; done
    done
}

# ---------------------------------------------------------------- A: cam/mlp/ssf, seed 43
OUT43=outputs/headswap_s43
mkdir -p "$OUT43"
A=()
for s in "${HS_STEMS[@]}"; do for f in 0 1 2 3 4; do
    A+=("python -m improve.head_swap --head $s --fold $f --seed $HS_SEED|logs/${s}_s${HS_SEED}_f${f}_train.log|${NET}/hs_${s}_f${f}.npz|${OUT43}/hs_${s}_f${f}.npz")
done; done
echo "  A. training seed ${HS_SEED}: ${HS_STEMS[*]}  (${#A[@]} runs)"
run_waves A "A"
echo

# ---------------------------------------------------------------- B: measure them, pooled
B=()
for s in "${HS_STEMS[@]}"; do for f in 0 1 2 3 4; do
    B+=("python -m improve.head_swap --head $s --fold $f --measure-only --map softmax|logs/${s}_s${HS_SEED}_f${f}_sm.log|${NET}/hs_${s}_mapsoftmax_f${f}.npz|${OUT43}/hs_${s}_mapsoftmax_f${f}.npz")
done; done
echo "  B. measuring seed ${HS_SEED} in the pooled reading  (${#B[@]} runs, no training)"
run_waves B "B"
echo

# ---------------------------------------------------------------- C: cond, one seed at a time
for sd in "${COND_SEEDS[@]}"; do
    OUTC="outputs/headswap_cond_s${sd}"
    mkdir -p "$OUTC"
    C=()
    for f in 0 1 2 3 4; do
        C+=("python -m improve.cond_head --fold $f --seed $sd --stage 2 --map softmax|logs/cond_s${sd}_f${f}.log|${NET}/hs_cond_f${f}.npz|${OUTC}/hs_cond_f${f}.npz")
    done
    echo "  C. cond, seed ${sd}  (${#C[@]} runs)"
    run_waves C "cond-s${sd}"
    echo
done

# ---------------------------------------------------------------- restore, then report
trap - EXIT INT TERM
rc=0
if ls "$HOLD"/*.npz >/dev/null 2>&1; then mv "$HOLD"/*.npz "$NET"/ || rc=1; fi
rmdir "$HOLD" 2>/dev/null
cp -p "$CK"/head_*_f[0-4].pth "$NET"/ || rc=1
echo "  D. held archives and original checkpoints restored (rc=${rc})"
echo

python - <<'PY'
import glob
import os
import numpy as np

G = 0.0120
# What Table S10 prints for seed 42, so each new seed is read against the published row.
PUB = {"ssf": (0.276, 0.200, -0.0680), "lowrank": (0.515, 0.308, -0.0840),
       "cam": (0.277, 0.245, -0.0400), "mlp": (0.407, 0.268, -0.1180),
       "cond": (0.265, 0.174, -0.0260)}
WHERE = {
    ("lowrank", 42): "outputs/headswap/hs_lowrank_mapsoftmax_f*.npz",
    ("lowrank", 43): "outputs/headswap_s43/hs_lowrank_mapsoftmax_f*.npz",
    ("lowrank", 44): "outputs/headswap_s44/hs_lowrank_mapsoftmax_f*.npz",
}
for s in ("cam", "mlp", "ssf"):
    WHERE[(s, 42)] = f"outputs/headswap/hs_{s}_mapsoftmax_f*.npz"
    WHERE[(s, 43)] = f"outputs/headswap_s43/hs_{s}_mapsoftmax_f*.npz"
    WHERE[(s, 44)] = f"outputs/headswap_s44/hs_{s}_mapsoftmax_f*.npz"
for sd in (42, 43, 44):
    WHERE[("cond", sd)] = f"outputs/headswap_cond_s{sd}/hs_cond_f*.npz"


def read(pat):
    P, I, W = [], [], []
    for p in sorted(glob.glob(pat)):
        with np.load(p, allow_pickle=True) as z:
            R = z["metrics"][:, 0, :]
            P.append(float(np.nanmean(R[:, 0])))
            I.append(float(np.nanmean(R[:, 2])))
            W.append(float(np.asarray(z["faith_head"], dtype=float).ravel()[5]))
    return P, I, W


def sd1(v):
    if len(v) < 2:
        return float("nan")
    m = sum(v) / len(v)
    return (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5


print(f"  within-lesion value by readout seed, pooled reading")
print(f"  centered Gaussian = {G:+.4f}; a value above it is a trained map beating a map that "
      f"knows nothing\n")
print(f"  {'mechanism':<9}{'seed':>5}{'folds':>6}{'point':>9}{'IoU':>8}{'within':>10}"
      f"{'sd':>8}{'above':>8}")
tally = {}
for stem in ("ssf", "lowrank", "cam", "mlp", "cond"):
    rows = []
    for sd in (42, 43, 44):
        P, I, W = read(WHERE[(stem, sd)])
        if not W:
            continue
        rows.append((sd, P, I, W))
    if not rows:
        print(f"  {stem:<9}  nothing found")
        continue
    for sd, P, I, W in rows:
        ab = sum(1 for w in W if w > G)
        print(f"  {stem:<9}{sd:>5}{len(W):>6}{np.mean(P):>9.3f}{np.mean(I):>8.3f}"
              f"{np.mean(W):>+10.4f}{sd1(W):>8.4f}{ab:>6}/{len(W)}")
    allW = [w for _, _, _, W in rows for w in W]
    tally[stem] = (len(rows), sum(1 for w in allW if w > G), len(allW),
                   min(np.mean(W) for _, _, _, W in rows),
                   max(np.mean(W) for _, _, _, W in rows))
    print(f"  {'':14}  seeds {', '.join(str(s) for s, _, _, _ in rows)};  "
          f"{tally[stem][1]} of {tally[stem][2]} folds above the reference")

print()
print(f"  {'mechanism':<9}{'seeds':>7}{'folds above':>13}{'within range':>26}")
for stem, (ns, ab, n, lo, hi) in tally.items():
    print(f"  {stem:<9}{ns:>7}{ab:>7}/{n:<5}   {lo:>+8.4f} to {hi:>+8.4f}")
crossing = [s for s, t in tally.items() if t[1] > 0]
mx = max((t[4] for t in tally.values()), default=float("nan"))
print()
if crossing:
    print(f"  cross the reference at some seed: {', '.join(crossing)}")
    print(f"  highest trained-map within-lesion value over every seed measured: {mx:+.4f}")
    for nm, v in (("Integrated Gradients", 0.167), ("Grad-CAM (stage-3)", 0.137),
                  ("Grad-CAM (stage-4)", 0.096), ("Grad-CAM (stage-2)", 0.073)):
        print(f"    {nm:<22}{v:+.3f}   {v - mx:+.4f} above it")
    print()
    print("  So the sentence 'every trained map on that backbone at or below it' is a property")
    print("  of seed 42, not of the mechanisms, and has to go. What survives is the gap to the")
    print("  gradient maps, which is wide at every seed measured, and the dissociation itself:")
    print("  the seed moves the within-lesion value by up to a tenth while leaving overlap")
    print("  almost where it was, and mask supervision does the reverse.")
else:
    print("  no mechanism crosses the reference at any seed measured.")
PY
echo
echo "  ALL DONE  $(date)"
