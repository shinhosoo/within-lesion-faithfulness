#!/usr/bin/env bash
# Three readout seeds of the lowrank head, five folds each, across four GPUs.
#   nohup bash run_lowrank_seeds.sh > logs/sweep.log 2>&1 &
set -u
cd "$(dirname "$(readlink -f "$0")")" || exit 1

RANK_OPT="${RANK_OPT:-}"
STEM=lowrank
WANT_PARAMS=773
mkdir -p logs

# --map is deliberately NOT passed. Passing it tags the filename (hs_lowrank_mapsoftmax_f0.npz),
WANT_POINT=0.515
WANT_ENERGY=0.295

if [ -f env.sh ]; then
    # shellcheck disable=SC1091
    . ./env.sh
fi
for v in HAM_DATA_ROOT HAM_SPLIT_ROOT HAM_MASK_DIR; do
    if [ -z "${!v:-}" ]; then
        echo "ABORT: $v is not set. Create env.sh and set the three data paths first;"
        echo "       a run without them fails after the queue has already been launched."
        exit 1
    fi
    printf '  %-16s %s\n' "$v" "${!v}"
done
echo "  RANK_OPT         '${RANK_OPT}'   (empty means head_swap's default)"
echo

echo "  parameter counts, no GPU:"
python -m improve.head_swap --head "$STEM" --count ${RANK_OPT} 2>&1 | sed 's/^/    /'
echo

# ---- one job: run it, then immediately collect its output ----
one () {
    local S=$1 F=$2 G=$3
    local D="outputs/headswap_s${S}"
    local f="outputs/headswap/hs_${STEM}_f${F}.npz"
    mkdir -p "$D"
    if [ -e "${D}/hs_${STEM}_f${F}.npz" ]; then
        echo "  [skip ] s${S} f${F}: already collected"
        return 0
    fi
    if [ -e "$f" ]; then
        echo "  [REFUSE] s${S} f${F}: ${f} exists and was not collected by an earlier job."
        echo "           Move it out of the way before re-running; overwriting it is how the"
        echo "           published archives were lost."
        return 1
    fi
    echo "  [start] s${S} f${F} gpu${G}  $(date +%H:%M:%S)"
    CUDA_VISIBLE_DEVICES="$G" python -m improve.head_swap \
        --head "$STEM" --fold "$F" --seed "$S" ${RANK_OPT} \
        > "logs/${STEM}_s${S}_f${F}.log" 2>&1
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo "  [FAIL ] s${S} f${F} rc=${rc}; see logs/${STEM}_s${S}_f${F}.log"
        return $rc
    fi
    if [ ! -e "$f" ]; then
        echo "  [FAIL ] s${S} f${F}: the run succeeded but wrote no archive"
        return 1
    fi
    mv "$f" "${D}/" || return 1
    echo "  [done ] s${S} f${F} -> ${D}/  $(date +%H:%M:%S)"
}

# ---- the first job alone, so the configuration is checked before 14 more are queued ----
echo "  checking the configuration on one fold before queueing the rest"
one 42 0 0 || { echo "ABORT: the first job failed."; exit 1; }

python - "outputs/headswap_s42/hs_${STEM}_f0.npz" "$WANT_PARAMS" "$WANT_POINT" "$WANT_ENERGY" <<'PY'
import sys
import numpy as np
path = sys.argv[1]
want_n, want_p, want_e = int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
with np.load(path, allow_pickle=True) as z:
    n = int(z["params_mechanism"])
    R = z["metrics"][:, 0, :]
    point, energy = float(np.nanmean(R[:, 0])), float(np.nanmean(R[:, 1]))
    v = np.asarray(z["faith_head"], dtype=float).ravel()
    sha = z["source_sha1"].item()
print(f"    params_mechanism {n}            (want {want_n})")
print(f"    pointing         {point:.4f}       (seed 42 pooled is {want_p:.3f})")
print(f"    energy           {energy:.4f}       (pooled {want_e:.3f}; rectified is near 0.524)")
print(f"    Ins-Del {v[2]:+.4f}   within {v[5]:+.4f}   sha {sha}")
bad = []
if n != want_n:
    bad.append(f"params_mechanism is {n}, not {want_n}: not the mechanism the row reports")
if abs(energy - want_e) > 0.05:
    bad.append(f"energy {energy:.4f} is not the pooled reading ({want_e:.3f}); the default "
               f"reading is not the one the row is measured in")
# This is fold 0 of the seed whose five-fold values are already known, so a pointing far from
# the known one means the default reading or the configuration differs, not that the seed moved.
if abs(point - want_p) > 0.08:
    bad.append(f"pointing {point:.4f} is far from seed 42's {want_p:.3f}; this is not "
               f"reproducing the row the sweep is measured against")
if bad:
    for b in bad:
        print(f"    PROBLEM: {b}")
    sys.exit(3)
print("    the default reading reproduces the published row; queueing the remaining 14 runs")
PY
rc=$?
if [ $rc -ne 0 ]; then
    echo
    echo "ABORT: the first fold does not match the row this sweep is meant to replace."
    echo "       Nothing further was launched, and outputs/headswap_s42/ holds only that one"
    echo "       fold. If params_mechanism was wrong, re-run as:"
    echo "           rm -r outputs/headswap_s42"
    echo "           RANK_OPT='--rank 1' nohup bash run_lowrank_seeds.sh > logs/sweep.log 2>&1 &"
    exit $rc
fi
echo

# ---- the rest, four at a time, never two jobs on the same fold ----
wave () {
    local spec S F G
    for spec in "$@"; do
        S=${spec%%:*}; G=${spec##*:}; F=${spec#*:}; F=${F%:*}
        one "$S" "$F" "$G" &
    done
    wait
}

echo "  wave 1/4"; wave 42:1:0 42:2:1 42:3:2 42:4:3
echo "  wave 2/4"; wave 43:0:0 43:1:1 43:2:2 43:3:3
echo "  wave 3/4"; wave 43:4:0 44:0:1 44:1:2 44:2:3
echo "  wave 4/4"; wave 44:3:0 44:4:1

echo
echo "  collected:"
for S in 42 43 44; do
    printf '    seed %s: %s file(s)\n' "$S" \
        "$(ls outputs/headswap_s${S}/hs_${STEM}_f*.npz 2>/dev/null | wc -l)"
done

python - <<'PY'
import glob
import numpy as np
print()
print(f"  {'seed':<6}{'folds':>6}{'pointing':>11}{'energy':>10}{'IoU':>9}"
      f"{'Ins-Del':>10}{'within':>10}")
rows = {}
for s in (42, 43, 44):
    P = E = I = None
    ins, wit, pts, ens, ious = [], [], [], [], []
    for p in sorted(glob.glob(f"outputs/headswap_s{s}/hs_lowrank_f*.npz")):
        with np.load(p, allow_pickle=True) as z:
            R = z["metrics"][:, 0, :]
            pts.append(float(np.nanmean(R[:, 0])))
            ens.append(float(np.nanmean(R[:, 1])))
            ious.append(float(np.nanmean(R[:, 2])))
            v = np.asarray(z["faith_head"], dtype=float).ravel()
            ins.append(float(v[2]))
            wit.append(float(v[5]))
    if not wit:
        continue
    rows[s] = (np.mean(pts), np.mean(ens), np.mean(ious), np.mean(ins), np.mean(wit))
    print(f"  {s:<6}{len(wit):>6}{rows[s][0]:>11.4f}{rows[s][1]:>10.4f}{rows[s][2]:>9.4f}"
          f"{rows[s][3]:>+10.4f}{rows[s][4]:>+10.4f}")
if len(rows) > 1:
    A = np.array(list(rows.values()))
    print()
    print(f"  {'mean':<6}{'':>6}" + "".join(
        f"{v:>{w}.4f}" if i < 3 else f"{v:>+{w}.4f}"
        for i, (v, w) in enumerate(zip(A.mean(0), (11, 10, 9, 10, 10)))))
    print(f"  {'sd':<6}{'':>6}" + "".join(
        f"{v:>{w}.4f}" for v, w in zip(A.std(0), (11, 10, 9, 10, 10))))
    print()
    print(f"  within over seeds: {A[:, 4].mean():+.4f} +/- {A[:, 4].std():.4f}")
    print(f"  the centered Gaussian is +0.0120; a trained map at or below it leaves the")
    print(f"  manuscript's claim standing, and above it does not.")
    print(f"  Table S10 prints -0.0840 for this row, from a single run that cannot be")
    print(f"  reproduced; the mean above is what replaces it.")
PY
echo
echo "  ALL DONE  $(date)"
