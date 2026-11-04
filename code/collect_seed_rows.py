#!/usr/bin/env python3
"""Collect the per-fold pooled values of every mechanism at every readout seed.

    python collect_seed_rows.py # inventory and the JSON
    python collect_seed_rows.py --out seed_folds.json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

STEMS = ("lowrank", "ssf", "mlp", "cam", "cond", "s3_mlp")
SEEDS = (42, 43, 44)
COLS = ("point", "energy", "iou", "insdel", "within")

# Candidate directories per (stem, seed), in the order they are tried. Several are listed where
# a sweep wrote the same seed to more than one place over the course of the session.
WHERE = {}
for s in STEMS:
    WHERE[(s, 42)] = ["outputs/headswap"]
    WHERE[(s, 43)] = ["outputs/headswap_s43", "outputs/headswap/seed43"]
    WHERE[(s, 44)] = ["outputs/headswap_s44", "outputs/headswap/seed44"]
# ssf seed 42 was retrained after its checkpoint was overwritten; run_ssf_s42.sh put the pooled
# archives here and left the flat directory holding the seed-44 run.
WHERE[("ssf", 42)] = ["outputs/headswap_s42_pooled", "outputs/headswap"]
WHERE[("cam", 42)] = ["outputs/headswap_s42_pooled", "outputs/headswap"]
# cond_head.py calls seed_out(), so its non-42 seeds were never in the flat directory.
WHERE[("cond", 43)] = ["outputs/headswap/seed43", "outputs/headswap_s43"]
WHERE[("cond", 44)] = ["outputs/headswap/seed44", "outputs/headswap_s44"]

# The rows Table S10 prints today, for the seed-42 self-check, transcribed from the built
PRINTED_DEFAULT = {
    "lowrank": [("Linear score (rank 1)", 773,
                 [0.515, 0.295, 0.308, +0.019, -0.084])],
    "ssf":     [("Scale and shift, per channel", 384,
                 [0.287, 0.286, 0.173, -0.031, -0.072])],
    "mlp":     [("Position-wise MLP", 6593,
                 [0.407, 0.285, 0.268, -0.024, -0.118])],
    "cam":     [("Class activation map", 1735,
                 [0.278, 0.278, 0.222, +0.019, -0.002])],
    "cond":    [("Image-conditional (stage 2)", 80480,
                 [0.265, 0.273, 0.174, -0.034, -0.026]),
                ("Conditional at the third stage", 87200,
                 [0.612, 0.591, 0.343, +0.059, -0.013])],
    "s3_mlp":  [("MLP at the third stage", 13121,
                 [0.667, 0.567, 0.474, +0.134, -0.011])],
}


def sd1(v):
    m = sum(v) / len(v)
    return (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5 if len(v) > 1 else 0.0


def read_one(path):
    """The five column values of one fold, plus the provenance keys."""
    with np.load(path, allow_pickle=True) as z:
        R = z["metrics"][:, 0, :]
        v = np.asarray(z["faith_head"], dtype=float).ravel()
        if v.size != 6:
            return None, f"faith_head has {v.size} elements, expected 6"
        if not (abs((v[1] - v[0]) - v[2]) < 1e-6 and abs((v[4] - v[3]) - v[5]) < 1e-6):
            return None, "faith_head does not satisfy its own arithmetic; layout unknown"
        out = dict(zip(COLS, (float(np.nanmean(R[:, 0])), float(np.nanmean(R[:, 1])),
                              float(np.nanmean(R[:, 2])), float(v[2]), float(v[5]))))
        out["sha"] = str(z["source_sha1"]) if "source_sha1" in z.files else "?"
        for k in ("mech_params", "head_params", "n_params"):
            if k in z.files:
                out["mech"] = int(np.asarray(z[k]).ravel()[0])
                break
        else:
            out["mech"] = None
        out["n"] = int(R.shape[0])
        return out, None


def locate(stem, seed):
    """Every directory that holds pooled archives for this (stem, seed), with the fold count."""
    found = []
    for d in WHERE[(stem, seed)]:
        n = len(glob.glob(os.path.join(d, f"hs_{stem}_mapsoftmax_f[0-4].npz")))
        if n:
            found.append((d, n))
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="write the JSON here")
    ap.add_argument("--printed", default=None,
                    help="JSON file of {stem: [point, energy, iou, insdel, within]} as Table S10 "
                         "prints them, for the seed-42 self-check")
    a = ap.parse_args()

    printed = PRINTED_DEFAULT
    if a.printed:
        raw = json.load(open(a.printed))
        printed = {}
        for stem, v in raw.items():
            # accept either {stem: [p, e, i, d, w]} or {stem: [[label, params, [..]], ...]}
            if v and not isinstance(v[0], (list, tuple)):
                printed[stem] = [("--printed", None, [float(x) for x in v])]
            else:
                printed[stem] = [(str(r[0]), (int(r[1]) if r[1] is not None else None),
                                  [float(x) for x in r[2]]) for r in v]

    print("  inventory, pooled reading only")
    print(f"    {'stem':<9}{'seed':<6}{'folds':<7}{'where':<34}{'sha':<12}{'params':>8}")
    D, problems = {}, []
    for stem in STEMS:
        for seed in SEEDS:
            hits = locate(stem, seed)
            if not hits:
                print(f"    {stem:<9}{seed:<6}{'-':<7}{'not found':<34}")
                # cond is written by cond_head.py rather than head_swap.py and need not use the
                near = []
                for d in WHERE[(stem, seed)]:
                    near += [os.path.basename(p)
                             for p in sorted(glob.glob(os.path.join(d, f"*{stem}*.npz")))]
                if near:
                    print(f"    {'':<15}{'':<7}present instead: "
                          + ", ".join(sorted(set(near))[:6])
                          + (" ..." if len(set(near)) > 6 else ""))
                problems.append(f"{stem} seed {seed}: no hs_{stem}_mapsoftmax_f*.npz in "
                                f"{', '.join(WHERE[(stem, seed)])}"
                                + (f"; files naming this stem there: "
                                   f"{', '.join(sorted(set(near))[:6])}" if near else ""))
                continue
            if len(hits) > 1:
                where = "; ".join(f"{d} ({n})" for d, n in hits)
                print(f"    {stem:<9}{seed:<6}{'?':<7}{where:<34}")
                problems.append(f"{stem} seed {seed}: archives in {len(hits)} places ({where}); "
                                f"which is this seed cannot be told from the files, so it is "
                                f"skipped rather than guessed")
                continue
            d, n = hits[0]
            rows, err = [], None
            for f in range(5):
                p = os.path.join(d, f"hs_{stem}_mapsoftmax_f{f}.npz")
                if not os.path.exists(p):
                    continue
                r, e = read_one(p)
                if e:
                    err = f"{os.path.basename(p)}: {e}"
                    break
                rows.append(r)
            if err:
                print(f"    {stem:<9}{seed:<6}{'!':<7}{err:<34}")
                problems.append(f"{stem} seed {seed}: {err}")
                continue
            shas = sorted({r["sha"] for r in rows})
            mechs = sorted({r["mech"] for r in rows if r["mech"] is not None})
            print(f"    {stem:<9}{seed:<6}{len(rows):<7}{d:<34}"
                  f"{','.join(shas)[:11]:<12}"
                  f"{(mechs[0] if len(mechs) == 1 else '?'):>8}")
            if len(rows) != 5:
                problems.append(f"{stem} seed {seed}: {len(rows)} folds, not 5")
                continue
            if len(shas) != 1:
                problems.append(f"{stem} seed {seed}: folds span sources {shas}")
            D[f"{stem}_{seed}"] = {
                "sha": shas[0] if len(shas) == 1 else "/".join(shas),
                "mech": mechs[0] if len(mechs) == 1 else None,
                "folds": 5,
                "where": d,
                "images": [r["n"] for r in rows],
                **{c: [round(r[c], 6) for r in rows] for c in COLS},
            }

    # ---- the seed-42 rows against the table ----
    print("\n  seed-42 rows against what Table S10 prints")
    bad = []
    for stem in STEMS:
        k = f"{stem}_42"
        if k not in D:
            print(f"    {stem:<9} missing, cannot check")
            continue
        cands = printed.get(stem)
        if not cands:
            print(f"    {stem:<9} no printed row given")
            continue
        got = [sum(D[k][c]) / 5 for c in COLS]
        shown = "  ".join(f"{g:+.3f}" for g in got)
        best, bestd = None, None
        for label, params, want in cands:
            d = max(abs(round(g, 3) - w) for g, w in zip(got, want))
            if bestd is None or d < bestd:
                best, bestd = (label, params, want), d
        label, params, want = best
        ok = bestd < 5e-4
        pok = (D[k]["mech"] is None or D[k]["mech"] == params)
        D[k]["reproduces_printed"] = bool(ok and pok)
        D[k]["printed_row"] = label
        print(f"    {stem:<9}{shown}")
        print(f"    {'':<9}" + "  ".join(f"{w:+.3f}" for w in want)
              + f"   <- {label}"
              + ("" if ok else f"   MISMATCH (max {bestd:.4f})"))
        if not pok:
            print(f"    {'':<9}parameters {D[k]['mech']:,} in the archive against {params:,} "
                  f"printed for that row")
        if len(cands) > 1 and ok:
            print(f"    {'':<9}of {len(cands)} candidate rows for this stem, that is the one it "
                  f"reproduces, so the sweep trained that configuration")
        if not ok:
            bad.append(stem)
        if not pok:
            bad.append(f"{stem} (parameters)")

    # ---- the comparison the claim rests on ----
    print("\n  within-lesion value by seed, and folds above the centered Gaussian at +0.012")
    print(f"    {'stem':<9}{'seed 42':>11}{'seed 43':>11}{'seed 44':>11}"
          f"{'above':>9}{'fold min':>10}{'fold max':>10}")
    GW = 0.012
    for stem in STEMS:
        have = [s for s in SEEDS if f"{stem}_{s}" in D]
        if not have:
            continue
        cells, allf = [], []
        for s in SEEDS:
            k = f"{stem}_{s}"
            if k in D:
                cells.append(f"{sum(D[k]['within']) / 5:+.3f}")
                allf += D[k]["within"]
            else:
                cells.append("-")
        above = sum(1 for w in allf if w > GW)
        print(f"    {stem:<9}{cells[0]:>11}{cells[1]:>11}{cells[2]:>11}"
              f"{f'{above}/{len(allf)}':>9}{min(allf):>+10.4f}{max(allf):>+10.4f}")

    print("\n  pointing score range across the seeds, for the same mechanisms")
    for stem in STEMS:
        pts = [sum(D[f"{stem}_{s}"]["point"]) / 5 for s in SEEDS if f"{stem}_{s}" in D]
        if len(pts) > 1:
            print(f"    {stem:<9}{min(pts):.3f} .. {max(pts):.3f}   range {max(pts) - min(pts):.3f}")

    if problems:
        print(f"\n  {len(problems)} thing(s) to resolve before the table is rebuilt")
        for p in problems:
            print(f"    - {p}")
    if bad:
        print(f"\n  seed-42 mismatches: {', '.join(bad)}. A mismatch means either the archive is "
              f"not the\n  run the row was printed from, or the row is in the other reading. "
              f"Resolve before writing.")

    D["_note"] = ("Pooled reading only, per-fold. Means and ddof=1 standard deviations are "
                  "computed by the patcher, not stored. 'where' records the directory each "
                  "seed's archives were read from, because head_swap.py does not put the seed in "
                  "the filename. 'reproduces_printed' is the seed-42 check against Table S10.")
    if a.out:
        json.dump(D, open(a.out, "w"), indent=1, sort_keys=True)
        print(f"\n  written: {a.out}  ({len([k for k in D if not k.startswith('_')])} entries)")
    else:
        print("\n  --out not given, nothing written. Re-run with:")
        print("    python collect_seed_rows.py --out seed_folds.json")
    return 1 if (problems or bad) else 0


if __name__ == "__main__":
    sys.exit(main())
