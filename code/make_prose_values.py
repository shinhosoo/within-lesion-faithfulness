#!/usr/bin/env python3
"""One dump of everything the body text might quote about the explanation head.

    python3 make_prose_values.py --out outputs/headswap --json prose_values.json
    python3 make_prose_values.py --out outputs/headswap --json prose_values.json --merge rows.json --merge fig3_values.json
"""

import argparse
import glob
import json
import os
import re

import numpy as np

COLS = (("pointing", 0), ("energy", 1), ("IoU", 2))
MEASURES = ("pointing", "energy", "IoU", "Ins-Del", "within")
REF_NAMES = ("gc1", "gc2", "gc3", "gc4", "ig", "mask", "center", "random")
FOLDS = range(5)
STEMS = {"unsupervised": "s3_mlp", "mask supervised": "s3_mlp_sup0.2"}


def seed_dir(out, seed):
    return out if int(seed) == 42 else os.path.join(out, f"seed{int(seed)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/headswap")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--single", type=int, default=42)
    ap.add_argument("--json", default="prose_values.json")
    ap.add_argument("--merge", action="append", default=[],
                    help="another dump to fold in, so one file goes back")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]

    V, notes = {}, []

    # --- the head rows, and the per-image split by prediction correctness -------------------
    for label, stem in STEMS.items():
        per_unit = {m: [] for m in MEASURES}
        split = {"flat_pred": {True: [], False: []}, "readout_pred": {True: [], False: []}}
        for seed in seeds:
            for fold in FOLDS:
                p = os.path.join(seed_dir(a.out, seed), f"hs_{stem}_f{fold}.npz")
                if not os.path.exists(p):
                    notes.append(f"missing {p}")
                    continue
                with np.load(p, allow_pickle=True) as z:
                    R = z["metrics"]
                    for name, c in COLS:
                        per_unit[name].append(float(np.nanmean(R[:, 0, c])))
                    fh = z["faith_head"]
                    per_unit["Ins-Del"].append(float(fh[2]))
                    per_unit["within"].append(float(fh[5]) if len(fh) > 5 else np.nan)
                    iou = R[:, 0, 2]
                    y = z["label"] if "label" in z.files else None
                    for src in ("flat_pred", "readout_pred"):
                        if y is None or src not in z.files:
                            continue
                        ok = np.asarray(z[src]) == np.asarray(y)
                        for truth in (True, False):
                            sel = iou[ok == truth]
                            sel = sel[~np.isnan(sel)]
                            if sel.size:
                                split[src][truth].append(float(sel.mean()))
                if seed == a.single:
                    pass
        for m, v in per_unit.items():
            if not v:
                continue
            arr = np.array(v, float)
            V[f"head|{label}|{m}"] = float(np.nanmean(arr))
            V[f"head|{label}|{m} sd"] = float(np.nanstd(arr, ddof=1))
        # the single-seed reading too, so a sentence quoting the published row still resolves
        # and is reported as such rather than as an unexplained number
        one = {m: [] for m in MEASURES}
        for fold in FOLDS:
            p = os.path.join(seed_dir(a.out, a.single), f"hs_{stem}_f{fold}.npz")
            if not os.path.exists(p):
                continue
            with np.load(p, allow_pickle=True) as z:
                R = z["metrics"]
                for name, c in COLS:
                    one[name].append(float(np.nanmean(R[:, 0, c])))
                fh = z["faith_head"]
                one["Ins-Del"].append(float(fh[2]))
                one["within"].append(float(fh[5]) if len(fh) > 5 else np.nan)
        for m, v in one.items():
            if v:
                V[f"head|{label} (seed {a.single} only)|{m}"] = float(np.nanmean(v))
                V[f"head|{label} (seed {a.single} only)|{m} sd"] = float(
                    np.nanstd(np.array(v, float), ddof=1))
        # Per seed, and the distance between the seed means. The Methods paragraph claims the
        for seed in seeds:
            got = {m: [] for m in MEASURES}
            for fold in FOLDS:
                p = os.path.join(seed_dir(a.out, seed), f"hs_{stem}_f{fold}.npz")
                if not os.path.exists(p):
                    continue
                with np.load(p, allow_pickle=True) as z:
                    R = z["metrics"]
                    for name, c in COLS:
                        got[name].append(float(np.nanmean(R[:, 0, c])))
                    fh = z["faith_head"]
                    got["Ins-Del"].append(float(fh[2]))
                    got["within"].append(float(fh[5]) if len(fh) > 5 else np.nan)
                    # The head's own validation accuracy, which separates "the map moved" from
                    # "the head did not train". -1 means the run did not record it.
                    if "readout_val_acc" in z.files:
                        acc = float(z["readout_val_acc"])
                        if acc >= 0:
                            got.setdefault("val acc", []).append(acc)
            for m in list(MEASURES) + ["val acc"]:
                if got.get(m):
                    V[f"seed|{label}|{m} seed {seed}"] = float(
                        np.nanmean(np.array(got[m], float)))
        for m in list(MEASURES) + ["val acc"]:
            means = [V[f"seed|{label}|{m} seed {s}"] for s in seeds
                     if f"seed|{label}|{m} seed {s}" in V]
            if len(means) > 1:
                V[f"seed|{label}|{m} seed-mean span"] = float(max(means) - min(means))

        for src, d in split.items():
            which = ("the frozen classifier's prediction" if src == "flat_pred"
                     else "the head's own discarded prediction")
            for truth, v in d.items():
                if v:
                    V[f"head|{label}|IoU on {'correct' if truth else 'misclassified'} "
                      f"({which})"] = float(np.mean(v))

    # --- the reference maps -----------------------------------------------------------------
    ref = {nm: {} for nm in REF_NAMES}
    for fold in FOLDS:
        ov = os.path.join(seed_dir(a.out, a.single), f"refs_ov_f{fold}.npz")
        fa = os.path.join(seed_dir(a.out, a.single), f"refs2_f{fold}.npz")
        if os.path.exists(ov):
            with np.load(ov, allow_pickle=True) as z:
                names = [str(x) for x in z["names"]]
                R = z["metrics"]
                for nm in REF_NAMES:
                    if nm in names:
                        i = names.index(nm)
                        for name, c in COLS:
                            ref[nm].setdefault(name, []).append(float(np.nanmean(R[:, i, c])))
        if os.path.exists(fa):
            with np.load(fa, allow_pickle=True) as z:
                names = [str(x) for x in z["names"]]
                for nm in REF_NAMES:
                    if nm in names:
                        i = names.index(nm)
                        ref[nm].setdefault("Ins-Del", []).append(float(z["faith"][i]))
                        ref[nm].setdefault("within", []).append(float(z["within"][i]))
    for nm, d in ref.items():
        for m, v in d.items():
            if v:
                V[f"reference|{nm}|{m}"] = float(np.mean(v))
                V[f"reference|{nm}|{m} sd"] = float(np.std(np.array(v, float), ddof=1))

    # --- spans, for the sentences that say a set of maps falls within some distance ---------
    for m in ("Ins-Del", "within", "IoU"):
        group = {}
        for key, nm in (("mask supervised", None), ("unsupervised", None)):
            if f"head|{key}|{m}" in V:
                group[key] = V[f"head|{key}|{m}"]
        for nm in ("center", "gc2", "gc4"):
            if f"reference|{nm}|{m}" in V:
                group[nm] = V[f"reference|{nm}|{m}"]
        if len(group) > 1:
            trio = {k: v for k, v in group.items() if k in ("mask supervised", "center", "gc2")}
            if len(trio) == 3:
                V[f"span|supervised, centre and Grad-CAM stage 2|{m}"] = (
                    max(trio.values()) - min(trio.values()))
            trio4 = {k: v for k, v in group.items()
                     if k in ("mask supervised", "center", "gc4")}
            if len(trio4) == 3:
                V[f"span|supervised, centre and Grad-CAM final|{m}"] = (
                    max(trio4.values()) - min(trio4.values()))

    for extra in a.merge:
        if not os.path.exists(extra):
            notes.append(f"merge file not found: {extra}")
            continue
        with open(extra, encoding="utf-8") as fh:
            for k, v in json.load(fh).items():
                if isinstance(v, (int, float)):
                    V.setdefault(k, v)

    with open(a.json, "w") as fh:
        json.dump(V, fh, indent=1, sort_keys=True)
    print(f"  wrote {a.json}  ({len(V)} keys)")
    for n in notes:
        print(f"  note: {n}")

    print("\n  the values the stale sentences need")
    for key in sorted(V):
        if re.search(r"\|(IoU|Ins-Del|within)( sd)?$|IoU on ", key) and (
                "unsupervised" in key or "mask supervised" in key or "center" in key):
            print(f"    {key:<78} {V[key]:+.4f}")
    print()


if __name__ == "__main__":
    main()
