#!/usr/bin/env python3
"""The unsupervised head read two ways: as relu(raw), as the manuscript reports it, and as

    python3 compare_map_modes.py --out outputs/headswap
    python3 compare_map_modes.py --out outputs/headswap --json map_modes.json
"""

import argparse
import json
import os

import numpy as np

COLS = (("pointing", 0), ("energy", 1), ("IoU", 2))
MEASURES = ("pointing", "energy", "IoU", "Ins-Del", "within")
SIGNED = {"Ins-Del", "within"}
FOLDS = range(5)
REFS = ("center", "gc2", "gc4", "random")

ROWS = [
    ("unsupervised, relu (as published)", "s3_mlp"),
    ("unsupervised, softmax (pooling map)", "s3_mlp_mapsoftmax"),
    ("mask supervised (softmax already)", "s3_mlp_sup0.2"),
]


def seed_dir(out, seed):
    return out if int(seed) == 42 else os.path.join(out, f"seed{int(seed)}")


def units(out, stem, seeds):
    got, missing = [], []
    for seed in seeds:
        for fold in FOLDS:
            p = os.path.join(seed_dir(out, seed), f"hs_{stem}_f{fold}.npz")
            if not os.path.exists(p):
                missing.append(p)
                continue
            with np.load(p, allow_pickle=True) as z:
                R = np.asarray(z["metrics"])
                u = {name: float(np.nanmean(R[:, 0, c])) for name, c in COLS}
                fh = np.asarray(z["faith_head"])
                u["Ins-Del"] = float(fh[2])
                u["within"] = float(fh[5]) if fh.size > 5 else float("nan")
                iou = R[:, 0, 2].astype(float)
                area = (np.asarray(z["lesion_area"]).astype(float)
                        if "lesion_area" in z.files else None)
                if area is not None:
                    ok = ~(np.isnan(iou) | np.isnan(area))
                    u["degenerate"] = float((np.abs(iou[ok] - area[ok]) <= 0.005).mean())
                u["seed"], u["fold"] = seed, fold
            got.append(u)
    return got, missing


def stat(us, m):
    v = np.array([u[m] for u in us if m in u], float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return float("nan"), float("nan"), 0
    return float(v.mean()), (float(v.std(ddof=1)) if v.size > 1 else 0.0), v.size


def fmt(m, sd, signed):
    return f"${m:+.3f} \\pm {sd:.3f}$" if signed else f"${m:.3f} \\pm {sd:.3f}$"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/headswap")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--single", type=int, default=42)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]

    dump, problems, table = {}, [], {}

    print(f"  {'row':<38}{'seed':>6}{'degen':>8}{'pointing':>10}{'energy':>9}"
          f"{'IoU':>9}{'Ins-Del':>10}{'within':>9}")
    for label, stem in ROWS:
        us, miss = units(a.out, stem, seeds)
        problems += [f"missing {p}" for p in miss]
        if not us:
            problems.append(f"no archive for {stem}")
            continue
        table[stem] = us
        for seed in seeds:
            sel = [u for u in us if u["seed"] == seed]
            if not sel:
                continue
            d = stat(sel, "degenerate")[0]
            vals = [stat(sel, m)[0] for m in MEASURES]
            print(f"  {label:<38}{seed:>6}{d:>7.1%}"
                  + "".join(f"{v:>{10 if m in SIGNED else 9}.4f}"
                            for v, m in zip(vals, MEASURES)))
        one = [u for u in us if u["seed"] == a.single]
        for m in MEASURES:
            for tag, sub in (("seed42", one), ("pooled", us)):
                mu, sd, n = stat(sub, m)
                dump[f"{stem}|{m} {tag}"] = mu
                dump[f"{stem}|{m} {tag} sd"] = sd
                dump[f"{stem}|{m} {tag} n"] = n
        dump[f"{stem}|degenerate share"] = stat(us, "degenerate")[0]
        print()

    # --- the references, which do not depend on the readout ---------------------------------
    print(f"  {'reference':<38}{'':>6}{'':>8}{'pointing':>10}{'energy':>9}"
          f"{'IoU':>9}{'Ins-Del':>10}{'within':>9}")
    ref = {}
    for fold in FOLDS:
        ov = os.path.join(seed_dir(a.out, a.single), f"refs_ov_f{fold}.npz")
        fa = os.path.join(seed_dir(a.out, a.single), f"refs2_f{fold}.npz")
        if os.path.exists(ov):
            with np.load(ov, allow_pickle=True) as z:
                names = [str(x) for x in z["names"]]
                R = np.asarray(z["metrics"])
                for nm in REFS:
                    if nm in names:
                        i = names.index(nm)
                        for name, c in COLS:
                            ref.setdefault(nm, {}).setdefault(name, []).append(
                                float(np.nanmean(R[:, i, c])))
        if os.path.exists(fa):
            with np.load(fa, allow_pickle=True) as z:
                names = [str(x) for x in z["names"]]
                for nm in REFS:
                    if nm in names:
                        i = names.index(nm)
                        ref.setdefault(nm, {}).setdefault("Ins-Del", []).append(
                            float(z["faith"][i]))
                        ref[nm].setdefault("within", []).append(float(z["within"][i]))
    for nm in REFS:
        if nm not in ref:
            continue
        vals = [float(np.mean(ref[nm][m])) if m in ref[nm] else float("nan") for m in MEASURES]
        print(f"  {nm:<38}{'':>6}{'':>8}"
              + "".join(f"{v:>{10 if m in SIGNED else 9}.4f}" for v, m in zip(vals, MEASURES)))
        for m, v in zip(MEASURES, vals):
            dump[f"reference|{nm}|{m}"] = v

    # --- the claims that depend on which reading is reported --------------------------------
    soft = table.get("s3_mlp_mapsoftmax")
    if soft and "center" in ref:
        print("\n  what changes if the softmax reading is the one reported "
              "(seed 42, five folds)")
        one = [u for u in soft if u["seed"] == a.single]
        relu_one = [u for u in table.get("s3_mlp", []) if u["seed"] == a.single]
        for m in MEASURES:
            r = stat(relu_one, m)[0]
            s = stat(one, m)[0]
            c = float(np.mean(ref["center"][m])) if m in ref["center"] else float("nan")
            print(f"    {m:<10} relu {r:+.4f}   softmax {s:+.4f}   centered Gaussian {c:+.4f}"
                  f"   {'below' if s < c else 'ABOVE'} the Gaussian")
        for nm in ("gc2", "random"):
            if nm in ref and "IoU" in ref[nm]:
                g = float(np.mean(ref[nm]["IoU"]))
                print(f"    IoU against {nm}: relu {stat(relu_one, 'IoU')[0] - g:+.3f}, "
                      f"softmax {stat(one, 'IoU')[0] - g:+.3f}")
        w = stat(one, "within")[0]
        print(f"    the within-lesion value is {w:+.4f}; the paper's central claim needs it "
              f"near zero, and it is {'still' if abs(w) < 0.03 else 'NOT'} near zero")

    # --- LaTeX, both readings, so the chosen one is copied and not retyped -------------------
    print("\n  Table 1 rows, seed 42 over five folds")
    for label, stem in ROWS:
        if stem not in table:
            continue
        one = [u for u in table[stem] if u["seed"] == a.single]
        cells = " & ".join(fmt(*stat(one, m)[:2], m in SIGNED)
                           for m in ("pointing", "energy", "IoU", "Ins-Del"))
        print(f"    {label:<38} & {cells} \\\\")

    for p in problems:
        print(f"  PROBLEM  {p}")
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(dump, fh, indent=1, sort_keys=True)
        print(f"\n  wrote {a.json}  ({len(dump)} keys)")


if __name__ == "__main__":
    main()
