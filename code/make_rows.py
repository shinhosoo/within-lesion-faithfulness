#!/usr/bin/env python3
"""The explanation-head table rows, at one seed and pooled over seeds, straight from the archives.

    python3 make_rows.py --out outputs/headswap
    python3 make_rows.py --out outputs/headswap --json rows.json
"""

import argparse
import json
import os

import numpy as np

COLS = (("pointing", 0), ("energy", 1), ("IoU", 2))
MEASURES = ("pointing", "energy", "IoU", "Ins-Del", "within")
FOLDS = range(5)

STEMS = (("Unsupervised", "s3_mlp"),
         ("Mask supervision ($\\lambda = 0.2$)", "s3_mlp_sup0.2"))


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
                R = z["metrics"]
                u = {name: float(np.nanmean(R[:, 0, c])) for name, c in COLS}
                fh = z["faith_head"]
                u["Ins-Del"] = float(fh[2])
                u["within"] = float(fh[5]) if len(fh) > 5 else float("nan")
                u["seed"], u["fold"] = seed, fold
            got.append(u)
    return got, missing


def stat(us, measure):
    v = np.array([u[measure] for u in us], float)
    return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else 0.0), len(v)


def fmt(m, sd, signed=False):
    s = f"{m:+.3f}" if signed else f"{m:.3f}"
    return f"${s} \\pm {sd:.3f}$"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/headswap")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--single", type=int, default=42,
                    help="the seed the published rows were computed from")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]

    dump, problems = {}, []
    for title, stem in STEMS:
        pooled, miss = units(a.out, stem, seeds)
        problems += [f"missing: {p}" for p in miss]
        one = [u for u in pooled if u["seed"] == a.single]
        if not one or not pooled:
            problems.append(f"no archives for {stem}")
            continue

        print(f"\n=== {title}   ({stem}) " + "=" * max(0, 46 - len(title)))
        print(f"  {'measure':<10}{'seed ' + str(a.single) + ', 5 folds':>22}"
              f"{'all seeds, ' + str(len(pooled)) + ' units':>24}{'moves by':>10}")
        for measure in MEASURES:
            m1, s1, n1 = stat(one, measure)
            m2, s2, n2 = stat(pooled, measure)
            print(f"  {measure:<10}{m1:>13.4f} +- {s1:<6.4f}{m2:>15.4f} +- {s2:<6.4f}"
                  f"{abs(m2 - m1):>10.4f}")
            for tag, (m, s, n) in (("single", (m1, s1, n1)), ("pooled", (m2, s2, n2))):
                dump[f"rows|{stem}|{measure} {tag}"] = m
                dump[f"rows|{stem}|{measure} {tag} sd"] = s
                dump[f"rows|{stem}|{measure} {tag} n"] = n

        # The LaTeX, in the column order each table uses.
        signed = {"Ins-Del", "within"}
        for tag, us in (("single seed " + str(a.single), one), (f"{len(pooled)} units", pooled)):
            t3 = " & ".join(fmt(*stat(us, m)[:2], signed=m in signed)
                            for m in ("pointing", "energy", "IoU", "Ins-Del"))
            s16 = " & ".join(fmt(*stat(us, m)[:2], signed=m in signed)
                             for m in ("pointing", "IoU", "Ins-Del", "within"))
            print(f"  Table 1  ({tag:<14}) {title} & {t3} \\\\")
            print(f"  S16      ({tag:<14}) {title} & MambaOut-Tiny & {s16} \\\\")

    # Which measures survive the choice and which do not, so the edit can be scoped.
    print("\n=== what the choice changes " + "=" * 38)
    for title, stem in STEMS:
        pooled, _ = units(a.out, stem, seeds)
        one = [u for u in pooled if u["seed"] == a.single]
        if not one or not pooled:
            continue
        moved = []
        for measure in MEASURES:
            m1, s1, _ = stat(one, measure)
            m2, s2, _ = stat(pooled, measure)
            # Does the printed row change at three decimals, which is what the tables show?
            if f"{m1:.3f}" != f"{m2:.3f}" or f"{s1:.3f}" != f"{s2:.3f}":
                moved.append(f"{measure} ({m1:.3f}+-{s1:.3f} -> {m2:.3f}+-{s2:.3f})")
        if moved:
            print(f"  {title}: {len(moved)} of {len(MEASURES)} printed values change")
            for x in moved:
                print(f"      {x}")
        else:
            print(f"  {title}: no printed value changes")

    for p in problems:
        print(f"  PROBLEM  {p}")
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(dump, fh, indent=1, sort_keys=True)
        print(f"\n  wrote {a.json}  ({len(dump)} keys)")
    print()


if __name__ == "__main__":
    main()
