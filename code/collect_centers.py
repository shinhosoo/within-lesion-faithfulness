#!/usr/bin/env python3
"""Per-fold within-lesion values of the two learning-free references, for the paired tests.

    python collect_centers.py
"""
import glob
import os
import sys

import numpy as np

ROOT = "outputs/headswap"
STEMS = [("mlp", "hs_mlp_mapsoftmax"), ("s3_mlp", "hs_s3_mlp_mapsoftmax"),
         ("cond", "hs_cond_mapsoftmax"), ("s3_cond", "hs_s3_cond_mapsoftmax"),
         ("gradgc3_mlp", "hs_gradgc3_mlp_sup0.2"), ("mlp_sup", "hs_mlp_sup0.2"),
         ("s3_mlp_sup", "hs_s3_mlp_sup0.2"), ("s3_cond_sup", "hs_s3_cond_sup0.2")]


def sd1(v):
    m = sum(v) / len(v)
    return (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5


def main():
    print("  centered Gaussian, within-lesion, from each head's own archive (faith_center[5])\n")
    allv = {}
    for key, stem in STEMS:
        paths = sorted(glob.glob(os.path.join(ROOT, f"{stem}_f[0-4].npz")))
        if len(paths) != 5:
            print(f"  {key:<14}{len(paths)}/5, skipped")
            continue
        vals = []
        for p in paths:
            with np.load(p, allow_pickle=True) as z:
                if "faith_center" not in z.files:
                    vals = None
                    break
                v = np.asarray(z["faith_center"], dtype=float).ravel()
                if v.size != 6 or abs((v[4] - v[3]) - v[5]) > 1e-6:
                    vals = None
                    break
                vals.append(float(v[5]))
        if vals is None:
            print(f"  {key:<14}faith_center missing or malformed")
            continue
        allv[key] = vals
        print(f"  {key:<14}" + " ".join(f"{x:+.6f}" for x in vals)
              + f"   mean {sum(vals)/5:+.4f} ± {sd1(vals):.4f}")

    print("\n  do the Gaussians agree across heads? they should: same backbone, same folds,")
    print("  and the map knows nothing about the head.")
    if len(allv) > 1:
        ks = list(allv)
        worst = max(abs(allv[a][i] - allv[b][i])
                    for a in ks for b in ks for i in range(5))
        print(f"    largest disagreement on any fold: {worst:.6f}")
        if worst > 1e-4:
            print("    NOTE: the heads were not all measured against the same reference pass;")
            print("    use each head's own Gaussian for its own paired test, not a pooled one.")

    print("\n  random map, within-lesion, from refs2_f*.npz")
    paths = sorted(glob.glob(os.path.join(ROOT, "refs2_f[0-4].npz")))
    if len(paths) != 5:
        print(f"    {len(paths)}/5 found; cannot report")
        return 1
    rnd, cen = [], []
    for p in paths:
        with np.load(p, allow_pickle=True) as z:
            nm = [str(x) for x in z["names"]]
            w = np.asarray(z["within"], dtype=float)
            rnd.append(float(w[nm.index("random")]))
            cen.append(float(w[nm.index("center")]))
    print("    random  " + " ".join(f"{x:+.6f}" for x in rnd)
          + f"   mean {sum(rnd)/5:+.4f} ± {sd1(rnd):.4f}")
    print("    center  " + " ".join(f"{x:+.6f}" for x in cen)
          + f"   mean {sum(cen)/5:+.4f} ± {sd1(cen):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
