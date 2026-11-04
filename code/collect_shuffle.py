#!/usr/bin/env python3
"""Aggregate the within-lesion shuffle control over the five folds.

    python collect_shuffle.py
    python collect_shuffle.py --latex
"""
import argparse
import glob
import os
import sys

import numpy as np

DEFAULT = "outputs/headswap"


def sd1(v):
    m = sum(v) / len(v)
    return (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5 if len(v) > 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT)
    ap.add_argument("--latex", action="store_true", help="also print the LaTeX rows")
    a = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(a.root, "refs2_shuf_f[0-4].npz")))
    print(f"  {len(paths)} archive(s) in {a.root}")
    if len(paths) != 5:
        sys.exit(f"REFUSED: expected five folds, found {len(paths)}. "
                 f"{'Still running?' if paths else 'Nothing to read.'}")

    names, shas, rows, gauss, nwithin = None, set(), [], [], []
    for p in paths:
        with np.load(p, allow_pickle=True) as z:
            nm = [str(x) for x in z["names"]]
            if names is None:
                names = nm
            elif nm != names:
                sys.exit(f"REFUSED: {os.path.basename(p)} has a different map list.")
            shas.add(str(z["source_sha1"]) if "source_sha1" in z.files else "?")
            nwithin.append(int(z["n_within"]) if "n_within" in z.files else -1)
            R = z["metrics"]                      # (images, maps, 3) pointing, energy, IoU
            rows.append(dict(point=np.nanmean(R[:, :, 0], axis=0),
                             energy=np.nanmean(R[:, :, 1], axis=0),
                             iou=np.nanmean(R[:, :, 2], axis=0),
                             within=np.asarray(z["within"], dtype=float),
                             faith=np.asarray(z["faith"], dtype=float),
                             R=R))
    if len(shas) != 1:
        sys.exit(f"REFUSED: the folds carry {len(shas)} source hashes {sorted(shas)}; they were "
                 f"measured by different versions of the code.")
    sha = shas.pop()
    print(f"  source {sha};  images with a usable lesion area per fold: "
          f"{', '.join(str(x) for x in nwithin)}")

    base = [n for n in names if not n.endswith("_shuf")]
    pairs = [(n, n + "_shuf") for n in base if n + "_shuf" in names]
    if not pairs:
        sys.exit("REFUSED: no shuffled maps in these archives. Was --shuffle passed?")

    # ---- the invariance check, per fold ----
    EPS32 = float(np.finfo(np.float32).eps)
    print("\n  invariance of the overlap measures, over the five folds")
    print(f"    {'map':<10}{'d IoU':>10}{'d energy':>12}"
          f"{'pointing flips':>16}{'share':>9}{'d mean pointing':>18}")
    hard = soft = 0.0
    worst_share, worst_dmean = 0.0, 0.0
    for b, s in pairs:
        i0, i1 = names.index(b), names.index(s)
        di = max(float(np.nanmax(np.abs(r["R"][:, i0, 2] - r["R"][:, i1, 2]))) for r in rows)
        de = max(float(np.nanmax(np.abs(r["R"][:, i0, 1] - r["R"][:, i1, 1]))) for r in rows)
        flip = sum(int(np.nansum(r["R"][:, i0, 0] != r["R"][:, i1, 0])) for r in rows)
        tot = sum(r["R"].shape[0] for r in rows)
        dmean = max(abs(float(np.nanmean(r["R"][:, i0, 0]) - np.nanmean(r["R"][:, i1, 0])))
                    for r in rows)
        hard = max(hard, di)
        soft = max(soft, de)
        worst_share = max(worst_share, flip / tot)
        worst_dmean = max(worst_dmean, dmean)
        print(f"    {b:<10}{di:>10.2e}{de:>12.2e}{f'{flip}/{tot}':>16}"
              f"{flip / tot:>8.2%}{dmean:>18.4f}")
    if hard != 0.0:
        sys.exit(f"\n  REFUSED: the area-matched IoU moved by {hard:.2e}. It compares counts of "
                 f"selected positions and is exact under an in-lesion permutation, so the "
                 f"permutation is reaching outside the mask. This is a bug, not a result.")
    if soft > 100 * EPS32:
        sys.exit(f"\n  REFUSED: energy moved by {soft:.2e}, which is {soft / EPS32:.0f} float32 "
                 f"epsilon and too large for summation order alone.")
    print(f"\n    IoU bit-identical, so the permutation stays inside the mask; energy within "
          f"{soft / EPS32:.1f} float32 epsilon.")
    if worst_share == 0:
        print("    No pointing flips at all.")
    else:
        print(f"    Pointing flips on at most {worst_share:.2%} of images, moving a fold's mean "
              f"pointing\n    score by at most {worst_dmean:.4f}: these are plateau ties at the "
              f"lesion boundary,\n    where the arg max is settled by raster order.")
    if worst_share > 0.01:
        sys.exit(f"\n  REFUSED: {worst_share:.1%} of images flip their pointing score. Ties at "
                 f"the\n  boundary are rare; this many means the permutation is not doing what "
                 f"it should.")

    # ---- the table ----
    def agg(key, idx):
        v = [float(r[key][idx]) for r in rows]
        return sum(v) / len(v), sd1(v)

    print("\n  within-lesion faithfulness, five-fold mean and sample s.d.")
    print(f"    {'map':<24}{'IoU':>16}{'within':>18}{'shuffled':>18}{'change':>10}")
    out = []
    for b, s in pairs:
        i0, i1 = names.index(b), names.index(s)
        iou = agg("iou", i0)
        w0, w1 = agg("within", i0), agg("within", i1)
        out.append((b, iou, w0, w1))
        print(f"    {b:<24}{iou[0]:>9.3f} ± {iou[1]:.3f}"
              f"{w0[0]:>+11.4f} ± {w0[1]:.4f}{w1[0]:>+11.4f} ± {w1[1]:.4f}"
              f"{w1[0] - w0[0]:>+10.4f}")

    if a.latex:
        print("\n  LaTeX rows (map & IoU & within & shuffled)")
        for b, iou, w0, w1 in out:
            print(f"    {b} & ${iou[0]:.3f} \\pm {iou[1]:.3f}$ & "
                  f"${w0[0]:+.3f} \\pm {w0[1]:.3f}$ & ${w1[0]:+.3f} \\pm {w1[1]:.3f}$ \\\\")

    print("\n  Read this as: the IoU column is identical between a map and its shuffled twin by")
    print("  construction, and the within column collapses. A measure that moves when only the")
    print("  ordering inside the lesion changes, while every overlap measure does not, is")
    print("  measuring something the overlap measures cannot see.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
