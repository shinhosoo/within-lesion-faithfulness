#!/usr/bin/env python3
"""Which row of Table S10 does each head archive actually hold?

    python audit_archives.py --out outputs/headswap
    python audit_archives.py --out outputs/headswap/pre_retrain
"""
import argparse
import glob
import os
import re
import sys

import numpy as np

COLS = ("pointing", "energy", "IoU", "Ins-Del", "within")

# (block, label, pointing, energy, IoU, Ins-Del, within), parsed from Table S10.
S15 = [
    ("unsup", "Scale and shift, per channel", 0.276, 0.273, 0.200, +0.006, -0.0680),
    ("unsup", "Linear score (rank 1)", 0.515, 0.295, 0.308, +0.019, -0.0840),
    ("unsup", "Class activation map", 0.277, 0.342, 0.245, +0.017, -0.0400),
    ("unsup", "Position-wise MLP", 0.407, 0.285, 0.268, -0.024, -0.1180),
    ("unsup", "Image-conditional", 0.265, 0.273, 0.174, -0.034, -0.0260),
    ("sup", "Linear score (rank 1)", 0.953, 0.689, 0.780, +0.143, -0.0440),
    ("sup", "Class activation map", 0.952, 0.706, 0.792, +0.145, -0.0320),
    ("sup", "Position-wise MLP", 0.958, 0.779, 0.818, +0.154, -0.0120),
    ("sup", "Image-conditional", 0.957, 0.757, 0.809, +0.145, -0.0520),
    ("interv", "MLP at the third stage", 0.667, 0.567, 0.474, +0.134, -0.0110),
    ("interv", "MLP at the third stage + masks", 0.987, 0.848, 0.872, +0.176, -0.0080),
    ("interv", "Conditional at the third stage", 0.612, 0.591, 0.343, +0.059, -0.0130),
    ("interv", "Conditional at the third stage + masks", 0.990, 0.816, 0.834, +0.177, -0.0050),
    ("interv", "MLP, gradient target", 0.940, 0.766, 0.793, +0.146, -0.0010),
    ("rect", "Rank 1, second stage", 0.514, 0.524, 0.298, +0.050, -0.0650),
    ("rect", "MLP, second stage", 0.406, 0.416, 0.271, -0.007, -0.1250),
    ("rect", "Conditional, second stage", 0.259, 0.244, 0.182, -0.017, -0.0050),
    ("rect", "MLP, third stage", 0.590, 0.590, 0.274, +0.050, +0.0010),
    ("rect", "Conditional, third stage", 0.261, 0.255, 0.266, +0.017, -0.0010),
]

# Which row the table prints for each archive stem. A stem whose archive matches a DIFFERENT row
# is the defect this script looks for.
OWNER = {
    "ssf": ("unsup", "Scale and shift, per channel"),
    "lowrank": ("unsup", "Linear score (rank 1)"),
    "cam": ("unsup", "Class activation map"),
    "mlp": ("unsup", "Position-wise MLP"),
    "cond": ("unsup", "Image-conditional"),
    "ssf_sup0.2": ("sup", "Scale and shift, per channel"),
    "lowrank_sup0.2": ("sup", "Linear score (rank 1)"),
    "cam_sup0.2": ("sup", "Class activation map"),
    "mlp_sup0.2": ("sup", "Position-wise MLP"),
    "cond_sup0.2": ("sup", "Image-conditional"),
    "s3_mlp": ("interv", "MLP at the third stage"),
    "s3_mlp_sup0.2": ("interv", "MLP at the third stage + masks"),
    "s3_cond": ("interv", "Conditional at the third stage"),
    "s3_cond_sup0.2": ("interv", "Conditional at the third stage + masks"),
}

TOL = 1.5e-3            # the table prints three decimals, so this is about one printed digit
NAME = re.compile(r"^hs_(?P<stem>.+)_f(?P<fold>\d+)\.npz$")


def read_faith(F, key):
    """[2] is the whole-image insertion-minus-deletion value, [5] the within-lesion one."""
    F = np.asarray(F, dtype=float).ravel()
    if F.shape[0] != 6:
        sys.exit(f"REFUSED: {key} has {F.shape[0]} entries, not 6; re-derive the layout.")
    for lo, hi, d, what in ((0, 1, 2, "whole-image"), (3, 4, 5, "within-lesion")):
        if not np.isclose(F[hi] - F[lo], F[d], atol=5e-4):
            sys.exit(f"REFUSED: {key} {what} entries fail [{hi}]-[{lo}]=[{d}]; layout differs.")
    return float(F[2]), float(F[5])


def fold_values(path):
    with np.load(path, allow_pickle=True) as z:
        if "metrics" not in z or "faith_head" not in z:
            return None
        R = z["metrics"]
        R = R[:, 0, :] if R.ndim == 3 else R          # index 0 is the head's own map
        p, e, i = (float(np.nanmean(R[:, k])) for k in range(3))
        d, w = read_faith(z["faith_head"], os.path.basename(path))
    return dict(zip(COLS, (p, e, i, d, w)))


def best_match(v):
    """The row of S15 this archive is closest to, and how far off it is."""
    best, how = None, 1e9
    for row in S15:
        dev = max(abs(v[c] - row[2 + k]) for k, c in enumerate(COLS))
        if dev < how:
            best, how = row, dev
    return best, how


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/headswap", help="directory holding hs_*.npz")
    ap.add_argument("--tol", type=float, default=TOL)
    a = ap.parse_args()

    stems = {}
    for p in sorted(glob.glob(os.path.join(a.out, "hs_*.npz"))):
        m = NAME.match(os.path.basename(p))
        if m:
            stems.setdefault(m.group("stem"), {})[int(m.group("fold"))] = p
    if not stems:
        sys.exit(f"REFUSED: no hs_*.npz in {a.out}")

    print(f"  {a.out}\n")
    print(f"  {'stem':<17}{'folds':>5}  "
          + "".join(f"{c:>9}" for c in COLS) + "   matches")
    wrong, missing, nomatch = [], [], []
    for stem, folds in sorted(stems.items()):
        per = [fold_values(folds[f]) for f in sorted(folds)]
        per = [x for x in per if x]
        if not per:
            print(f"  {stem:<17}{len(folds):>5}   no metrics stored")
            continue
        v = {c: float(np.mean([x[c] for x in per])) for c in COLS}
        row, dev = best_match(v)
        owner = OWNER.get(stem)
        line = (f"  {stem:<17}{len(per):>5}  "
                + f"{v['pointing']:>9.3f}{v['energy']:>9.3f}{v['IoU']:>9.3f}"
                + f"{v['Ins-Del']:>+9.3f}{v['within']:>+9.4f}   ")
        if dev > a.tol:
            line += f"nothing (nearest {row[0]}/{row[1]}, off by {dev:.3f})"
            nomatch.append(stem)
        elif owner is None:
            line += f"{row[0]}/{row[1]}   (stem not in the table's map)"
        elif (row[0], row[1]) == owner:
            line += f"{row[0]}/{row[1]}   ok"
        else:
            line += f"{row[0]}/{row[1]}   ** the table prints {owner[0]}/{owner[1]} **"
            wrong.append((stem, owner, (row[0], row[1])))
        print(line)
        if len(per) < 5:
            missing.append((stem, sorted(folds)))

    print()
    for stem, owner, got in wrong:
        print(f"  WRONG ROW  {stem}: the archive holds {got[0]}/{got[1]} while Table S10 prints")
        print(f"             {owner[0]}/{owner[1]} for it. The row cannot be regenerated from")
        print(f"             this archive, and any comparison against it measures the wrong thing.")
    for stem, folds in missing:
        print(f"  INCOMPLETE {stem}: folds {folds}; a five-fold mean cannot be formed.")
    for stem in nomatch:
        print(f"  NO MATCH   {stem}: matches no row of the table at tol {a.tol}.")
    if not (wrong or missing or nomatch):
        print(f"  every archive in {a.out} matches the row Table S10 prints for it.")
        return 0
    print(f"\n  {len(wrong)} wrong row(s), {len(missing)} incomplete, {len(nomatch)} unmatched.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
