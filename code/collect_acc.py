#!/usr/bin/env python3
"""The readout's own validation accuracy, per saliency mechanism."""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

# The unsupervised mechanism block of Table S10, in the order the table prints it, plus the
STEMS = [
    ("Scale and shift, per channel", "ssf"),
    ("Linear score (rank 1)", "lowrank"),
    ("Class activation map", "cam"),
    ("Position-wise MLP", "mlp"),
    ("Image-conditional", "cond"),
    ("MLP at the third stage", "s3_mlp"),
    ("Conditional at the third stage", "s3_cond"),
]
SUP = [("Position-wise MLP + masks", "mlp_sup0.2"),
       ("MLP at the third stage + masks", "s3_mlp_sup0.2")]


def acc_of(path):
    """The stored accuracy, or None where the run did not compute one."""
    with np.load(path, allow_pickle=True) as z:
        if "readout_val_acc" not in z:
            return None
        v = float(z["readout_val_acc"])
    return None if (v < 0 or not np.isfinite(v)) else v


def folds_for(out, stem):
    pat = re.compile(r"^hs_" + re.escape(stem) + r"_f(\d+)\.npz$")
    got = {}
    for p in sorted(glob.glob(os.path.join(out, f"hs_{stem}_f*.npz"))):
        m = pat.match(os.path.basename(p))
        if m:
            got[int(m.group(1))] = p
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/headswap",
                    help="the directory holding hs_*.npz")
    ap.add_argument("--json", default="", help="also write the values here")
    a = ap.parse_args()

    rows, values = [], {}
    for label, stem in STEMS + SUP:
        fs = folds_for(a.out, stem)
        if not fs:
            rows.append((label, stem, None, "no archive"))
            continue
        accs = {f: acc_of(p) for f, p in sorted(fs.items())}
        have = [v for v in accs.values() if v is not None]
        if not have:
            rows.append((label, stem, None,
                         f"{len(accs)} folds, none stores an accuracy"))
            continue
        note = "" if len(have) == len(accs) else f"{len(have)} of {len(accs)} folds"
        rows.append((label, stem, (float(np.mean(have)), float(np.std(have)), len(have)), note))
        values[stem] = {"mean": float(np.mean(have)), "sd": float(np.std(have)),
                        "folds": len(have),
                        "per_fold": {str(k): v for k, v in accs.items()}}

    w = max(len(l) for l, _, _, _ in rows)
    print(f"\n  readout validation accuracy, backbone frozen, by saliency mechanism")
    print(f"  {'mechanism':<{w}}  {'folds':>5}  {'accuracy':>18}")
    measured = []
    for label, stem, stat, note in rows:
        if stat is None:
            print(f"  {label:<{w}}  {'--':>5}  {'--':>18}   {note}")
            continue
        m, sd, n = stat
        print(f"  {label:<{w}}  {n:>5}  {m:>9.4f} +/- {sd:.4f}" + (f"   {note}" if note else ""))
        if stem in dict(STEMS).values() or stem in [s for _, s in STEMS]:
            measured.append((label, m))

    if len(measured) >= 2:
        lo = min(measured, key=lambda t: t[1])
        hi = max(measured, key=lambda t: t[1])
        print(f"\n  across the {len(measured)} mechanisms the spread is "
              f"{100 * (hi[1] - lo[1]):.2f} pp, from {lo[0]} at {lo[1]:.4f} "
              f"to {hi[0]} at {hi[1]:.4f}")
        print("  Read this against the fold-to-fold spread of a single mechanism above, not")
        print("  against the pipeline ablation: the backbone is frozen here and retrained there,")
        print("  and the splits differ, so the two numbers are not comparable.")

    if a.json:
        json.dump(values, open(a.json, "w"), indent=1, sort_keys=True)
        print(f"\n  wrote {a.json}  ({len(values)} stems)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
