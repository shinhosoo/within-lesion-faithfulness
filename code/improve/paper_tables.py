"""Per-class performance of the single-stage classifier, seven classes over five seeds.

    python -m improve.paper_tables
"""
import argparse
import os

import numpy as np
from scipy.stats import binomtest

from SkinCancer.configs.training_config import OUTPUT_ROOT

CLS = ["NV", "MEL", "BKL", "DF", "VASC", "BCC", "AKIEC"]
SEEDS = [42, 43, 44, 45, 46]
R = OUTPUT_ROOT


def ld(p):
    return np.load(p, allow_pickle=True) if os.path.exists(p) else None


def preds(name, s):
    """The single-stage classifier's test predictions for one seed."""
    assert name == "flat", name
    d = ld(os.path.join(R, "hier", ("flat" if s == 42 else f"flat_s{s}") + "_test_predictions.npz"))
    return (d["y_prob"], d["y_true"].astype(int)) if d is not None else None


# ------------------------------------------------------------ 1) per class
def class_wise():
    print("=" * 96)
    print(" 1) per-class performance - 1-stage (mean ± sd over 5 seeds, leakage-free split)")
    print("=" * 96)
    acc = {c: [] for c in range(7)}
    rec, pre, f1s, sup = dict(acc), {c: [] for c in range(7)}, {c: [] for c in range(7)}, {}
    rec = {c: [] for c in range(7)}
    for s in SEEDS:
        r = preds("flat", s)
        if r is None:
            continue
        P, y = r
        p = P.argmax(1)
        for c in range(7):
            tp = ((p == c) & (y == c)).sum(); fp = ((p == c) & (y != c)).sum()
            fn = ((p != c) & (y == c)).sum()
            rec[c].append(tp / max(tp + fn, 1)); pre[c].append(tp / max(tp + fp, 1))
            f1s[c].append(2 * tp / max(2 * tp + fp + fn, 1)); sup[c] = int((y == c).sum())
    if not rec[0]:
        print("  no prediction files"); return
    print(f"  {'class':<8}{'support':>9}{'recall':>18}{'precision':>18}{'F1':>18}")
    for c in range(7):
        print(f"  {CLS[c]:<8}{sup[c]:>9}"
              f"{np.mean(rec[c]):>12.3f}±{np.std(rec[c], ddof=1):.3f}"
              f"{np.mean(pre[c]):>12.3f}±{np.std(pre[c], ddof=1):.3f}"
              f"{np.mean(f1s[c]):>12.3f}±{np.std(f1s[c], ddof=1):.3f}")
    print(f"  {'macro':<8}{sum(sup.values()):>9}"
          f"{np.mean([np.mean(rec[c]) for c in range(7)]):>12.3f}       "
          f"{np.mean([np.mean(pre[c]) for c in range(7)]):>12.3f}       "
          f"{np.mean([np.mean(f1s[c]) for c in range(7)]):>12.3f}")


def main():
    argparse.ArgumentParser().parse_args()
    class_wise()


if __name__ == "__main__":
    main()
