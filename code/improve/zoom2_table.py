"""Margin sweep summary results (read-only)."""
import glob
import os

import numpy as np
from scipy.stats import binomtest

from SkinCancer.configs.training_config import OUTPUT_ROOT

D = os.path.join(OUTPUT_ROOT, "zoom")
MARGINS = [0, 20, 40, 60, 80, 120]


def macro_f1(y, p, n=7):
    f = []
    for c in range(n):
        tp = ((p == c) & (y == c)).sum(); fp = ((p == c) & (y != c)).sum(); fn = ((p != c) & (y == c)).sum()
        f.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(f))


def main():
    fs = sorted(glob.glob(os.path.join(D, "zoom2_f*_s*.npz")))
    if not fs:
        print("no results yet"); return
    parts = [np.load(f, allow_pickle=True) for f in fs]
    y = np.concatenate([p["y_true"] for p in parts]).astype(int)
    get = lambda k: np.concatenate([p[k] for p in parts])
    b = get("base").argmax(1)
    print("=" * 84)
    print(f" Zoom margin sweep ({len(fs)} files, n={len(y)}, inference only, averaged with base)")
    print("=" * 84)
    print(f"  {'box / margin':<16}" + "".join(f"{str(m)+'%':>10}" for m in MARGINS))
    for src, label in [("mask", "mask box")]:
        accs, f1s = [], []
        for m in MARGINS:
            k = f"{src}_m{m}"
            if k not in parts[0].files:
                accs.append(np.nan); f1s.append(np.nan); continue
            p = get(k).argmax(1)
            accs.append((p == y).mean()); f1s.append(macro_f1(y, p))
        print(f"  {label+' Acc':<16}" + "".join(f"{v:>10.4f}" for v in accs))
        print(f"  {'   vs base':<16}" + "".join(f"{v-(b==y).mean():>+10.4f}" for v in accs))
        print(f"  {label+' F1':<16}" + "".join(f"{v:>10.4f}" for v in f1s))
    print(f"\n  baseline (base): Acc {(b==y).mean():.4f}  F1 {macro_f1(y,b):.4f}")
    best = None
    for src in ("mask",):
        for m in MARGINS:
            k = f"{src}_m{m}"
            if k not in parts[0].files:
                continue
            p = get(k).argmax(1); acc = (p == y).mean()
            if best is None or acc > best[1]:
                best = (k, acc, p)
    if best:
        k, acc, p = best
        n10 = int(((b == y) & (p != y)).sum()); n01 = int(((b != y) & (p == y)).sum())
        pv = binomtest(n10, n10 + n01, .5).pvalue if n10 + n01 else 1.0
        print(f"  best: {k}  Acc {acc:.4f} ({acc-(b==y).mean():+.4f})  "
              f"base only {n10} / {k} only {n01}  p={pv:.3g}")
    print("\n  how to read: if larger margins keep helping, background matters; the knee is the best point,")
    print("           and that value sets the range for the lcrop training augmentation.")


if __name__ == "__main__":
    main()
