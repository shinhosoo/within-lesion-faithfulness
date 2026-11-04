"""Model comparison over 7,991 5-fold out-of-fold predictions. test unused."""
import argparse, os
import numpy as np
from scipy.stats import binomtest
from SkinCancer.configs.training_config import OUTPUT_ROOT
from SkinCancer.eval.metrics import calculate_detailed_multiclass_metrics

OUT = os.path.join(OUTPUT_ROOT, "residual")
F = ["nv", "mel", "bkl", "df", "vasc", "bcc", "akiec"]
K = ["Accuracy", "Sensitivity", "F1", "AUC", "PPV"]


def oof(model, seed):
    P, Y, I, fold = [], [], [], []
    for k in range(5):
        f = os.path.join(OUT, f"{model}_f{k}_s{seed}_val_predictions.npz")
        if not os.path.exists(f):
            return None, k
        d = np.load(f, allow_pickle=True)
        P.append(d["y_prob"]); Y.append(d["y_true"]); I.append(d["image_id"]); fold += [k] * len(d["y_true"])
    I = np.concatenate(I); o = np.argsort(I)
    return (np.concatenate(P)[o], np.concatenate(Y)[o].astype(int), I[o], np.array(fold)[o]), 5


def stats(y, P):
    o, _ = calculate_detailed_multiclass_metrics(y, P, F); p = P.argmax(1)
    return [o[k] for k in K] + [100 * (p[y == 1] == 1).mean()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["flat"])
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    R = {}
    for m in a.models:
        r, n = oof(m, a.seed)
        if r is None:
            print(f"  [{m}] fold {n} has no predictions - still running or failed"); continue
        R[m] = r
    if not R:
        return
    ref = next(iter(R.values()))
    for m, r in R.items():
        assert np.array_equal(r[2], ref[2]), f"{m} sample mismatch"
    y, folds = ref[1], ref[3]
    print("=" * 84); print(f" 5-fold cross-validated out-of-fold results  (n={len(y)}, test unused)"); print("=" * 84)
    print(f"  {'model':<8}" + "".join(f"{k[:6]:>10}" for k in K) + f"{'MELrec':>10}{'fold Acc range':>18}")
    for m, (P, *_) in R.items():
        s = stats(y, P)
        fa = [(P[folds == k].argmax(1) == y[folds == k]).mean() for k in range(5)]
        print(f"  {m:<8}" + "".join(f"{v:>10.4f}" for v in s[:5]) + f"{s[5]:>9.1f}%   {min(fa):.4f}~{max(fa):.4f}")
    if "flat" in R:
        pf = R["flat"][0].argmax(1)
        print(f"\n  paired McNemar vs flat (1-stage)   n01 > n10 means that model wins")
        for m, (P, *_) in R.items():
            if m == "flat":
                continue
            pm = P.argmax(1); ca, cb = pf == y, pm == y
            n10, n01 = int((ca & ~cb).sum()), int((~ca & cb).sum())
            p = binomtest(n10, n10 + n01, .5).pvalue if n10 + n01 else 1.0
            wins = sum((P[folds == k].argmax(1) == y[folds == k]).mean()
                       > (R["flat"][0][folds == k].argmax(1) == y[folds == k]).mean() for k in range(5))
            print(f"    {m:<6} flat only correct {n10:>4} / {m} only correct {n01:>4}   p={p:.4g}   folds won {wins}/5")

if __name__ == "__main__":
    main()
