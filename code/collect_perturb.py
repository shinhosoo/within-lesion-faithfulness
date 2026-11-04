"""Pool the baseline-sensitivity runs over the folds that have been measured.

    python collect_perturb.py
    python collect_perturb.py --folds 0,1,2,3,4
    python collect_perturb.py --csv perturb_summary.csv

improve/perturb_variants.py writes one npz per fold, each holding the whole-image and
within-lesion scores of every map under every perturbation baseline. A single fold answers
whether the ordering of the maps moved on that fold. This reads the folds together and
reports the rank correlation fold by fold, with the mean and the spread across folds and the
number of folds in which the ordering held, which is the unit the rest of this study uses for
its comparisons.

Nothing here re-measures anything. If a fold is missing it is named and skipped, and the
summary says how many folds it rests on.
"""

import argparse
import json
import os

import numpy as np

# Reads only the saved npz, so it needs no GPU, no dataset and no model.
OUT = os.path.join(os.environ.get("HAM_ARCHIVE_ROOT", "archives/outputs"), "headswap")
PUBLISHED = "blur20"


def spearman(a, b):
    from scipy.stats import spearmanr

    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan"), float("nan")
    r, p = spearmanr(a[m], b[m])
    return float(r), float(p)


def load(fold, root):
    path = os.path.join(root, f"perturb_variants_f{fold}.npz")
    if not os.path.exists(path):
        return None, path
    d = np.load(path, allow_pickle=False)
    return d, path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=lambda s: [int(x) for x in s.split(",") if x],
                    default=[0, 1, 2, 3, 4])
    ap.add_argument("--root", default=OUT,
                    help="directory holding perturb_variants_f*.npz; the shipped copies are "
                         "under archives/outputs/headswap")
    ap.add_argument("--csv", default=None, help="also write the per-map means to this file")
    ap.add_argument("--json", default=None, help="also write the values here")
    a = ap.parse_args()

    runs, missing = {}, []
    for f in a.folds:
        d, path = load(f, a.root)
        if d is None:
            missing.append((f, path))
        else:
            runs[f] = d
    if missing:
        for f, path in missing:
            print(f"  [missing] fold {f}: {path}")
    if not runs:
        raise SystemExit("No runs found. Run improve.perturb_variants first.")

    folds = sorted(runs)
    ref = runs[folds[0]]
    names = [str(x) for x in ref["names"]]
    vb = [str(x) for x in ref["variant_baseline"]]
    vs = [int(x) for x in ref["variant_steps"]]
    vlab = [f"{b}, {s} steps" for b, s in zip(vb, vs)]

    # Every fold has to describe the same experiment, or pooling them means nothing.
    for f in folds[1:]:
        d = runs[f]
        if [str(x) for x in d["names"]] != names or \
                [str(x) for x in d["variant_baseline"]] != vb or \
                [int(x) for x in d["variant_steps"]] != vs:
            raise SystemExit(f"fold {f} measured a different set of maps or variants; "
                             "re-run it with the same --maps and --baselines.")
        if int(d["n"]) != int(ref["n"]):
            raise SystemExit(f"fold {f} used --n {int(d['n'])} against "
                             f"{int(ref['n'])} elsewhere.")

    try:
        base_i = vlab.index(f"{PUBLISHED}, {vs[0]} steps")
    except ValueError:
        base_i = 0
    print(f"[collect_perturb] {len(folds)} folds: {', '.join(str(f) for f in folds)}")
    print(f"[collect_perturb] {len(names)} maps, {len(vlab)} variants, "
          f"{int(ref['n'])} images per fold")
    print(f"[collect_perturb] ordering compared against {vlab[base_i]}")
    print()

    for title, key in (("whole-image", "whole"), ("within-lesion", "within")):
        print(f"    rank correlation of the map ordering, {title}")
        head = "".join(f"{'f' + str(f):>9}" for f in folds)
        print(f"    {'variant':<24}{head}{'mean':>10}{'s.d.':>9}{'held':>7}")
        for vi in range(len(vlab)):
            if vi == base_i:
                continue
            rs, ps = [], []
            for f in folds:
                r, p = spearman(runs[f][key][base_i], runs[f][key][vi])
                rs.append(r)
                ps.append(p)
            rs = np.array(rs)
            # "held" counts the folds in which the two orderings agree at the 0.05 level,
            # which is the same per-fold accounting the paper uses elsewhere.
            held = int(np.sum((np.array(ps) < 0.05) & (rs > 0)))
            row = "".join(f"{r:>+9.3f}" for r in rs)
            sd = f"{rs.std(ddof=1):>9.3f}" if len(rs) > 1 else f"{'':>9}"
            print(f"    {vlab[vi]:<24}{row}{rs.mean():>+10.3f}{sd}{held:>4}/{len(folds)}")
        print()

    for title, key in (("whole-image INS-DEL", "whole"), ("within-lesion INS-DEL", "within")):
        print(f"    {title}, mean over {len(folds)} folds")
        W = max(14, max(len(v) for v in vlab) + 2)
        print(f"    {'map':<30}" + "".join(f"{v:>{W}}" for v in vlab))
        for ni, nm in enumerate(names):
            vals = [np.mean([runs[f][key][vi, ni] for f in folds]) for vi in range(len(vlab))]
            print(f"    {nm:<30}" + "".join(f"{v:>+{W}.4f}" for v in vals))
        print()

    print(f"    IoU, mean over {len(folds)} folds, the same under every variant")
    for ni, nm in enumerate(names):
        print(f"    {nm:<30}{np.mean([runs[f]['iou'][ni] for f in folds]):>9.4f}")
    print()

    stamps = sorted({str(runs[f]["source_sha1"]) for f in folds})
    print(f"    source stamp: {', '.join(stamps)}")
    if len(stamps) > 1:
        print("    WARNING: the folds were produced by different versions of the code.")

    if a.json:
        os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
        rho = {}
        for key in ("whole", "within"):
            rho[key] = {}
            for vi, v in enumerate(vlab):
                if vi == base_i:
                    continue
                rs, ps = [], []
                for f in folds:
                    r, p_ = spearman(runs[f][key][base_i], runs[f][key][vi])
                    rs.append(r)
                    ps.append(p_)
                rs = np.array(rs)
                rho[key][v] = {"per_fold": [round(x, 4) for x in rs],
                               "mean": round(float(rs.mean()), 4),
                               "sd": round(float(rs.std(ddof=1)), 4) if len(rs) > 1 else None,
                               "folds_held": int(np.sum((np.array(ps) < 0.05) & (rs > 0))),
                               "folds": len(folds)}
        with open(a.json, "w") as fh:
            json.dump({"folds": folds, "maps": names, "variants": vlab,
                       "images_per_fold": int(ref["n"]), "reference": vlab[base_i],
                       "rank_correlation": rho,
                       "mean_over_folds": {
                           key: {nm: {v: round(float(np.mean(
                               [runs[f][key][vi, ni] for f in folds])), 4)
                               for vi, v in enumerate(vlab)}
                               for ni, nm in enumerate(names)}
                           for key in ("whole", "within")},
                       "source_sha1": sorted(stamps)}, fh, indent=1)
        print(f"    wrote {a.json}")

    if a.csv:
        import csv
        with open(a.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["map", "variant", "measure", "mean_over_folds", "n_folds"])
            for key, mlab in (("whole", "whole-image"), ("within", "within-lesion")):
                for ni, nm in enumerate(names):
                    for vi, v in enumerate(vlab):
                        m = np.mean([runs[f][key][vi, ni] for f in folds])
                        w.writerow([nm, v, mlab, f"{m:.4f}", len(folds)])
        print(f"    wrote {a.csv}")


if __name__ == "__main__":
    main()
