"""Per-fold values behind the paired tests, fold counts and correlations in the text.

The tables print means and spreads over the five cross-validation folds. The statements in
the text that count folds or report a paired test need the fold values themselves, so this
writes them out. Run from the repository root:

    python code/collect_fold_level.py results/fold_level.json
"""
import glob
import json
import os
import sys

import numpy as np

ROOTS = {
    "main": "archives/outputs",
    "convnext_tiny": "archives/outputs_convnext_tiny",
    "nested": "archives/outputs_nested",
}
SEED_DIRS = {42: "headswap", 43: "headswap/seed43", 44: "headswap/seed44"}
# the two pooled rows sit outside the live output root
EXTRA = {42: ["headswap_s42_pooled"], 43: ["headswap_s43"], 44: ["headswap_s44"]}
COLS = ["pointing", "energy", "iou", "ins_del", "within"]


def head_rows(root):
    out = {}
    for seed, sub in SEED_DIRS.items():
        dirs = [os.path.join(root, sub)] + [os.path.join(root, d) for d in EXTRA.get(seed, [])]
        for d in dirs:
            for f in sorted(glob.glob(os.path.join(d, "hs_*_f*.npz"))):
                base = os.path.basename(f)[3:-4]
                stem, _, tail = base.rpartition("_f")
                if not tail.isdigit():
                    continue  # suffixed variants such as _f0_lim64 are not fold files
                fold = int(tail)
                z = np.load(f, allow_pickle=True)
                if "faith_head" not in z.files:
                    continue
                v = np.asarray(z["faith_head"], dtype=float).ravel()
                key = f"{stem}|seed{seed}"
                rec = out.setdefault(key, {"archive": os.path.relpath(d), "folds": {}})
                rec["folds"][fold] = {
                    "within": float(v[-1]),
                    "faith_head": [float(x) for x in v],
                    "params_total": int(z["params_total"]) if "params_total" in z.files else None,
                    "params_mechanism": int(z["params_mechanism"]) if "params_mechanism" in z.files else None,
                }
    return out


def ref_rows(root):
    out = {}
    for f in sorted(glob.glob(os.path.join(root, "headswap", "refs2_f*.npz"))):
        fold = int(os.path.basename(f).rsplit("_f", 1)[1].split(".")[0])
        z = np.load(f, allow_pickle=True)
        names = [str(x) for x in z["names"]]
        for j, n in enumerate(names):
            rec = out.setdefault(n, {"archive": os.path.relpath(os.path.dirname(f)), "folds": {}})
            rec["folds"][fold] = {
                "within": float(z["within"][j]),
                "whole_image": float(z["faith"][j]),
                "iou_subset": float(z["iou"][j]),
            }
    return out


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "results/fold_level.json"
    doc = {"_note": (
        "Per-fold values behind the paired tests, fold counts and correlations reported in the "
        "text. The manuscript tables print means and spreads over these. Head rows are keyed by "
        "archive stem and readout seed, reference rows by the name the refs2 archives use. "
        "Produced by code/collect_fold_level.py from the archives in this repository.")}
    for name, root in ROOTS.items():
        if not os.path.isdir(root):
            continue
        doc[name] = {"heads": head_rows(root), "references": ref_rows(root)}
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1, sort_keys=True)
    n = sum(len(v["heads"]) for k, v in doc.items() if k != "_note")
    print(f"[collect_fold_level] {n} head rows -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
