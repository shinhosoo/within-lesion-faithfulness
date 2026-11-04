#!/usr/bin/env python3
"""Per-fold pooled values for the intervention rows, so the Discussion can be re-derived.

    python collect_interventions.py
    python collect_interventions.py --json interventions.json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = "outputs/headswap"

# (key, the archive stem to read, what it is in the table)
WANT = [
    ("mlp",            "hs_mlp_mapsoftmax",            "Position-wise MLP, second stage"),
    ("s3_mlp",         "hs_s3_mlp_mapsoftmax",         "MLP at the third stage"),
    ("cond",           "hs_cond_mapsoftmax",           "Image-conditional, second stage"),
    ("s3_cond",        "hs_s3_cond_mapsoftmax",        "Conditional at the third stage"),
    ("s3_cond_sup",    "hs_s3_cond_sup0.2",            "Conditional at the third stage, sup."),
    ("gradgc3_mlp",    "hs_gradgc3_mlp_sup0.2",        "MLP, gradient target"),
    ("mlp_sup",        "hs_mlp_sup0.2",                "Position-wise MLP, supervised"),
    ("s3_mlp_sup",     "hs_s3_mlp_sup0.2",             "MLP at the third stage, sup."),
]
COLS = ("point", "energy", "iou", "insdel", "within")


def read(path):
    with np.load(path, allow_pickle=True) as z:
        R = z["metrics"][:, 0, :]
        v = np.asarray(z["faith_head"], dtype=float).ravel()
        if v.size != 6 or abs((v[1] - v[0]) - v[2]) > 1e-6 or abs((v[4] - v[3]) - v[5]) > 1e-6:
            return None, "faith_head does not satisfy its own arithmetic"
        d = dict(zip(COLS, (float(np.nanmean(R[:, 0])), float(np.nanmean(R[:, 1])),
                            float(np.nanmean(R[:, 2])), float(v[2]), float(v[5]))))
        d["sha"] = str(z["source_sha1"]) if "source_sha1" in z.files else "?"
        for k in ("params_mechanism", "mech_params", "params_total"):
            if k in z.files:
                d["mech"] = int(np.asarray(z[k]).ravel()[0])
                break
        else:
            d["mech"] = None
        return d, None


def sd1(v):
    m = sum(v) / len(v)
    return (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    out, missing = {}, []
    print(f"  reading seed 42 from {a.root}\n")
    for key, stem, label in WANT:
        paths = sorted(glob.glob(os.path.join(a.root, f"{stem}_f[0-4].npz")))
        if len(paths) != 5:
            print(f"  {label:<38}{len(paths)}/5 found   ({stem}_f*.npz)")
            near = sorted({os.path.basename(p) for p in
                           glob.glob(os.path.join(a.root, f"*{key}*.npz"))})[:6]
            if near:
                print(f"  {'':<38}present: {', '.join(near)}")
            missing.append(label)
            continue
        rows, err = [], None
        for p in paths:
            r, e = read(p)
            if e:
                err = f"{os.path.basename(p)}: {e}"
                break
            rows.append(r)
        if err:
            print(f"  {label:<38}{err}")
            missing.append(label)
            continue
        shas = sorted({r["sha"] for r in rows})
        out[key] = {"label": label, "stem": stem, "sha": "/".join(shas),
                    "mech": rows[0]["mech"],
                    **{c: [round(r[c], 6) for r in rows] for c in COLS}}
        print(f"  {label:<38}5/5   {shas[0] if len(shas)==1 else shas}")
        if len(shas) != 1:
            print(f"  {'':<38}WARNING: folds span more than one source")

    print(f"\n  {'row':<38}{'pointing':>16}{'energy':>16}{'IoU':>16}{'within':>18}")
    for key, v in out.items():
        cells = "".join(f"{sum(v[c])/5:>+11.3f} ± {sd1(v[c]):.3f}"
                        for c in ("point", "energy", "iou"))
        w = v["within"]
        print(f"  {v['label']:<38}{cells}{sum(w)/5:>+12.4f} ± {sd1(w):.4f}")

    print("\n  per-fold values, for the paired tests")
    for key, v in out.items():
        print(f"  {key} {v['sha']} {v['mech']}")
        for c in COLS:
            print("    " + c + " " + " ".join(f"{x:.6f}" for x in v[c]))

    if missing:
        print(f"\n  {len(missing)} row(s) not found: {', '.join(missing)}")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1, sort_keys=True)
        print(f"\n  written: {a.json}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
