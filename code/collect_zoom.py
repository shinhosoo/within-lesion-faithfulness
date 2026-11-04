"""The inference-time magnification rows of Supplementary Table S15.

    python code/collect_zoom.py
    python code/collect_zoom.py --json results/zoom_inference.json

Reads only the archives, so it needs no GPU, no dataset and no model.

Supplementary Table S4 restricts the classifier to the lesion during training and reports
what that costs. This is the other half of that comparison: the same frozen classifier,
not retrained, given the lesion's bounding box at inference with a margin around it. The
accuracy barely moves, which is what separates "training under a restriction hurts" from
"the background carries information the classifier uses".

Accuracy is the out-of-fold total over the five folds, so each image counts once and the
figure is comparable with Supplementary Tables S3 and S4 rather than being an average of
fold averages. The per-fold spread is reported beside it for reference only.

The archives also hold a second box drawn from an attention map rather than from the mask.
It is not read here and not shipped, because the mechanism that produced it is not one this
manuscript describes, and the figure the text quotes comes from the mask box.
"""

import argparse
import glob
import json
import os

import numpy as np

MARGINS = (0, 20, 40, 60, 80, 120)
LABEL = {"base": "Unrestricted (reference)"}
for _m in MARGINS:
    LABEL[f"mask_m{_m}"] = f"Lesion box, margin {_m}\\%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(
                        os.environ.get("HAM_ARCHIVE_ROOT", "archives/outputs"), "zoom"),
                    help="directory holding zoom2_f*_s42.npz; the shipped copies are under "
                         "archives/outputs/zoom")
    ap.add_argument("--json", default=None, help="also write the values here")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.root, "zoom2_f*_s42.npz")))
    if not files:
        raise SystemExit(f"No zoom2_f*_s42.npz under {a.root!r}. "
                         "Point --root at the directory that holds them.")
    conds = ["base"] + [f"mask_m{m}" for m in MARGINS]
    hit = {c: 0 for c in conds}
    tot = {c: 0 for c in conds}
    per = {c: [] for c in conds}
    for f in files:
        d = np.load(f, allow_pickle=True)
        missing = [c for c in conds if c not in d.files]
        if missing:
            raise SystemExit(f"{f} has no {', '.join(missing)}. This file is not one of the "
                             "magnification runs.")
        y = d["y_true"]
        for c in conds:
            ok = int((d[c].argmax(1) == y).sum())
            hit[c] += ok
            tot[c] += len(y)
            per[c].append(ok / len(y))
    if len(set(tot.values())) != 1:
        raise SystemExit("the conditions do not cover the same images")

    acc = {c: hit[c] / tot[c] for c in conds}
    base = acc["base"]
    print(f"[collect_zoom] {len(files)} folds, {tot['base']} images, inference only")
    print(f"    {'Input at inference':<28}{'Accuracy':>10}{'Delta (pp)':>12}{'fold s.d.':>11}")
    for c in conds:
        d_pp = "" if c == "base" else f"{(acc[c] - base) * 100:>+12.2f}"
        print(f"    {LABEL[c].replace(chr(92), ''):<28}{acc[c]:>10.4f}{d_pp:>12}"
              f"{np.std(per[c], ddof=1):>11.4f}")
    worst = max(abs(acc[c] - base) for c in conds if c != "base")
    print()
    print(f"    largest change from the unrestricted input: {worst * 100:.2f} pp")

    if a.json:
        os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
        with open(a.json, "w") as fh:
            json.dump({"folds": len(files), "images": tot["base"],
                       "accuracy": {c: round(acc[c], 6) for c in conds},
                       "delta_pp": {c: round((acc[c] - base) * 100, 4)
                                    for c in conds if c != "base"},
                       "fold_sd": {c: round(float(np.std(per[c], ddof=1)), 6) for c in conds},
                       "largest_change_pp": round(worst * 100, 4),
                       "files": [os.path.basename(f) for f in files]}, fh, indent=1)
        print(f"    wrote {a.json}")


if __name__ == "__main__":
    main()
