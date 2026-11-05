"""The centered-Gaussian reference as a function of its width, Supplementary Table S7.

The table asks a narrow question: the reference the trained maps are compared against is a
Gaussian of one fixed width, so how much of its advantage comes from that choice. Answering
it needs no classifier and no GPU. The map is a function of the grid alone, and the three
overlap measures compare it with the lesion mask, so the only inputs are the fixed splits
and the masks.

The published width is the one improve.localize.center_map builds, and this script refuses
to run unless its own builder reproduces that array exactly at that width. Two different
centre conventions are in the code base, half a pixel apart, and the wrong one would move
the energy ratio while leaving the other two measures untouched, which is precisely the
pattern the table reports and so would not look wrong.

Scoring is improve.localize.metrics, unchanged, the same function every other overlap
number in this work is produced by. The pointing score and the energy ratio are defined for
every mask. The area-matched IoU is not defined where the lesion covers none or all of the
frame, metrics returns nan there, and those images drop out of that column alone, which is
why the table reports a smaller count for it.

    python -m improve.sigma_sweep
    python -m improve.sigma_sweep --widths 0.25 --folds 0
    python -m improve.sigma_sweep --out ~/sigma_sweep.json

Nothing is written beside the published runs: --out refuses a path inside the project, the
splits or the data.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())

from SkinCancer.configs.model_config import IMG_SIZE                      # noqa: E402
from SkinCancer.data.split_io import build_image_path_map                 # noqa: E402
from improve.head_swap import MASK_DIR                                    # noqa: E402
from improve.localize import center_map, metrics                          # noqa: E402
from improve.residual import eval_split                                   # noqa: E402

WIDTHS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)
PUBLISHED_WIDTH = 0.25
# Supplementary Table S7's row for the published width, which is also the centered Gaussian
# of Table 1 and of Supplementary Table S10. Reproducing it is the check that the map, the
# image set and the scoring are the published ones; everything else in the table is the same
# computation with one number changed.
PUBLISHED_ROW = {"pointing": 0.9655, "energy": 0.4688, "iou": 0.655671}


def gaussian(frac):
    """The centered Gaussian at a width of frac of the image side.

    improve.localize.center_map is this function at PUBLISHED_WIDTH. It is not called with a
    width because it does not take one, so the formula is repeated here and checked against
    it rather than trusted.
    """
    yy, xx = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]
    c, s = (IMG_SIZE - 1) / 2, IMG_SIZE * frac
    return np.exp(-((xx - c) ** 2 + (yy - c) ** 2) / (2 * s ** 2))


def check_builder():
    mine, theirs = gaussian(PUBLISHED_WIDTH), center_map()
    if mine.shape != theirs.shape or not np.array_equal(mine, theirs):
        d = float(np.abs(mine - theirs).max()) if mine.shape == theirs.shape else float("nan")
        sys.exit(f"the width-{PUBLISHED_WIDTH} map does not equal improve.localize."
                 f"center_map(), largest difference {d:.3e}. The sweep would not be of the "
                 f"published reference, so nothing is measured.")
    print(f"  the builder reproduces improve.localize.center_map() at width "
          f"{PUBLISHED_WIDTH} exactly")


def masks(folds):
    """Every held-out mask of the given folds, in fold and row order, one at a time.

    Yielded rather than returned as a list: eight thousand masks at the model's resolution
    are over a gigabyte even as booleans, and nothing here needs more than one at once.
    """
    from PIL import Image
    pm = build_image_path_map()
    missing = []
    for fold in folds:
        ev = eval_split(fold, pm)
        for iid in ev["image_id"]:
            p = os.path.join(MASK_DIR, f"{iid}_segmentation.png")
            if not os.path.exists(p):
                missing.append(iid)
                continue
            yield np.array(Image.open(p).convert("L")
                           .resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)) > 127
    if missing:
        sys.exit(f"{len(missing)} mask(s) not found under {MASK_DIR}, the first being "
                 f"{missing[0]}. The table is over every held-out mask, so a partial set "
                 f"would not reproduce it.")


def refuse_inside(path):
    """The published outputs are not this script's to touch."""
    ap = os.path.abspath(path)
    roots = [os.path.abspath(os.getcwd())]
    for v in ("HAM_OUTPUT_ROOT", "HAM_SPLIT_ROOT", "HAM_DATA_ROOT", "HAM_MASK_DIR"):
        q = os.environ.get(v)
        if q:
            roots.append(os.path.abspath(q))
    for r in roots:
        if ap == r or ap.startswith(r + os.sep):
            sys.exit(f"--out {path} is inside {r}, where this script writes nothing")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", default=",".join(f"{w:g}" for w in WIDTHS),
                    help="Gaussian widths as a fraction of the image side")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--out", default="", help="where to write the values as JSON")
    a = ap.parse_args()
    if a.out:
        refuse_inside(a.out)
    widths = [float(x) for x in a.widths.split(",") if x.strip()]
    folds = [int(x) for x in a.folds.split(",") if x.strip()]

    check_builder()
    maps = np.stack([gaussian(w) for w in widths])
    rows, usable = [], 0
    for n, m in enumerate(masks(folds), 1):
        usable += 0 < m.mean() < 1
        rows.append(metrics(maps, m))
        if n % 1000 == 0:
            print(f"  {n} masks scored")
    if not rows:
        sys.exit("no masks were read, so nothing was measured")
    rows = np.stack(rows)                                 # [images, widths, 3]
    print(f"  {len(rows)} held-out masks over folds {folds}, {usable} with a usable "
          f"lesion area")

    print(f"\n  {'width':>7}{'px':>8}{'Pointing':>12}{'Energy':>10}{'IoU':>12}{'n IoU':>8}")
    doc = {"image_size": IMG_SIZE, "folds": folds, "images": len(rows),
           "images_with_usable_area": int(usable), "published_width": PUBLISHED_WIDTH,
           "scored_by": "improve.localize.metrics", "widths": {}}
    for i, w in enumerate(widths):
        p, e, u = (float(np.nanmean(rows[:, i, c])) for c in range(3))
        n_iou = int(np.sum(~np.isnan(rows[:, i, 2])))
        print(f"  {w:>7.2f}{IMG_SIZE * w:>8.1f}{p:>12.4f}{e:>10.4f}{u:>12.6f}{n_iou:>8}")
        doc["widths"][f"{w:g}"] = {"sigma_px": IMG_SIZE * w, "pointing": p,
                                   "energy": e, "iou": u, "n_iou": n_iou}

    if PUBLISHED_WIDTH in widths:
        got = doc["widths"][f"{PUBLISHED_WIDTH:g}"]
        bad = []
        for k, want in PUBLISHED_ROW.items():
            # Compared at the precision the table prints, which is the precision the claim
            # is made at.
            dec = len(str(want).split(".")[1])
            if abs(round(got[k], dec) - want) > 10 ** -dec * 0.501:
                bad.append(f"{k} {got[k]:.6f} rounds to {got[k]:.{dec}f}, published {want}")
        if bad:
            sys.exit("\n  the published width does not reproduce its published row:\n    "
                     + "\n    ".join(bad))
        print(f"\n  the width-{PUBLISHED_WIDTH} row reproduces Supplementary Table S7 and the "
              f"centered Gaussian of Table 1")
    else:
        print(f"\n  note: width {PUBLISHED_WIDTH} was not measured, so the published row was "
              f"not checked")

    if a.out:
        doc["source_sha1"] = hashlib.sha1(
            open(os.path.abspath(__file__), "rb").read()).hexdigest()[:10]
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=1)
            fh.write("\n")
        print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
