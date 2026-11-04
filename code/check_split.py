"""Phase 0 split check, run before any training starts.

    python check_split.py --split-root ./splits_paper_seed42 --data-root ./data/HAM10000_extracted
"""

import argparse
import glob
import itertools
import os
import sys

import pandas as pd

FINAL7 = ["nv", "mel", "bkl", "df", "vasc", "bcc", "akiec"]
PAPER_TEST = {"nv": 1336, "mel": 236, "bkl": 243, "df": 18, "vasc": 34, "bcc": 90, "akiec": 67}

def load(split_root, prefix, subset):
    for name in (f"{prefix}_{subset}.csv",):
        p = os.path.join(split_root, name)
        if os.path.exists(p):
            df = pd.read_csv(p)
            if "diagnosis" not in df.columns and "dx" in df.columns:
                df = df.rename(columns={"dx": "diagnosis"})
            df["diagnosis"] = df["diagnosis"].str.lower()
            return df
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-root", default=os.environ.get("HAM_SPLIT_ROOT", "./splits"))
    ap.add_argument("--data-root", default=os.environ.get("HAM_DATA_ROOT"))
    args = ap.parse_args()

    ok = True
    print(f"split_root = {os.path.abspath(args.split_root)}\n")

    # 1. CSV files present
    print("[1] CSV files present")
    g = {}
    for subset in ["train", "val", "test"]:
        g[subset] = load(args.split_root, "global", subset)
        if g[subset] is None:
            print(f"    {subset:<6} missing  <- global_{subset}.csv")
            ok = False
    if any(v is None for v in g.values()):
        sys.exit(1)

    # 2. subset sizes
    tot_i = sum(len(v) for v in g.values())
    tot_l = pd.concat(g.values()).lesion_id.nunique() if "lesion_id" in g["train"].columns else None
    print("\n[2] Sizes")
    print(f"    {'subset':<7}{'images':>8}{'%':>9}{'lesions':>9}")
    for s in ["train", "val", "test"]:
        L = g[s].lesion_id.nunique() if tot_l else 0
        print(f"    {s:<7}{len(g[s]):>8}{100*len(g[s])/tot_i:>8.2f}%{L:>9}")
    print(f"    {'total':<7}{tot_i:>8}{'':>9}{tot_l if tot_l else '-':>9}")
    if tot_i != 10015:
        print(f"    warning: total is not 10,015 images ({tot_i})")
        ok = False

    # 3. leakage
    print("\n[3] Leakage check")
    ids = pd.concat(g.values()).image_id
    dup = int(ids.duplicated().sum())
    print(f"    duplicate image_id: {dup}{'' if dup == 0 else '   <-- problem'}")
    ok &= dup == 0
    if tot_l:
        for a, b in itertools.combinations(["train", "val", "test"], 2):
            n = len(set(g[a].lesion_id) & set(g[b].lesion_id))
            print(f"    lesion_id overlap {a}/{b}: {n}{'' if n == 0 else '   <-- leakage'}")
            ok &= n == 0
    else:
        print("    no lesion_id column, leakage check skipped  <-- needs checking")
        ok = False

    # 4. class counts of the test split
    print("\n[4] test split class counts")
    tc = g["test"].diagnosis.value_counts()
    print(f"    {'class':<7}{'CSV':>7}{'paper':>7}{'diff':>6}")
    hit = True
    for c in FINAL7:
        n = int(tc.get(c, 0))
        d = n - PAPER_TEST[c]
        hit &= d == 0
        print(f"    {c.upper():<7}{n:>7}{PAPER_TEST[c]:>7}{d:>+6}")
    print(f"    {'total':<7}{len(g['test']):>7}{sum(PAPER_TEST.values()):>7}"
          f"{len(g['test'])-sum(PAPER_TEST.values()):>+6}")
    print(f"    => {'match (same split as the paper experiment)' if hit else 'mismatch (different split)'}")
    ok &= hit

    # 4b. how far the class mix drifts between the three subsets. The split is grouped by
    print("\n[4b] class share by subset")
    print(f"    {'class':<7}{'train':>9}{'val':>9}{'test':>9}{'spread':>9}")
    worst, worst_c = 0.0, None
    for c in FINAL7:
        sh = [100 * float((g[s].diagnosis == c).mean()) for s in ["train", "val", "test"]]
        spread = max(sh) - min(sh)
        if spread > worst:
            worst, worst_c = spread, c
        print(f"    {c.upper():<7}{sh[0]:>8.1f}%{sh[1]:>8.1f}%{sh[2]:>8.1f}%{spread:>8.2f}")
    print(f"    => largest spread {worst:.2f} pp, in {worst_c.upper()}")

    # 5. image files present
    print("\n[5] Image files present")
    if not args.data_root:
        print("    --data-root not given, skipped")
    else:
        have, dirs = set(), []
        if os.path.isdir(args.data_root):
            entries = {n.lower(): n for n in os.listdir(args.data_root)}
            for part in ("ham10000_images_part_1", "ham10000_images_part_2"):
                if part in entries:
                    dirs.append(os.path.join(args.data_root, entries[part]))
        if not dirs:
            print(f"    warning: under {args.data_root} no part_1 / part_2 folder was found")
            ok = False
        for d in dirs:
            n = {os.path.splitext(os.path.basename(p))[0]
                 for p in glob.glob(os.path.join(d, "*.jpg"))}
            print(f"    {os.path.basename(d)}: {len(n)} images")
            have |= n
        print(f"    images on disk: {len(have)} total")
        for s in ["train", "val", "test"]:
            miss = len(set(g[s].image_id) - have)
            print(f"    {s:<6} missing {miss} images{'' if miss == 0 else '   <-- dropped during training'}")
            ok &= miss == 0

    print("\n" + ("=" * 46))
    print("  PHASE 0 PASSED — ready to proceed to Phase 1" if ok else "  PHASE 0 FAILED — fix the items above and rerun")
    print("=" * 46)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
