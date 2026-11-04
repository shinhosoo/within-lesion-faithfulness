#!/usr/bin/env python3
"""Say, for every row of Supplementary Table S10, which archive produced it and under which code.

    python manifest.py # live root ./outputs, search from .
    python manifest.py --root outputs --tree .
"""
import argparse
import os
import re
import sys

import numpy as np

# (table label, stem). The first block of the table is the pooled reading, whose stems carry the
SPECS = [
    ("pooled, unsupervised", "ssf_mapsoftmax"),
    ("pooled, unsupervised", "lowrank_mapsoftmax"),
    ("pooled, unsupervised", "cam_mapsoftmax"),
    ("pooled, unsupervised", "mlp_mapsoftmax"),
    ("pooled, unsupervised", "cond_mapsoftmax"),
    ("pooled, supervised", "lowrank_sup0.2"),
    ("pooled, supervised", "cam_sup0.2"),
    ("pooled, supervised", "mlp_sup0.2"),
    ("pooled, supervised", "cond_sup0.2"),
    ("pooled, intervention", "s3_mlp_mapsoftmax"),
    ("pooled, intervention", "s3_mlp_sup0.2"),
    ("pooled, intervention", "s3_cond_mapsoftmax"),
    ("pooled, intervention", "s3_cond_sup0.2"),
    ("pooled, intervention", "gradgc3_mlp_sup0.2"),
    ("RECTIFIED", "lowrank"),
    ("RECTIFIED", "mlp"),
    ("RECTIFIED", "cond"),
    ("RECTIFIED", "s3_mlp"),
    ("RECTIFIED", "s3_cond"),
]
HEAD = ("point", "enrgy", "IoU", "InsDl", "withn")


def field(z, name):
    if name not in z.files:
        return "-"
    v = z[name]
    try:
        return v.item()
    except (ValueError, AttributeError):
        return v


def read(path):
    with np.load(path, allow_pickle=True) as z:
        R = z["metrics"]
        r = [float(np.nanmean(R[:, 0, c])) for c in range(3)]
        fh = z["faith_head"] if "faith_head" in z.files else None
        r.append(float(fh[2]) if fh is not None else float("nan"))
        r.append(float(fh[5]) if fh is not None and len(fh) > 5 else float("nan"))
        return r, (str(field(z, "select")), str(field(z, "source_sha1"))[:10])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs", help="the live output root the package cites")
    ap.add_argument("--tree", default=".", help="where to search for stray archives")
    a = ap.parse_args()
    live = os.path.normpath(os.path.join(a.root, "headswap"))

    stems = [s for _, s in SPECS]
    pat = re.compile(r"^hs_(" + "|".join(re.escape(s) for s in stems) + r")_f(\d+)\.npz$")
    byd = {}
    for dirpath, _, names in os.walk(a.tree):
        for n in names:
            m = pat.match(n)
            if m:
                byd.setdefault(m.group(1), {}).setdefault(
                    os.path.normpath(dirpath), []).append(os.path.join(dirpath, n))

    flags = []
    for block, stem in SPECS:
        dirs = byd.get(stem, {})
        inlive = live in dirs
        print(f"\n{stem}   [{block}]" + ("" if inlive else "   <-- NOT in the live root"))
        if not dirs:
            print("    no archive anywhere"); flags.append((stem, "absent")); continue
        seen = {}
        for d in sorted(dirs, key=lambda x: (x != live, x)):
            paths = sorted(dirs[d])
            rows, meta = [], set()
            for p in paths:
                r, mt = read(p)
                rows.append(r); meta.add(mt)
            arr = np.array(rows)
            mu = np.nanmean(arr, 0) if arr.size else arr
            key = tuple(None if np.isnan(v) else round(float(v), 3) for v in mu)
            seen.setdefault(key, []).append(d)
            tag = " (live)" if d == live else ""
            sel = ",".join(sorted({s for s, _ in meta}))
            sha = ",".join(sorted({h for _, h in meta}))
            print(f"    {os.path.relpath(d, a.tree):<42}{len(paths)}f  select {sel or '-':<5} "
                  f"src {sha}{tag}")
            print("      " + "  ".join(
                f"{h} {'   nan' if np.isnan(v) else format(v, '+.3f')}"
                for h, v in zip(HEAD, mu)))
        real = {k: v for k, v in seen.items() if not any(x is None for x in k)}
        if len(real) > 1:
            print(f"    !! {len(real)} DIFFERENT readings of this stem across directories")
            flags.append((stem, f"{len(real)} readings"))
        if not inlive:
            flags.append((stem, "not in live root"))

    print("\n" + "=" * 70)
    if flags:
        for stem, why in flags:
            print(f"  {stem:<24}{why}")
        print("\n  A stem with more than one reading, or none in the live root, cannot be\n"
              "  reproduced from the root the package cites. For the rectified block this is\n"
              "  also a claim about the paper: those rows are stated to be the same checkpoints\n"
              "  as the pooled rows, re-read, so a different code hash there needs explaining.")
    else:
        print("  every stem resolves to one reading in the live root.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
