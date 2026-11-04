#!/usr/bin/env python3
"""Seed-to-seed spread of a head's map metrics, against the retrain difference.

    python seed_spread.py --inventory outputs/headswap outputs/headswap_s4*
    python seed_spread.py --orig outputs/headswap/pre_retrain --dirs outputs/headswap outputs/headswap_s43 outputs/headswap_s44
"""
import argparse
import glob
import os
import re
import sys

import numpy as np

STEMS = ("lowrank", "ssf")
# The benchmark the paper already reports: the span of s3_mlp across its three readout seeds.
SMLP_POINTING_SPAN = 0.082

NAME = re.compile(r"^hs_(?P<stem>[a-z0-9_.]+?)_f(?P<fold>\d+)"
                  r"(?:_s(?P<seed>\d+))?\.npz$")


def find(dirs):
    """{(stem, seed): {fold: path}} over every directory given."""
    got = {}
    for d in dirs:
        for p in sorted(glob.glob(os.path.join(d, "hs_*.npz"))):
            m = NAME.match(os.path.basename(p))
            if not m:
                continue
            stem = m.group("stem")
            # A seed can be in the filename or in the directory name.
            seed = m.group("seed")
            if seed is None:
                dm = re.search(r"_s(\d+)$", os.path.basename(os.path.normpath(d)))
                seed = dm.group(1) if dm else "42"
            got.setdefault((stem, int(seed)), {})[int(m.group("fold"))] = p
    return got


def keys_of(path):
    with np.load(path, allow_pickle=True) as z:
        return sorted(z.files)


def metrics(path):
    """Every scalar the archive supports, as {name: value}. Absent keys are simply absent."""
    out = {}
    with np.load(path, allow_pickle=True) as z:
        if "metrics" in z:
            R = z["metrics"]
            R = R[:, 0, :] if R.ndim == 3 else R
            for i, name in enumerate(("pointing", "energy", "IoU")):
                if i < R.shape[1]:
                    out[name] = float(np.nanmean(R[:, i]))
            out["n"] = int(R.shape[0])
        for key, name in (("faith_head", "Ins-Del"),
                          ("faith_within", "within"),
                          ("faith_head_within", "within"),
                          ("within_head", "within")):
            if key in z and name not in out:
                out[name] = float(np.nanmean(z[key]))
        if "readout_val_acc" in z:
            v = float(z["readout_val_acc"])
            if np.isfinite(v) and v >= 0:
                out["val AUC"] = v
    return out


def mean_over_folds(paths):
    per = [metrics(p) for p in paths]
    names = [k for k in ("pointing", "energy", "IoU", "Ins-Del", "within", "val AUC")
             if any(k in m for m in per)]
    return {k: float(np.mean([m[k] for m in per if k in m])) for k in names}, len(per)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="*", default=["outputs/headswap"],
                    help="directories holding the retrained / seed archives")
    ap.add_argument("--orig", default="outputs/headswap/pre_retrain",
                    help="directory holding the archives the published table was built from")
    ap.add_argument("--inventory", nargs="*", default=None,
                    help="just list what is on disk, with each archive's keys, and stop")
    a = ap.parse_args()

    if a.inventory is not None:
        found = find(a.inventory or a.dirs)
        if not found:
            print("  nothing matched hs_*.npz in: " + ", ".join(a.inventory or a.dirs))
            return 1
        print(f"  {'stem':<14}{'seed':>6}{'folds':>7}   keys")
        for (stem, seed), folds in sorted(found.items()):
            ks = keys_of(folds[min(folds)])
            print(f"  {stem:<14}{seed:>6}{len(folds):>7}   {', '.join(ks)}")
            print(f"  {'':14}{'':6}{'':7}   folds present: "
                  f"{sorted(folds)}")
        return 0

    new = find(a.dirs)
    orig = find([a.orig]) if os.path.isdir(a.orig) else {}

    for stem in STEMS:
        seeds = sorted(s for (st, s) in new if st == stem)
        if not seeds:
            print(f"\n  {stem}: no archive found in {', '.join(a.dirs)}")
            continue
        print(f"\n  {'=' * 72}\n  {stem}")
        rows = {}
        for s in seeds:
            folds = new[(stem, s)]
            rows[s], n = mean_over_folds([folds[f] for f in sorted(folds)])
            rows[s]["_folds"] = n
        base = None
        if (stem, 42) in orig:
            f = orig[(stem, 42)]
            base, _ = mean_over_folds([f[k] for k in sorted(f)])

        names = [k for k in ("pointing", "energy", "IoU", "Ins-Del", "within", "val AUC")
                 if any(k in r for r in rows.values()) or (base and k in base)]
        hdr = "  " + f"{'measure':<10}" + "".join(f"{'seed ' + str(s):>12}" for s in seeds)
        if base:
            hdr += f"{'published':>12}{'span':>10}{'retrain d':>12}"
        else:
            hdr += f"{'span':>10}"
        print(hdr)
        verdict = {}
        for k in names:
            vals = [rows[s].get(k) for s in seeds]
            line = f"  {k:<10}" + "".join(
                f"{v:>12.4f}" if v is not None else f"{'--':>12}" for v in vals)
            have = [v for v in vals if v is not None]
            span = (max(have) - min(have)) if len(have) > 1 else None
            if base:
                b = base.get(k)
                line += f"{b:>12.4f}" if b is not None else f"{'--':>12}"
                line += f"{span:>10.4f}" if span is not None else f"{'--':>10}"
                # the retrain difference is seed 42 re-run minus the published seed 42
                d = (rows[42].get(k) - b) if (42 in rows and b is not None
                                              and rows[42].get(k) is not None) else None
                line += f"{d:>+12.4f}" if d is not None else f"{'--':>12}"
                if span is not None and d is not None:
                    verdict[k] = (span, abs(d))
            else:
                line += f"{span:>10.4f}" if span is not None else f"{'--':>10}"
            print(line)
        print(f"  {'folds':<10}" + "".join(f"{rows[s]['_folds']:>12d}" for s in seeds))

        if verdict:
            print()
            noise = [k for k, (sp, d) in verdict.items() if d <= sp]
            outside = [k for k, (sp, d) in verdict.items() if d > sp]
            for k, (sp, d) in sorted(verdict.items()):
                tag = "within the seed span" if d <= sp else "OUTSIDE the seed span"
                print(f"    {k:<10} retrain moved {d:.4f}, seeds span {sp:.4f}  -> {tag}")
            if outside and not noise:
                print(f"\n    Every measure moved further than the seeds do. The published row is")
                print(f"    not reproducible at its own seed, and that is the finding: it cannot be")
                print(f"    swapped for a new number.")
            elif noise and not outside:
                print(f"\n    Every measure moved less than the seeds do, so the retrain difference")
                print(f"    is training noise. The row has to be reported as a spread over seeds")
                print(f"    rather than as one run, and the claims that rest on its level need the")
                print(f"    spread attached.")
            else:
                print(f"\n    Mixed: {', '.join(outside)} moved further than the seeds do while "
                      f"{', '.join(noise)} did not.")
                print(f"    Neither reading covers it; report the per-seed values and say so.")
        if 'pointing' in verdict:
            sp = verdict['pointing'][0]
            print(f"\n    For scale, s3_mlp's pointing span over its three readout seeds is "
                  f"{SMLP_POINTING_SPAN:.3f};")
            print(f"    {stem}'s is {sp:.3f} over {len(seeds)} seeds.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
