#!/usr/bin/env python3
"""Train the explanation head with the classifier's shortcut to the global feature removed.

    for f in 0 1 2 3 4; do python -m improve.ablate_macro --fold $f; done
    python -m improve.ablate_macro --compare
"""
import argparse
import glob
import os
import re
import sys
import types

import numpy as np

import improve.deep_head as dh
from improve.head_swap import OUT, build_head as _build_head

SUFFIX = "_nomacro"


def build_head_nomacro(name, net, s2_dim, final_dim, n_out, rank, signed=False,
                       map_mode="relu"):
    """build_head, with the macro term dropped from fuse when the name asks for it."""
    base = name[:-len(SUFFIX)] if name.endswith(SUFFIX) else name
    head = _build_head(base, net, s2_dim, final_dim, n_out, rank, signed, map_mode)
    if not name.endswith(SUFFIX):
        return head
    if not hasattr(head, "macro_proj"):
        raise SystemExit(f"{base} has no macro_proj; there is nothing to ablate")
    head.fuse = types.MethodType(
        lambda self, macro, pooled: self.norm_out(self.micro_vec(pooled)), head)
    for p in head.macro_proj.parameters():
        p.requires_grad_(False)
        p.detach().zero_()
    print(f"  [{base}{SUFFIX}] macro path removed: fuse reads the pooled vector alone, "
          f"and macro_proj is zeroed and frozen")
    return head


def row(paths):
    rs = []
    for p in paths:
        with np.load(p, allow_pickle=True) as z:
            R = z["metrics"]
            fh = z["faith_head"]
            rs.append([float(np.nanmean(R[:, 0, c])) for c in range(3)]
                      + [float(fh[2]), float(fh[5])])
    a = np.array(rs)
    sd = np.nanstd(a, 0, ddof=1) if len(rs) > 1 else np.zeros(a.shape[1])
    return np.nanmean(a, 0), sd, len(rs)


def archives(d, stem):
    pat = re.compile(r"^hs_" + re.escape(stem) + r"_f\d+\.npz$")
    return sorted(p for p in glob.glob(os.path.join(d, "hs_*.npz"))
                  if pat.fullmatch(os.path.basename(p)))


def compare(stage):
    d = OUT
    pairs = [("with the global feature (published)", f"s{stage}_mlp_mapsoftmax"),
             ("without it (this ablation)", f"s{stage}_mlp{SUFFIX}")]
    cols = ("pointing", "energy", "IoU", "Ins-Del", "within")
    print(f"{'':<36}" + "".join(f"{c:>13}" for c in cols))
    got = {}
    for label, stem in pairs:
        ps = archives(d, stem)
        if not ps:
            print(f"{label:<36}  no archive for {stem}")
            continue
        mu, sd, n = row(ps)
        got[stem] = mu
        print(f"{label:<36}" + "".join(f"{m:>+9.3f}±{s:.3f}" for m, s in zip(mu, sd))
              + f"   {n} folds")
    if len(got) == 2:
        a, b = got[f"s{stage}_mlp_mapsoftmax"], got[f"s{stage}_mlp{SUFFIX}"]
        print(f"{'difference':<36}" + "".join(f"{v:>+15.3f}" for v in (b - a)))
        print()
        if b[2] - a[2] > 0.02:
            print("  Overlap rises without the shortcut: the global-feature path is part of why")
            print("  the unsupervised map does not localise, and the paper can say so.")
        elif abs(b[2] - a[2]) <= 0.02:
            print("  Overlap is unchanged without the shortcut: removing the path the classifier")
            print("  could have used instead does not make the attention find the lesion, which")
            print("  strengthens the reading that localisation tracks the backbone.")
        else:
            print("  Overlap falls without the shortcut, which needs explaining before it is used.")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--stage", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--limit", type=int, default=0, help="smoke test on a few images per class")
    ap.add_argument("--faith-n", type=int, default=400)
    ap.add_argument("--faith-steps", type=int, default=10)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--compare", action="store_true",
                    help="print the ablation beside the published row and stop")
    a = ap.parse_args()
    if a.compare:
        return compare(a.stage)

    # The published unsupervised row is the pooled reading, so the ablation is measured the same
    a.head = "mlp" + SUFFIX
    a.map = "softmax"
    a.supervise = 0.0
    a.measure_only = False
    a.select = "att"
    dh.build_head = build_head_nomacro
    dh.run(a)
    print()
    print("  when every fold has run:  python -m improve.ablate_macro --compare")
    return 0


if __name__ == "__main__":
    sys.exit(main())
