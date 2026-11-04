#!/usr/bin/env python3
"""Regenerate Figure 2 from the explanation head's fold archives.

    python3 make_fig3.py --out outputs/headswap
    python3 make_fig3.py --out outputs/headswap --fig figure_3.png --json fig3_values.json
"""

import argparse
import json
import os
import sys

import numpy as np

COLS = (("pointing", 0), ("energy", 1), ("IoU", 2))
MARKERS = {"pointing": "o", "energy": "s", "IoU": "^"}

# Row label as the manuscript prints it -> how the value is found. A ("head", stem) row is read
UNSUP_LAB = "Explanation head\n(unsupervised)"
SUP_LAB = "Explanation head\n+ mask supervision"

# The unsupervised stem is a flag, because the same checkpoints can be read two ways and the
UNSUP_STEM = "s3_mlp_mapsoftmax"

ROWS = [
    (UNSUP_LAB, ("head", UNSUP_STEM)),
    (SUP_LAB, ("head", "s3_mlp_sup0.2")),
    ("Centered Gaussian", ("ref", "center")),
    ("Grad-CAM\n(final stage)", ("ref", "gc4")),
    ("Grad-CAM\n(stage 2)", ("ref", "gc2")),
]

SEEDS = (42, 43, 44)
FOLDS = range(5)


def seed_dir(out, seed):
    """Where `seed_out()` puts a seed's archives: the root for 42, a subdirectory otherwise."""
    return out if int(seed) == 42 else os.path.join(out, f"seed{int(seed)}")


def head_units(out, stem):
    """Every fold-by-seed unit of one stem: the overlap triple, Ins-Del, and the image counts."""
    units, missing = [], []
    for seed in SEEDS:
        for fold in FOLDS:
            p = os.path.join(seed_dir(out, seed), f"hs_{stem}_f{fold}.npz")
            if not os.path.exists(p):
                missing.append(p)
                continue
            with np.load(p, allow_pickle=True) as z:
                R = z["metrics"]
                u = {name: float(np.nanmean(R[:, 0, c])) for name, c in COLS}
                u["Ins-Del"] = float(z["faith_head"][2])
                u["n overlap"] = int(R.shape[0])
                u["n faith"] = int(z["faith_n"]) if "faith_n" in z.files else 0
                # the centered Gaussian as this archive measured it, for the cross-check below
                u["_center"] = {name: float(np.nanmean(R[:, 1, c])) for name, c in COLS}
                u["_center"]["Ins-Del"] = float(z["faith_center"][2])
                u["_seed"], u["_fold"] = seed, fold
            units.append(u)
    return units, missing


def ref_values(out):
    """The reference maps, from the seed-42 reference archives, averaged over folds."""
    per = {}
    missing = []

    def put(nm, measure, fold, value):
        per.setdefault(nm, {}).setdefault(measure, {})[fold] = value

    for fold in FOLDS:
        stale = os.path.join(seed_dir(out, 42), f"refs_f{fold}.npz")
        if os.path.exists(stale):
            print(f"  note: ignoring {stale} (overlap there is over the 400-image subset, "
                  f"not the whole fold)")
        ov = os.path.join(seed_dir(out, 42), f"refs_ov_f{fold}.npz")
        fa = os.path.join(seed_dir(out, 42), f"refs2_f{fold}.npz")
        if not os.path.exists(ov):
            missing.append(ov)
        else:
            with np.load(ov, allow_pickle=True) as z:
                names = [str(x) for x in z["names"]]
                R = z["metrics"]
                for nm in names:
                    i = names.index(nm)
                    for name, c in COLS:
                        put(nm, name, fold, float(np.nanmean(R[:, i, c])))
                    put(nm, "n overlap", fold, int(R.shape[0]))
        if not os.path.exists(fa):
            missing.append(fa)
        else:
            with np.load(fa, allow_pickle=True) as z:
                names = [str(x) for x in z["names"]]
                for nm in names:
                    i = names.index(nm)
                    put(nm, "Ins-Del", fold, float(z["faith"][i]))
                    put(nm, "n faith", fold, int(z["n"]) if "n" in z.files else 0)
    return per, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/headswap",
                    help="the directory holding hs_*.npz and refs*_f*.npz")
    ap.add_argument("--fig", default="figure_3.png")
    ap.add_argument("--json", default="fig3_values.json")
    ap.add_argument("--no-spread", action="store_true",
                    help="draw means only, as the published caption describes")
    ap.add_argument("--seeds", default="42,43,44",
                    help="which readout seeds to average over, comma separated. The figure "
                         "should use the same set the tables do; --seeds 42 reproduces a "
                         "single-seed table row.")
    ap.add_argument("--unsup-stem", default=UNSUP_STEM,
                    help="the archive stem for the unsupervised row. The default is the "
                         "attention distribution the head pools with; pass s3_mlp for the "
                         "rectified score the module originally reported.")
    a = ap.parse_args()

    global SEEDS, ROWS
    SEEDS = tuple(int(s) for s in a.seeds.split(",") if s.strip())
    ROWS = [(lab, ("head", a.unsup_stem) if lab == UNSUP_LAB else kind_stem)
            for lab, kind_stem in ROWS]
    print(f"[make_fig3] unsupervised row from {a.unsup_stem}, seeds {SEEDS}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    problems = []
    vals = {}        # row label -> measure -> (mean, sd, n units)
    counts = {}

    # --- the two head rows ------------------------------------------------------------------
    center_seen = []
    for label, (kind, stem) in ROWS:
        if kind != "head":
            continue
        units, missing = head_units(a.out, stem)
        if missing:
            problems += [f"missing archive: {p}" for p in missing]
        if not units:
            problems.append(f"no archive at all for stem {stem}")
            continue
        vals[label] = {}
        for measure in [c for c, _ in COLS] + ["Ins-Del"]:
            v = np.array([u[measure] for u in units], float)
            vals[label][measure] = (float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                                    len(v))
        # Two counts, because they answer different questions and the caption uses the first.
        one = [u for u in units if u["_seed"] == 42]
        counts[label] = {"distinct overlap images": int(sum(u["n overlap"] for u in one)),
                         "distinct faithfulness images": int(sum(u["n faith"] for u in one)),
                         "overlap measurements": int(sum(u["n overlap"] for u in units)),
                         "faithfulness measurements": int(sum(u["n faith"] for u in units)),
                         "units": len(units)}
        center_seen.append((stem, units))

    # --- the reference rows ----------------------------------------------------------------
    per, missing = ref_values(a.out)
    if missing:
        problems += [f"missing reference archive: {p}" for p in missing]
    for label, (kind, nm) in ROWS:
        if kind != "ref":
            continue
        if nm not in per:
            problems.append(f"reference map {nm!r} is in no reference archive")
            continue
        vals[label] = {}
        for measure in [c for c, _ in COLS] + ["Ins-Del"]:
            seq = list(per[nm].get(measure, {}).values())
            if not seq:
                problems.append(f"reference map {nm!r} has no {measure}")
                continue
            v = np.array(seq, float)
            vals[label][measure] = (float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                                    len(v))
        # A reference map does not depend on the readout, so it is measured once per fold and
        # the distinct count and the measurement count are the same number.
        n_ov = int(sum(per[nm].get("n overlap", {}).values()))
        n_fa = int(sum(per[nm].get("n faith", {}).values()))
        counts[label] = {"distinct overlap images": n_ov,
                         "distinct faithfulness images": n_fa,
                         "overlap measurements": n_ov,
                         "faithfulness measurements": n_fa,
                         "units": len(per[nm].get("IoU", {}))}

    # --- cross-check: the Gaussian the head archives measured against the reference archives --
    worst, worst_at = 0.0, None
    for stem, units in center_seen:
        for u in units:
            for measure in [c for c, _ in COLS] + ["Ins-Del"]:
                got = per.get("center", {}).get(measure, {}).get(u["_fold"])
                if got is None:
                    continue
                d = abs(u["_center"][measure] - got)
                if d > worst:
                    worst, worst_at = d, (stem, u["_seed"], u["_fold"], measure)
    if worst > 5e-3:
        stem, sd_, fd, measure = worst_at
        problems.append(f"the centered Gaussian disagrees between the head and reference "
                        f"archives by up to {worst:.5f} (worst at {stem}, seed {sd_}, "
                        f"fold {fd}, {measure})")
    else:
        print(f"  cross-check: the centered Gaussian agrees between the head and reference "
              f"archives to {worst:.5f}, fold by fold")

    for p in problems:
        print(f"  PROBLEM  {p}")
    if problems:
        raise SystemExit("\nrefusing to draw a figure from archives that do not check out")

    # --- draw -------------------------------------------------------------------------------
    labels = [lab for lab, _ in ROWS if lab in vals]
    y = np.arange(len(labels))[::-1].astype(float)
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11.89, 3.76), dpi=220, gridspec_kw={"width_ratios": [1.35, 1.0]})

    # The three measures of a row are drawn on their own sub-lines rather than on one. Two of
    DY = {m: (1 - j) * 0.22 for j, (m, _) in enumerate(COLS)}
    # Spreading the markers vertically costs the eye the row they belong to, so every other row
    # gets a faint band. The band is what groups the three markers, not their height.
    for i in range(len(labels)):
        if i % 2 == 0:
            for ax in (ax1, ax2):
                ax.axhspan(y[i] - 0.5, y[i] + 0.5, color="0.945", lw=0, zorder=0)
    for i, lab in enumerate(labels):
        for measure, _ in COLS:
            m, sd, n = vals[lab][measure]
            yy = y[i] + DY[measure]
            if not a.no_spread and sd > 0:
                ax1.plot([max(0.0, m - sd), min(1.0, m + sd)], [yy, yy], lw=1.0, color="0.72",
                         solid_capstyle="butt", zorder=1)
            ax1.plot([m], [yy], MARKERS[measure], ms=6.5, mfc="white", mec="0.15", mew=1.3,
                     zorder=3)
    gauss_iou = vals["Centered Gaussian"]["IoU"][0]
    ax1.axvline(gauss_iou, ls="--", lw=1.1, color="0.45", zorder=0)
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=8.5)
    ax1.set_xlabel("overlap with the lesion mask", fontsize=9)
    ax1.set_xlim(0.0, 1.0)
    ax1.set_ylim(y.min() - 0.7, y.max() + 0.7)
    ax1.set_title("(a)  Overlap", loc="left", fontsize=10, fontweight="bold")
    ax1.tick_params(labelsize=8.5)
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)
    handles = [plt.Line2D([], [], ls="", marker=MARKERS[m], ms=7, mfc="white", mec="0.15",
                          mew=1.3, label=t)
               for m, t in (("pointing", "pointing game"), ("energy", "energy ratio"),
                            ("IoU", "area-matched IoU"))]
    ax1.legend(handles=handles, fontsize=8, frameon=False, loc="lower right")

    for i, lab in enumerate(labels):
        m, sd, n = vals[lab]["Ins-Del"]
        ax2.barh([y[i]], [m], height=0.5, color="0.80", edgecolor="0.15", lw=0.9, zorder=2)
        if not a.no_spread and sd > 0:
            ax2.plot([m - sd, m + sd], [y[i], y[i]], lw=1.0, color="0.25",
                     solid_capstyle="butt", zorder=3)
    ax2.axvline(0.0, ls=":", lw=1.2, color="0.25", zorder=1)
    ax2.set_yticks(y)
    ax2.set_yticklabels([])
    ax2.set_xlabel("insertion $-$ deletion", fontsize=9)
    ax2.set_ylim(y.min() - 0.7, y.max() + 0.7)
    ax2.set_title("(b)  Faithfulness", loc="left", fontsize=10, fontweight="bold")
    ax2.tick_params(labelsize=8.5)
    for s in ("top", "right", "left"):
        ax2.spines[s].set_visible(False)

    fig.tight_layout()
    fig.savefig(a.fig)
    print(f"  wrote {a.fig}")

    # --- the values, so the caption can be checked against them -----------------------------
    dump = {}
    for lab in labels:
        for measure, (m, sd, n) in vals[lab].items():
            dump[f"fig3|{lab.replace(chr(10), ' ')}|{measure}"] = m
            dump[f"fig3|{lab.replace(chr(10), ' ')}|{measure} sd"] = sd
            dump[f"fig3|{lab.replace(chr(10), ' ')}|{measure} units"] = n
    for lab, c in counts.items():
        for key, v in c.items():
            dump[f"fig3|{lab.replace(chr(10), ' ')}|{key}"] = v
    with open(a.json, "w") as fh:
        json.dump(dump, fh, indent=1, sort_keys=True)
    print(f"  wrote {a.json}  ({len(dump)} keys)")

    # --- what the caption has to say, read off the figure rather than remembered ------------
    print()
    print("  the numbers the caption depends on")
    w = max(len(l.replace("\n", " ")) for l in labels)
    print(f"    {'map':<{w}}{'pointing':>10}{'energy':>9}{'IoU':>9}{'Ins-Del':>10}")
    for lab in labels:
        v = vals[lab]
        print(f"    {lab.replace(chr(10), ' '):<{w}}"
              f"{v['pointing'][0]:>10.4f}{v['energy'][0]:>9.4f}{v['IoU'][0]:>9.4f}"
              f"{v['Ins-Del'][0]:>+10.4f}")
    # Per seed, so a figure averaged over three seeds can be reconciled with a table row that
    print()
    print("  each head row by readout seed (5 folds per seed)")
    w2 = max(len(s.replace("\n", " ")) for s, _ in [(l, k) for l, k in ROWS if k[0] == "head"])
    print(f"    {'map':<{w2}}{'seed':>6}{'pointing':>10}{'energy':>9}{'IoU':>9}{'Ins-Del':>10}")
    for stem, units in center_seen:
        lab = next(l for l, k in ROWS if k == ("head", stem))
        for seed in SEEDS:
            us = [u for u in units if u["_seed"] == seed]
            if not us:
                continue
            row = [float(np.mean([u[m] for u in us]))
                   for m in ("pointing", "energy", "IoU", "Ins-Del")]
            print(f"    {lab.replace(chr(10), ' '):<{w2}}{seed:>6}"
                  f"{row[0]:>10.4f}{row[1]:>9.4f}{row[2]:>9.4f}{row[3]:>+10.4f}")
        for m in ("pointing", "energy", "IoU", "Ins-Del"):
            per_seed = [float(np.mean([u[m] for u in units if u["_seed"] == s]))
                        for s in SEEDS if any(u["_seed"] == s for u in units)]
            if len(per_seed) > 1:
                swing = max(per_seed) - min(per_seed)
                if swing > 0.02:
                    print(f"    {'':<{w2}}{'':>6}  {m}: moves {swing:.4f} between seeds, so a "
                          f"row built from one seed is not this row")

    print()
    u = vals[UNSUP_LAB]
    beats = [m for m, _ in COLS if u[m][0] >= vals["Centered Gaussian"][m][0]]
    print(f"    panel (a): the dashed line is the centered Gaussian's IoU, {gauss_iou:.4f}")
    if beats:
        print(f"    CAPTION CHECK: the unsupervised head reaches or passes the Gaussian on "
              f"{', '.join(beats)}, so the clause 'does not approach it on any of the three "
              f"measures' is now false")
    else:
        print("    CAPTION CHECK: the unsupervised head is below the Gaussian on all three "
              "measures, as the caption says")
    nf = counts[UNSUP_LAB]["distinct faithfulness images"]
    nm_ = counts[UNSUP_LAB]["faithfulness measurements"]
    nu = counts[UNSUP_LAB]["units"]
    print(f"    panel (b): faithfulness is over {nf:,} distinct images, measured {nm_:,} times "
          f"across {nu} fold-by-seed units")
    if nf == 2000:
        print("    CAPTION CHECK: the caption's '2,000 images' matches the distinct count")
    else:
        print(f"    CAPTION CHECK: the caption says 2,000 images; the archives say {nf:,} "
              f"distinct. Change the caption to {nf:,}.")
    if not a.no_spread:
        print()
        print("    the bars behind the markers are the spread across the fifteen fold-by-seed")
        print("    units. If you keep them, add to the caption: 'Bars show one standard")
        print("    deviation across the fifteen fold-by-seed units.' Or redraw with --no-spread.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
