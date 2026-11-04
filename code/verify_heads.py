"""Recompute the explanation-head tables from their saved archives. No GPU, no dataset.

    python verify_heads.py table # print the tables
    python verify_heads.py dump my_heads.json # recompute, as a dump
    python compare_values.py values_heads.json my_heads.json
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

# The reference maps, in the order within_refs2.py stores them. Keeping the tuple here rather
# than importing it is deliberate: the import would pull in torch.
REF_NAMES = ("gc1", "gc2", "gc3", "gc4", "ig", "mask", "center", "random")

# Row label as the manuscript prints it -> the stem its archives are named with. The stem is
LADDER = [
    # (row label, stem, block)
    ("Scale and shift, per channel",      "ssf",               "unsupervised"),
    ("Linear score (rank 1)",             "lowrank_mapsoftmax", "unsupervised"),
    ("Linear score (rank 1), relu map",   "lowrank",           "unsupervised"),
    ("Class activation map",              "cam",               "unsupervised"),
    ("Position-wise MLP",                 "mlp_mapsoftmax",    "unsupervised"),
    ("Position-wise MLP, relu map",       "mlp",               "unsupervised"),
    ("Image-conditional",                 "cond_mapsoftmax",   "unsupervised"),
    ("Image-conditional, relu map",       "cond",              "unsupervised"),
    ("Linear score (rank 1)",             "lowrank_sup0.2",    "supervised"),
    ("Class activation map",              "cam_sup0.2",        "supervised"),
    ("Position-wise MLP",                 "mlp_sup0.2",        "supervised"),
    ("Image-conditional",                 "cond_sup0.2",       "supervised"),
    # The third-stage MLP is reported from its softmax map, which is the distribution the head
    ("MLP at the third stage",            "s3_mlp_mapsoftmax", "intervention"),
    ("MLP at the third stage, relu map",  "s3_mlp",            "intervention"),
    # The global-feature ablation. `improve/ablate_macro.py` removes the macro term from
    ("MLP at the third stage, no global feature", "s3_mlp_nomacro", "intervention"),
    ("MLP at the third stage, sup.",      "s3_mlp_sup0.2",     "intervention"),
    ("MLP at the fourth stage",           "s4_mlp",            "intervention"),
    ("MLP at the fourth stage, sup.",     "s4_mlp_sup0.2",     "intervention"),
    ("Image-conditional, third stage",    "s3_cond_mapsoftmax", "intervention"),
    ("Image-conditional, third stage, relu map", "s3_cond",    "intervention"),
    ("Image-conditional, third stage, sup.", "s3_cond_sup0.2", "intervention"),
    ("MLP, gradient target",              "gradgc3_mlp_sup0.2", "intervention"),
    # The rectified block as the manuscript now prints it. Those five rows were first measured
    ("Rank 1, second stage",              "lowrank_maprelu",   "unsupervised"),
    ("MLP, second stage",                 "mlp_maprelu",       "unsupervised"),
    ("Conditional, second stage",         "cond_maprelu",      "unsupervised"),
    ("MLP, third stage",                  "s3_mlp_maprelu",    "intervention"),
    ("Conditional, third stage",          "s3_cond_maprelu",   "intervention"),
]

# The label-efficiency sweep, which is a table of its own. The fractions are written as the
# module's %g formatting writes them, so "1" and not "1.0".
FRACTIONS = [
    ("10.1% of the training split",  "s3_mlp_frac0.1_sup0.2"),
    ("25.5%",                        "s3_mlp_frac0.25_sup0.2"),
    ("50.7%",                        "s3_mlp_frac0.5_sup0.2"),
    ("100%",                         "s3_mlp_frac1_sup0.2"),
]

# The reference maps as the manuscript labels them.
REF_ROWS = [
    ("Grad-CAM (stage-1)",     "gc1"),
    ("Grad-CAM (stage-2)",     "gc2"),
    ("Grad-CAM (stage-3)",     "gc3"),
    ("Grad-CAM (stage-4)",     "gc4"),
    ("Integrated Gradients",   "ig"),
    ("Lesion mask as a map",   "mask"),
    ("Centered Gaussian",       "center"),
    ("Random map",             "random"),
]

COLS = (("pointing", 0), ("energy", 1), ("IoU", 2))


def k(table, row, what):
    """verify_all.py's key format, so the two dumps merge and compare."""
    return f"{table}|{row}|{what}"


def source_id(n=10):
    import hashlib
    try:
        with open(os.path.abspath(__file__), "rb") as fh:
            return hashlib.sha1(fh.read()).hexdigest()[:n]
    except OSError:
        return "unknown"


def msd(v):
    """Mean and sample standard deviation, with ddof=1 as head_swap.py's own summary uses."""
    a = np.asarray(v, dtype=float)
    if a.size == 0:
        return float("nan"), float("nan")
    return float(a.mean()), float(a.std(ddof=1)) if a.size > 1 else 0.0


def archives(d, stem):
    """The five fold archives of one stem, and nothing else."""
    pat = re.compile(r"^hs_" + re.escape(stem) + r"_f\d+\.npz$")
    return sorted(p for p in glob.glob(os.path.join(d, "hs_*.npz"))
                  if pat.fullmatch(os.path.basename(p)))


def ref_archives(d, prefix):
    """The fold archives of a reference-map run, by the same exact rule."""
    pat = re.compile(r"^" + re.escape(prefix) + r"_f\d+\.npz$")
    return sorted(p for p in glob.glob(os.path.join(d, prefix + "_f*.npz"))
                  if pat.fullmatch(os.path.basename(p)))


def read_head(path):
    """One trained head's fold archive, reduced to the six numbers a table row needs."""
    with np.load(path, allow_pickle=True) as z:
        R = z["metrics"]
        o = {name: float(np.nanmean(R[:, 0, c])) for name, c in COLS}
        o["n overlap"] = int(R.shape[0])
        ref = {name: float(np.nanmean(R[:, 1, c])) for name, c in COLS}
        o["Ins-Del"] = float(z["faith_head"][2]) if "faith_head" in z.files else float("nan")
        o["within"] = (float(z["faith_head"][5])
                       if "faith_head" in z.files and len(z["faith_head"]) > 5 else float("nan"))
        ref["Ins-Del"] = float(z["faith_center"][2]) if "faith_center" in z.files else float("nan")
        ref["within"] = (float(z["faith_center"][5])
                         if "faith_center" in z.files and len(z["faith_center"]) > 5
                         else float("nan"))
        o["n faith"] = int(z["faith_n"]) if "faith_n" in z.files else 0
        o["n within"] = int(z["faith_n_within"]) if "faith_n_within" in z.files else 0
        o["mechanism"] = int(z["params_mechanism"]) if "params_mechanism" in z.files else None
        o["sha1"] = str(z["source_sha1"]) if "source_sha1" in z.files else "unstamped"
    return o, ref


def read_refs(d, fold):
    """The reference maps of one fold, taking each measure from the run that is entitled to it."""
    out = {nm: {} for nm in REF_NAMES}
    ov = os.path.join(d, f"refs_ov_f{fold}.npz")
    if os.path.exists(ov):
        with np.load(ov, allow_pickle=True) as z:
            names = [str(x) for x in z["names"]]
            R = z["metrics"]
            for nm in REF_NAMES:
                if nm in names:
                    i = names.index(nm)
                    for name, c in COLS:
                        out[nm][name] = float(np.nanmean(R[:, i, c]))
                    out[nm]["n overlap"] = int(R.shape[0])
    fa = os.path.join(d, f"refs2_f{fold}.npz")
    if os.path.exists(fa):
        with np.load(fa, allow_pickle=True) as z:
            names = [str(x) for x in z["names"]]
            for nm in REF_NAMES:
                if nm in names:
                    i = names.index(nm)
                    out[nm]["Ins-Del"] = float(z["faith"][i])
                    out[nm]["within"] = float(z["within"][i])
                    out[nm]["n faith"] = int(z["n"]) if "n" in z.files else 0
    return out


def collect(root, seed=42):
    """Every row the archives under one output root can support, with the cross-checks."""
    d = os.path.join(root, "headswap")
    # Seed 42 writes flat and every other seed into a subdirectory of its own, as
    # improve.localize has always done and as head_swap.seed_out now does for these heads.
    if seed != 42:
        d = os.path.join(d, f"seed{seed}")
    if not os.path.isdir(d):
        raise SystemExit(
            f"nothing to read at {d}\n"
            + (f"  seed {seed} writes to a subdirectory of its own; it has not been run under "
               f"this output root\n" if seed != 42 else
               f"  pass --root, or export HAM_OUTPUT_ROOT\n"))

    rows, notes, gauss = [], [], {}
    # The label-efficiency rows are read by the same path and the same checks. The 100% row is
    for label, stem, block in LADDER + [(l, s_, "label efficiency") for l, s_ in FRACTIONS]:
        files = archives(d, stem)
        if not files:
            continue
        per, refs, mech, sha = [], [], set(), set()
        for p in files:
            o, r = read_head(p)
            per.append(o)
            refs.append(r)
            mech.add(o["mechanism"])
            sha.add(o["sha1"])
            fold = os.path.basename(p).rsplit("_f", 1)[1].split(".")[0]
            gauss.setdefault(fold, []).append((stem, r))
        agg = {"row": label, "stem": stem, "block": block, "folds": len(files)}
        for name in ("pointing", "energy", "IoU", "Ins-Del", "within"):
            m, s = msd([x[name] for x in per])
            agg[name], agg[name + " sd"] = m, s
        for name in ("n overlap", "n faith", "n within"):
            agg[name] = sorted({x[name] for x in per})
        # The per-fold counts as a multiset, keyed by fold, which is what the cross-row check
        agg["n by fold"] = {os.path.basename(p).rsplit("_f", 1)[1].split(".")[0]: x["n overlap"]
                            for p, x in zip(files, per)}
        agg["mechanism"] = sorted(m for m in mech if m is not None)
        agg["sha1"] = sorted(sha)
        rows.append(agg)
        # A row short of the five folds is the quietest way a wrong number reaches a table: it
        # aggregates without complaint and prints a mean and a spread that look like the others.
        if len(files) != 5:
            notes.append(f"{stem}: {len(files)} fold archive(s), not 5. Its mean and spread are "
                         f"over those folds alone and are not comparable to a five-fold row")
        if len(mech) > 1:
            notes.append(f"{stem}: the folds report {len(mech)} different mechanism sizes "
                         f"{agg['mechanism']}, so they are not all the same head")
        if len(sha) > 1:
            notes.append(f"{stem}: the folds were written by {len(sha)} versions of the module "
                         f"{agg['sha1']}, so which fold came from which is unrecoverable")
    # Overlap is over the whole held-out fold, and the folds are not all the same size: a
    consensus = {}
    for f in sorted({f for r in rows for f in r["n by fold"]}):
        seen = [r["n by fold"][f] for r in rows if f in r["n by fold"]]
        consensus[f] = max(set(seen), key=seen.count)
    for r in rows:
        odd = {f: n for f, n in r["n by fold"].items() if n != consensus.get(f)}
        if odd:
            notes.append(f"{r['stem']}: measured fold(s) " + ", ".join(
                f"{f} over {n} images where every other row reports {consensus[f]}"
                for f, n in sorted(odd.items())) + ", so this row is over different images")

    ref_rows = []
    folds = sorted({os.path.basename(p).rsplit("_f", 1)[1].split(".")[0]
                    for p in ref_archives(d, "refs_ov") + ref_archives(d, "refs2")})
    per_fold = {f: read_refs(d, f) for f in folds}
    for label, nm in REF_ROWS:
        got = [per_fold[f][nm] for f in folds if per_fold[f][nm]]
        if not got:
            continue
        agg = {"row": label, "stem": nm, "block": "reference", "folds": len(got)}
        for name in ("pointing", "energy", "IoU", "Ins-Del", "within"):
            vals = [g[name] for g in got if name in g]
            m, s = msd(vals)
            agg[name], agg[name + " sd"] = m, s
        for name in ("n overlap", "n faith"):
            agg[name] = sorted({g[name] for g in got if name in g})
        ref_rows.append(agg)

    return rows, ref_rows, gauss, notes, folds


# ---------------------------------------------------------------- the three smaller tables

LAMBDAS = ("0.2", "0.5", "1", "5")          # as %g writes them: 1.0 is "1", not "1.0"
PIXELAP_HEADS = ("s3_mlp", "s3_mlp_mapsoftmax", "s3_mlp_sup0.2")


def collect_lambda(d):
    """The alignment-weight sweep. Reported per fold, because the sweep was run on one fold and"""
    out = []
    for lam in LAMBDAS:
        for p in archives(d, f"s3_mlp_sup{lam}"):
            fold = os.path.basename(p).rsplit("_f", 1)[1].split(".")[0]
            with np.load(p, allow_pickle=True) as z:
                out.append({"lambda": lam, "fold": fold,
                            "IoU": float(np.nanmean(z["metrics"][:, 0, 2])),
                            "Ins-Del": float(z["faith_head"][2]),
                            "within": float(z["faith_head"][5]),
                            "n": int(z["metrics"].shape[0])})
    return out


def collect_pixelap(d):
    """Threshold-free agreement. The rows that depend on the backbone and not on the head must"""
    out, shared = [], {}
    for head in PIXELAP_HEADS:
        fs = sorted(p for p in glob.glob(os.path.join(d, f"pixelap_{head}_f*.npz"))
                    if re.fullmatch(r"pixelap_" + re.escape(head) + r"_f\d+\.npz",
                                    os.path.basename(p)))
        if not fs:
            continue
        cols, area, folds = {}, [], []
        for p in fs:
            fold = os.path.basename(p).rsplit("_f", 1)[1].split(".")[0]
            folds.append(fold)
            with np.load(p, allow_pickle=True) as z:
                nm = [str(x) for x in z["names"]]
                for i, name in enumerate(nm):
                    v = float(np.nanmean(z["ap"][:, i]))
                    cols.setdefault(name, []).append(v)
                    if name != "head":
                        shared.setdefault((name, fold), set()).add(round(v, 6))
                area.append(float(np.mean(z["lesion_area"])))
        row = {"stem": head, "folds": len(fs)}
        for name, v in cols.items():
            row[name], row[name + " sd"] = msd(v)
        row["random"], row["random sd"] = msd(area)
        row["n"] = sorted({int(np.load(p)["ap"].shape[0]) for p in fs})
        out.append(row)
    disagree = [f"{name} differs between heads on fold {fold}: {sorted(v)}"
                for (name, fold), v in shared.items() if len(v) > 1]
    return out, disagree


def seed_dirs(root):
    """Every readout seed with archives under an output root, as (seed, directory)."""
    base = os.path.join(root, "headswap")
    got = [(42, base)] if glob.glob(os.path.join(base, "hs_*.npz")) else []
    for p in sorted(glob.glob(os.path.join(base, "seed*"))):
        m = re.fullmatch(r"seed(\d+)", os.path.basename(p))
        if m and glob.glob(os.path.join(p, "hs_*.npz")):
            got.append((int(m.group(1)), p))
    return got


def collect_foldlevel(root, stems=("s3_mlp_mapsoftmax", "s3_mlp", "s3_mlp_sup0.2")):
    """Paired comparisons at the level of folds, pooled over whatever readout seeds exist."""
    try:
        from scipy import stats
    except ImportError:
        return None, ["scipy is not installed, so the paired tests were not computed"]

    seeds = seed_dirs(root)
    base = os.path.join(root, "headswap")
    # Grad-CAM at the second stage, per fold, from the reference run. It reads the frozen
    # backbone and not the head, so it does not vary with the readout seed.
    gc2 = {}
    for p in ref_archives(base, "refs_ov"):
        fold = os.path.basename(p).rsplit("_f", 1)[1].split(".")[0]
        with np.load(p, allow_pickle=True) as z:
            nm = [str(x) for x in z["names"]]
            if "gc2" in nm:
                gc2[fold] = float(np.nanmean(z["metrics"][:, nm.index("gc2"), 2]))

    rows, notes = [], []
    for stem in stems:
        head, cen, g2, units = [], [], [], []
        for sd, d in seeds:
            for p in archives(d, stem):
                fold = os.path.basename(p).rsplit("_f", 1)[1].split(".")[0]
                with np.load(p, allow_pickle=True) as z:
                    head.append(float(np.nanmean(z["metrics"][:, 0, 2])))
                    cen.append(float(np.nanmean(z["metrics"][:, 1, 2])))
                g2.append(gc2.get(fold, float("nan")))
                units.append(f"s{sd}f{fold}")
        if not head:
            continue
        for ref_name, ref in (("Centered Gaussian", cen), ("Grad-CAM (stage-2)", g2)):
            a, b = np.array(head), np.array(ref)
            keep = ~np.isnan(b)
            a, b = a[keep], b[keep]
            if a.size < 2:
                notes.append(f"{stem} vs {ref_name}: {a.size} paired unit(s), too few to test")
                continue
            t = stats.ttest_rel(a, b)
            try:
                w = stats.wilcoxon(a, b).pvalue
            except ValueError:
                w = float("nan")
            rows.append({"stem": stem, "reference": ref_name, "n": int(a.size),
                         "delta IoU": float((a - b).mean()),
                         "t p": float(t.pvalue), "wilcoxon p": float(w),
                         "wilcoxon floor": 2.0 ** -(int(a.size) - 1),
                         "units": " ".join(units[:int(a.size)])})
    if rows and rows[0]["n"] < 15:
        notes.append(f"the paired tests rest on {rows[0]['n']} units; the published table's rest "
                     f"on 15, five folds by three readout seeds. Wilcoxon cannot go below "
                     f"{rows[0]['wilcoxon floor']:.4g} at this size whatever the effect")
    return rows, notes


def check_gaussian(gauss, tol=5e-4):
    """The centered Gaussian, which every trained head measured beside itself, compared across"""
    lines, worst = [], 0.0
    for fold in sorted(gauss):
        got = gauss[fold]
        for name in ("IoU", "Ins-Del", "within"):
            vals = [(s, r[name]) for s, r in got if name in r and not np.isnan(r[name])]
            if len(vals) < 2:
                continue
            lo, hi = min(vals, key=lambda t: t[1]), max(vals, key=lambda t: t[1])
            spread = hi[1] - lo[1]
            worst = max(worst, spread)
            if spread > tol:
                lines.append(f"  fold {fold}  {name:<8} spread {spread:.4f} over "
                             f"{len(vals)} heads: {lo[0]} {lo[1]:.4f} .. {hi[0]} {hi[1]:.4f}")
    return lines, worst


def table(rows, ref_rows, gauss, notes, folds, root, extra, seed=42):
    print(f"[verify_heads] produced by verify_heads.py {source_id()}")
    print(f"[verify_heads] output root {root}  seed {seed}")
    print(f"[verify_heads] folds with reference archives: {', '.join(folds) or 'none'}")
    print()
    w = max([24] + [len(r["row"]) for r in rows + ref_rows])
    hdr = (f"{'row':<{w}}{'n':>3}{'mech':>10}{'Pointing':>19}{'Energy':>17}{'IoU':>17}"
           f"{'Ins-Del':>17}{'within':>17}")
    last = None
    for r in rows + ref_rows:
        if r["block"] != last:
            print()
            print(f"-- {r['block']}")
            print(hdr)
            last = r["block"]
        mech = r.get("mechanism") or []
        ms = f"{mech[0]:,}" if len(mech) == 1 else ("---" if not mech else "MIXED")
        # The three overlap measures are non-negative by construction, so a sign on them would
        # only add noise; the two faithfulness measures are signed and their sign is the result.
        cells = "".join(f"{r[n]:>12.4f} ±{r[n + ' sd']:<4.4f}"
                        for n in ("pointing", "energy", "IoU"))
        cells += "".join(f"{r[n]:>+12.4f} ±{r[n + ' sd']:<4.4f}"
                         for n in ("Ins-Del", "within"))
        print(f"{r['row']:<{w}}{r['folds']:>3}{ms:>10}{cells}")

    lam, pap, pap_bad, fl, fl_notes = extra
    if lam:
        print()
        print("-- alignment weight, per fold (single-fold values; do not read against a "
              "five-fold mean)")
        print(f"{'lambda':<10}{'fold':>6}{'n':>7}{'IoU':>12}{'Ins-Del':>12}{'within':>12}")
        for r in lam:
            print(f"{r['lambda']:<10}{r['fold']:>6}{r['n']:>7}{r['IoU']:>12.4f}"
                  f"{r['Ins-Del']:>+12.4f}{r['within']:>+12.4f}")

    if pap:
        print()
        print("-- threshold-free agreement, average precision over the whole held-out fold")
        names = [n for n in ("head", "gc4", "gc2", "center", "random") if n in pap[0]]
        print(f"{'head':<18}{'folds':>6}" + "".join(f"{n:>18}" for n in names))
        for r in pap:
            cells = "".join(f"{r[n]:>11.4f} \u00b1{r[n + ' sd']:<5.4f}" for n in names)
            print(f"{r['stem']:<18}{r['folds']:>6}{cells}")
        print("    the rows other than `head` read the backbone and not the head, so they must "
              "come out")
        print("    identical for both heads of a fold; that is this measurement's own check")
        for b in pap_bad:
            print(f"    DISAGREES: {b}")

    if fl:
        print()
        print("-- paired comparisons at the level of folds")
        print(f"{'map':<18}{'reference':<22}{'n':>4}{'delta IoU':>12}{'t p':>12}"
              f"{'wilcoxon p':>13}{'floor':>10}")
        for r in fl:
            print(f"{r['stem']:<18}{r['reference']:<22}{r['n']:>4}{r['delta IoU']:>+12.4f}"
                  f"{r['t p']:>12.2e}{r['wilcoxon p']:>13.2e}{r['wilcoxon floor']:>10.4g}")
    for note in fl_notes:
        print(f"    note: {note}")

    print()
    print("-- cross-checks")
    lines, worst = check_gaussian(gauss)
    if not gauss:
        print("  centered Gaussian  no trained-head archives to compare")
    elif not lines:
        print(f"  centered Gaussian  agrees across every head of every fold "
              f"(largest spread {worst:.5f})")
    else:
        print(f"  centered Gaussian  DISAGREES; the rows below are not mutually comparable")
        for ln in lines:
            print(ln)

    # Reported per fold rather than per row. The folds are of unequal size, so a list of the
    byfold = {}
    for r in rows:
        for f, n in r["n by fold"].items():
            byfold.setdefault(f, {}).setdefault(n, []).append(r["stem"])
    if byfold:
        sizes = ", ".join(f"fold {f}: {list(v)[0]}" if len(v) == 1 else f"fold {f}: SPLIT {sorted(v)}"
                          for f, v in sorted(byfold.items()))
        bad = [f for f, v in byfold.items() if len(v) > 1]
        print(f"  {'n overlap':<17} {sizes}")
        print(f"  {'':<17} " + ("every row agrees about every fold"
                                if not bad else f"rows disagree about fold(s) {sorted(bad)}"))
    nf = sorted({v for r in rows for v in r.get("n faith", [])})
    if nf:
        print(f"  {'n faith':<17} " + (f"{nf[0]} images for every row" if len(nf) == 1
                                       else f"differs between rows: {nf}"))
    print(f"  {'overlap vs faith':<17} overlap is over the whole held-out fold and faithfulness "
          f"over a subset, so these two are expected to differ from each other")

    if notes:
        print()
        print("-- problems")
        for n in notes:
            print(f"  {n}")
    else:
        print()
        print("-- problems   none: every row is one head, one code version, one protocol")


def dump(rows, ref_rows, gauss, notes, folds, root, extra, out_path):
    V = {}
    for r in rows + ref_rows:
        for name in ("pointing", "energy", "IoU", "Ins-Del", "within"):
            V[k("S17", f"{r['block']} | {r['stem']}", name)] = r[name]
            V[k("S17", f"{r['block']} | {r['stem']}", name + " sd")] = r[name + " sd"]
        if r.get("mechanism") and len(r["mechanism"]) == 1:
            V[k("S17", f"{r['block']} | {r['stem']}", "mechanism")] = r["mechanism"][0]
        V[k("S17", f"{r['block']} | {r['stem']}", "folds")] = r["folds"]
    lam, pap, pap_bad, fl, fl_notes = extra
    for r in lam:
        for what in ("IoU", "Ins-Del", "within", "n"):
            V[k("S9", f"lambda {r['lambda']} | fold {r['fold']}", what)] = r[what]
    for r in pap:
        for name in ("head", "gc4", "gc2", "center", "random"):
            if name in r:
                V[k("S12", r["stem"], name)] = r[name]
                V[k("S12", r["stem"], name + " sd")] = r[name + " sd"]
        V[k("S12", r["stem"], "folds")] = r["folds"]
    for r in fl:
        for what in ("delta IoU", "t p", "wilcoxon p", "n"):
            V[k("S8", f"{r['stem']} vs {r['reference']}", what)] = r[what]
    V["verify_heads|pixel ap|disagreements"] = len(pap_bad)
    V["verify_heads|fold level|notes"] = len(fl_notes)
    lines, worst = check_gaussian(gauss)
    V["verify_heads|source|sha1"] = source_id()
    V["verify_heads|gaussian|largest spread"] = worst
    V["verify_heads|gaussian|disagreements"] = len(lines)
    V["verify_heads|problems|count"] = len(notes)
    V["verify_heads|output root"] = os.path.abspath(root)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(V, fh, indent=1, sort_keys=True)
    print(f"[verify_heads] {len(V)} values -> {out_path}")
    print(f"[verify_heads] produced by verify_heads.py {source_id()}")
    if lines or notes:
        print(f"[verify_heads] {len(lines)} Gaussian disagreements and {len(notes)} problems; "
              f"run `table` to see them")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=["table", "dump"])
    ap.add_argument("out", nargs="?", help="where to write the dump")
    ap.add_argument("--seed", type=int, default=42,
                    help="which readout seed to read; 42 is the flat directory, others a "
                         "seed<N> subdirectory of it")
    ap.add_argument("--root", default=None,
                    help="output root to read; defaults to HAM_OUTPUT_ROOT, then ./outputs")
    a = ap.parse_args()

    root = a.root or os.environ.get("HAM_OUTPUT_ROOT") or "outputs"
    rows, ref_rows, gauss, notes, folds = collect(root, a.seed)
    # The three smaller tables. The weight sweep and the threshold-free agreement are read from
    d = os.path.join(root, "headswap") if a.seed == 42 else \
        os.path.join(root, "headswap", f"seed{a.seed}")
    pap, pap_bad = collect_pixelap(d)
    fl, fl_notes = collect_foldlevel(root)
    extra = (collect_lambda(d), pap, pap_bad, fl or [], fl_notes)
    if not rows and not ref_rows:
        raise SystemExit(f"no archives found under {os.path.join(root, 'headswap')}")
    if a.mode == "table":
        table(rows, ref_rows, gauss, notes, folds, root, extra, a.seed)
    else:
        if not a.out:
            raise SystemExit("dump needs an output path")
        dump(rows, ref_rows, gauss, notes, folds, root, extra, a.out)
    return 0


if __name__ == "__main__":
    # Piping this into `head` closes the pipe and python then raises on the next print. The
    try:
        sys.exit(main())
    except BrokenPipeError:
        try:
            sys.stdout.close()
        finally:
            os._exit(0)
