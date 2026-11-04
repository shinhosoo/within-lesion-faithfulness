"""Supplementary Table S16, extended to the three explanation heads.

S16 measures how insertion and deletion respond to the perturbation baseline, over
the eight reference maps. This measures the same four baselines over the three heads
the paper's conclusion rests on: the unsupervised head in the pooled reading, the
same head with the global-feature path removed, and the mask-supervised head.

NOTHING IS RECONSTRUCTED AND NOTHING IS WRITTEN WHERE THE RUNS LIVE

The heads are not rebuilt here. All three already go through improve.deep_head.run,
which wraps the backbone in StageView so the third stage is presented where the
second normally is, builds the head with head_swap.build_head, loads the trained
checkpoint under --measure-only and calls head_swap.faith_of. The no-global-path
variant is the same path with improve.ablate_macro's build_head substituted, which
is how it was produced. This file supplies the arguments and one substitution:

    improve.faithful.blur  ->  the baseline under test

faith_of takes its replacement image from a single call to that function and hands
it to both curve and curve_within, so swapping it changes the baseline for the
whole-image and the within-lesion curves together and changes nothing else. The
baselines themselves are improve.perturb_variants.build_baseline, so they are the
same four S16 reports rather than a second implementation.

deep_head.run writes an npz next to the published archives. Every write is
redirected into --out, and writing anywhere under the output root, the splits or
the data is refused, so the archives cannot be touched.

WHAT MAKES THE RESULT TRUSTWORTHY

The first baseline, blur20, is the published protocol. Its faith_head values must
equal the ones in the published archive for the same head and fold. The script
checks that and stops if they differ, because a head rebuilt even slightly wrong
would still produce plausible numbers under the other three baselines.

    python perturb_heads.py --folds 0,1,2 --out ~/s16_heads
    python perturb_heads.py --folds 3,4   --out ~/s16_heads
    python perturb_heads.py --collect ~/s16_heads

Run it from the project root, beside SkinCancer/ and improve/.
"""
import argparse
import builtins
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())

import torch  # noqa: E402

# The three heads, as the arguments improve.deep_head.run expects. Each one names the
# archive its blur20 column has to reproduce.
HEADS = [
    {"key": "unsup", "head": "mlp", "supervise": 0.0, "nomacro": False,
     "stem": "s3_mlp", "published": "hs_s3_mlp_mapsoftmax_f{}.npz",
     "label": "explanation head, pooled reading"},
    {"key": "nomacro", "head": "mlp_nomacro", "supervise": 0.0, "nomacro": True,
     "stem": "s3_mlp_nomacro", "published": "hs_s3_mlp_nomacro_f{}.npz",
     "label": "no global-feature path"},
    {"key": "sup", "head": "mlp", "supervise": 0.2, "nomacro": False,
     "stem": "s3_mlp_sup0.2", "published": "hs_s3_mlp_sup0.2_f{}.npz",
     "label": "mask supervision, lambda 0.2"},
]

# The four S16 reports. blur20 first, because it is the one that has to reproduce.
BASELINES = ("blur20", "blur5", "mean", "noise")

FAITH = ("DEL_whole", "INS_whole", "InsDel_whole", "DEL_within", "INS_within", "InsDel_within")


# ----------------------------------------------------------- writing guards

def _protected():
    roots = []
    for v in ("HAM_OUTPUT_ROOT", "HAM_SPLIT_ROOT", "HAM_DATA_ROOT", "HAM_MASK_DIR"):
        q = os.environ.get(v)
        if q:
            roots.append(os.path.abspath(q))
    return roots


def _exempt(p):
    ap = os.path.abspath(p)
    return ap.endswith(".pyc") or "__pycache__" in ap.split(os.sep)


def _inside(p):
    if _exempt(p):
        return False
    ap = os.path.abspath(p)
    return any(ap == r or ap.startswith(r + os.sep) for r in _protected())


OUT_DIR = None
_real_savez = np.savez
_real_open = builtins.open
_real_makedirs, _real_mkdir = os.makedirs, os.mkdir


def _savez(file, *a, **k):
    """Keep every npz in --out, whatever path the published code chose."""
    name = os.path.basename(str(file))
    if not name.endswith(".npz"):
        name += ".npz"
    dest = os.path.join(OUT_DIR, name)
    return _real_savez(dest, *a, **k)


def _open(file, mode="r", *a, **k):
    if any(c in str(mode) for c in "wxa+") and _inside(file):
        raise RuntimeError(f"this script writes nothing under the run directories, and a "
                           f"code path tried to open {file} with mode {mode}")
    return _real_open(file, mode, *a, **k)


def _makedirs(name, mode=0o777, exist_ok=False):
    ap = os.path.abspath(name)
    if os.path.isdir(ap):
        return
    if _inside(ap):
        raise RuntimeError(f"this script creates no folders under the run directories, and a "
                           f"code path tried to make {name}")
    parent = os.path.dirname(ap)
    if parent and parent != ap and not os.path.isdir(parent):
        _makedirs(parent, mode, exist_ok=True)
    try:
        _real_mkdir(ap, mode)
    except FileExistsError:
        if not exist_ok:
            raise


def _mkdir(name, mode=0o777, *a, **k):
    if _inside(name):
        raise RuntimeError(f"this script creates no folders under the run directories, and a "
                           f"code path tried to make {name}")
    return _real_mkdir(os.path.abspath(name), mode, *a, **k)


np.savez = _savez
builtins.open = _open
os.makedirs = _makedirs
os.mkdir = _mkdir
torch.save = lambda *a, **k: (_ for _ in ()).throw(
    RuntimeError("this script trains nothing and saves no checkpoint"))


# -------------------------------------------------------- the baseline swap

_ORIG_BLUR = None


class Baseline:
    """improve.faithful.blur replaced by one of S16's four replacements.

    build_baseline is taken from improve.perturb_variants, so the four baselines are
    the ones that table reports. The noise baseline draws from a generator seeded here,
    so its realisation is a fresh draw rather than the one S16 happened to use. Noise is
    stochastic by construction and the table's own column is a single draw as well.

    build_baseline resolves improve.faithful.blur when it is called, not when it is
    imported, and its blur variants are that function with a different sigma. Standing
    in for the name it looks up would therefore call this object again, so the real
    function is put back for the duration of the call. The reentry flag turns what
    would otherwise be a recursion into a readable error.
    """

    def __init__(self, name, seed):
        from SkinCancer.configs.training_config import DEVICE
        from improve.perturb_variants import build_baseline
        self.name = name
        self._build = build_baseline
        self._gen = torch.Generator(device=DEVICE).manual_seed(seed + 20_000)
        self._busy = False

    def __call__(self, x, *a, **k):
        import improve.faithful as F
        if self._busy:
            raise RuntimeError("the baseline builder re-entered itself, which means the real "
                               "improve.faithful.blur was not restored for the call")
        self._busy = True
        mine = F.blur
        F.blur = _ORIG_BLUR
        try:
            return self._build(self.name, x, self._gen)
        finally:
            F.blur = mine
            self._busy = False


def with_baseline(name, seed):
    global _ORIG_BLUR
    import improve.faithful as F
    if _ORIG_BLUR is None:
        _ORIG_BLUR = F.blur                      # captured before anything replaces it
    F.blur = Baseline(name, seed)
    return F.blur


# ------------------------------------------------------------------ the run

def one(head, fold, baseline, seed, faith_n, faith_steps):
    """One head, one fold, one baseline, through the published measure-only path."""
    import improve.deep_head as dh
    from improve.ablate_macro import build_head_nomacro
    from improve.head_swap import build_head as published_build_head

    dh.build_head = build_head_nomacro if head["nomacro"] else published_build_head
    with_baseline(baseline, seed)

    a = argparse.Namespace(
        fold=fold, stage=3, seed=seed, head=head["head"], map="softmax",
        supervise=head["supervise"], measure_only=True, select="att",
        epochs=10, lr=5e-4, limit=0, faith_n=faith_n, faith_steps=faith_steps,
        quiet=True, signed=False, rank=4,
    )
    dh.run(a)
    # deep_head names its npz after the stem and fold; the redirect put it in OUT_DIR.
    for f in sorted(os.listdir(OUT_DIR)):
        if f.startswith("hs_" + head["stem"]) and f.endswith(f"_f{fold}.npz"):
            src = os.path.join(OUT_DIR, f)
            dst = os.path.join(OUT_DIR, f"{head['key']}_{baseline}_f{fold}.npz")
            os.replace(src, dst)
            return dst
    raise SystemExit(f"no npz appeared in {OUT_DIR} for {head['stem']} fold {fold}")


def check_published(head, fold, path):
    """The blur20 column must equal the archive, or the head was not rebuilt faithfully."""
    root = os.environ.get("HAM_OUTPUT_ROOT")
    if not root:
        print("      HAM_OUTPUT_ROOT unset, so the published archive was not compared")
        return
    pub = os.path.join(root, "headswap", head["published"].format(fold))
    if not os.path.exists(pub):
        print(f"      no published archive at {pub}, comparison skipped")
        return
    a = np.asarray(np.load(pub, allow_pickle=True)["faith_head"], dtype=np.float64)
    b = np.asarray(np.load(path, allow_pickle=True)["faith_head"], dtype=np.float64)
    d = np.abs(a - b).max()
    print(f"      against {os.path.basename(pub)}: max |difference| = {d:.2e}")
    for k, x, y in zip(FAITH, a, b):
        print(f"        {k:<14} published {x:+.4f}   here {y:+.4f}")
    if d > 5e-4:
        raise SystemExit("REFUSED: the published baseline does not reproduce the archive, so "
                         "the head is not the one that was measured and the other baselines "
                         "would not mean anything.")


def collect(d):
    rows = {}
    for f in sorted(os.listdir(d)):
        if not f.endswith(".npz") or f.count("_") < 2:
            continue
        key, baseline, fold = f[:-4].rsplit("_", 2)
        try:
            z = np.load(os.path.join(d, f), allow_pickle=True)
            v = np.asarray(z["faith_head"], dtype=np.float64)
            c = np.asarray(z["faith_center"], dtype=np.float64)
        except Exception:
            continue
        rows.setdefault((key, baseline), []).append((v, c, fold))
    if not rows:
        print(f"no results in {d}")
        return
    keys = [h["key"] for h in HEADS]

    def ms(v):
        v = np.asarray(v, float)
        return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)

    for which, idx in (("within-lesion INS-DEL", 5), ("whole-image INS-DEL", 2)):
        print(f"\n{which}")
        print(f"  {'head':<10}{'baseline':<9}{'folds':>6}{'head':>17}"
              f"{'centred Gaussian':>20}{'head - Gaussian':>20}{'folds above':>13}")
        for k in keys:
            for b in BASELINES:
                got = rows.get((k, b))
                if not got:
                    continue
                h = np.array([v[idx] for v, _, _ in got])
                c = np.array([cc[idx] for _, cc, _ in got])
                d = h - c
                hm, hs = ms(h); cm, cs = ms(c); dm, ds = ms(d)
                print(f"  {k:<10}{b:<9}{len(got):>6}"
                      f"{hm:>+11.4f} ± {hs:.3f}"
                      f"{cm:>+14.4f} ± {cs:.3f}"
                      f"{dm:>+14.4f} ± {ds:.3f}"
                      f"{int((d > 0).sum()):>8}/{len(d)}")
            print()
    print("The centred Gaussian is measured in the same pass under the same baseline. The")
    print("last column counts the folds in which the head is above it.")

    # The question the paper's claim about the interventions is actually about: did the
    # intervention move the score of the head it was applied to? That is a paired
    # quantity, fold by fold, and subtracting the Gaussian cannot answer it, because the
    # same Gaussian cancels: (H_int - G) - (H_base - G) = H_int - H_base.
    paired(rows)


def paired(rows):
    """Per-fold differences between each intervention and the head it was applied to."""
    base = "unsup"
    others = [h["key"] for h in HEADS if h["key"] != base]
    T975 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447}

    def perfold(key, baseline, idx):
        got = rows.get((key, baseline))
        if not got:
            return None
        return {f: v[idx] for v, _, f in got}

    print("\n\nPaired differences, fold by fold, against the head the intervention "
          "was applied to")
    print("  This is an exploratory analysis added after the published protocol was fixed.")
    print("  Five folds give a wide interval, and no test is reported.")
    for which, idx in (("within-lesion INS-DEL", 5), ("whole-image INS-DEL", 2)):
        print(f"\n  {which}")
        print(f"    {'intervention':<13}{'baseline':<9}{'folds':>6}{'mean d':>11}"
              f"{'s.d.':>9}{'95% CI':>22}{'d > 0':>8}")
        for k in others:
            for b in BASELINES:
                a = perfold(base, b, idx)
                c = perfold(k, b, idx)
                if not a or not c:
                    continue
                folds = sorted(set(a) & set(c))
                d = np.array([c[f] - a[f] for f in folds], dtype=np.float64)
                n = len(d)
                m = d.mean()
                sd = d.std(ddof=1) if n > 1 else float("nan")
                if n > 1:
                    half = T975.get(n, 2.776) * sd / np.sqrt(n)
                    ci = f"[{m - half:+.4f}, {m + half:+.4f}]"
                else:
                    ci = "n/a"
                print(f"    {k:<13}{b:<9}{n:>6}{m:>+11.4f}{sd:>9.4f}{ci:>22}"
                      f"{int((d > 0).sum()):>6}/{n}")
            print()


LATEX_NAME = {"unsup": "Classification loss alone",
              "nomacro": "Global-feature path removed",
              "sup": "Mask-supervised"}
LATEX_BL = {"blur20": "Blur $\\sigma=20$", "blur5": "Blur $\\sigma=5$",
            "mean": "Per-image mean", "noise": "Noise"}


def latex(d):
    """Emit the tabular body, so the printed table is not transcribed by hand.

    Every number in the new supplementary table comes from this function reading the
    npz files, which is the only way to keep the table and the archives in step. The
    paired difference is computed on the same folds as the two heads share, and the
    interval is the t interval on five folds, which is wide and is labelled as such
    in the caption rather than here.
    """
    rows = {}
    for f in sorted(os.listdir(d)):
        if not f.endswith(".npz") or f.count("_") < 2:
            continue
        key, baseline, fold = f[:-4].rsplit("_", 2)
        try:
            z = np.load(os.path.join(d, f), allow_pickle=True)
            rows.setdefault((key, baseline), []).append(
                (np.asarray(z["faith_head"], dtype=np.float64),
                 np.asarray(z["faith_center"], dtype=np.float64), fold))
        except Exception:
            continue
    if not rows:
        print(f"no results in {d}")
        return
    T975 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447}
    keys = [h["key"] for h in HEADS]

    def per(key, baseline, idx):
        got = rows.get((key, baseline))
        return None if not got else {f: v[idx] for v, _, f in got}

    def gauss(key, baseline, idx):
        got = rows.get((key, baseline))
        return None if not got else {f: c[idx] for _, c, f in got}

    missing = [(k, b) for k in keys for b in BASELINES if not rows.get((k, b))]
    if missing:
        print("% INCOMPLETE: no results for " + ", ".join(f"{k}/{b}" for k, b in missing))

    for which, idx in (("Within-lesion INS$-$DEL", 5), ("Whole-image INS$-$DEL", 2)):
        print(f"\\multicolumn{{7}}{{l}}{{\\emph{{{which}}}}} \\\\")
        for k in keys:
            for b in BASELINES:
                h = per(k, b, idx)
                if not h:
                    continue
                g = gauss(k, b, idx)
                folds = sorted(h)
                v = np.array([h[f] for f in folds])
                gv = np.array([g[f] for f in folds])
                hm = v.mean()
                hs = v.std(ddof=1) if len(v) > 1 else 0.0
                gm = gv.mean()
                name = LATEX_NAME[k] if b == BASELINES[0] else ""
                if k == "unsup":
                    diff = ci = pos = "---"
                else:
                    a = per("unsup", b, idx)
                    sh = sorted(set(a) & set(h))
                    dd = np.array([h[f] - a[f] for f in sh], dtype=np.float64)
                    n = len(dd)
                    m = dd.mean()
                    sd = dd.std(ddof=1) if n > 1 else float("nan")
                    half = T975.get(n, 2.776) * sd / np.sqrt(n) if n > 1 else float("nan")
                    diff = f"${m:+.4f}$"
                    ci = f"$[{m - half:+.4f}, {m + half:+.4f}]$"
                    pos = f"{int((dd > 0).sum())}/{n}"
                print(f"{name} & {LATEX_BL[b]} & ${hm:+.4f}\\pm{hs:.3f}$ & "
                      f"${gm:+.4f}$ & {diff} & {ci} & {pos} \\\\")
            if k != keys[-1]:
                print("\\addlinespace")
        if idx == 5:
            print("\\midrule")
    print("\n% the centred Gaussian column is measured in the same pass; it should match")
    print("% the Centered Gaussian row of Supplementary Table S16 at four decimals")



def as_json(d, out):
    """Write the same numbers the table prints, in the shape results/ already uses.

    Fold values are kept beside the summaries so the table can be rebuilt without the
    archives, and the file records which npz it read so a later reader can tell whether
    it is looking at the same run.
    """
    import hashlib
    rows, files = {}, []
    for f in sorted(os.listdir(d)):
        if not f.endswith(".npz") or f.count("_") < 2:
            continue
        key, baseline, fold = f[:-4].rsplit("_", 2)
        p = os.path.join(d, f)
        z = np.load(p, allow_pickle=True)
        rows.setdefault((key, baseline), []).append(
            (np.asarray(z["faith_head"], float), np.asarray(z["faith_center"], float), fold))
        files.append({"name": f,
                      "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest()})
    T975 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447}
    keys = [h["key"] for h in HEADS]
    doc = {"heads": keys, "baselines": list(BASELINES),
           "paired_against": "unsup",
           "interval": "unadjusted two-sided 95 percent paired t over the folds",
           "measures": {}, "files": files}
    for which, idx in (("within_lesion", 5), ("whole_image", 2)):
        block = {}
        for k in keys:
            for b in BASELINES:
                got = rows.get((k, b))
                if not got:
                    continue
                folds = sorted(f for _, _, f in got)
                h = {f: v[idx] for v, _, f in got}
                g = {f: c[idx] for _, c, f in got}
                v = np.array([h[f] for f in folds])
                e = {"folds": folds,
                     "per_fold": [float(x) for x in v],
                     "mean": float(v.mean()),
                     "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                     "centered_gaussian": float(np.array([g[f] for f in folds]).mean())}
                if k != "unsup":
                    a = {f: vv[idx] for vv, _, f in rows[("unsup", b)]}
                    sh = sorted(set(a) & set(h))
                    dd = np.array([h[f] - a[f] for f in sh])
                    n = len(dd)
                    m = float(dd.mean())
                    sd = float(dd.std(ddof=1)) if n > 1 else float("nan")
                    half = T975.get(n, 2.776) * sd / np.sqrt(n) if n > 1 else float("nan")
                    e["difference"] = {"mean": m, "sd": sd,
                                       "ci95": [m - half, m + half],
                                       "folds_positive": int((dd > 0).sum()),
                                       "folds": n}
                block[f"{k}/{b}"] = e
        doc["measures"][which] = block
    import json
    with _open(out, "w") as fh:
        json.dump(doc, fh, indent=1, sort_keys=False)
        fh.write("\n")
    print(f"wrote {out}  ({len(files)} archives, "
          f"{sum(len(v) for v in doc['measures'].values())} rows)")


def main():
    global OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--baselines", default=",".join(BASELINES))
    ap.add_argument("--heads", default=",".join(h["key"] for h in HEADS))
    ap.add_argument("--out", default=os.path.expanduser("~/s16_heads"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--faith-n", type=int, default=400)
    ap.add_argument("--faith-steps", type=int, default=10)
    ap.add_argument("--collect", metavar="DIR", help="add up an earlier run's npz and stop")
    ap.add_argument("--latex", metavar="DIR",
                    help="print the tabular body for the supplementary table and stop")
    ap.add_argument("--json", nargs=2, metavar=("DIR", "OUT"),
                    help="write the collected values as JSON and stop")
    a = ap.parse_args()

    if a.collect:
        collect(a.collect)
        return

    if a.latex:
        latex(a.latex)
        return

    if a.json:
        as_json(a.json[0], a.json[1])
        return

    OUT_DIR = os.path.abspath(a.out)
    if _inside(OUT_DIR):
        sys.exit(f"REFUSED: --out {OUT_DIR} is inside a run directory; choose a path outside")
    _makedirs(OUT_DIR, exist_ok=True)

    folds = [int(x) for x in a.folds.split(",") if x.strip() != ""]
    bl = [x for x in a.baselines.split(",") if x.strip()]
    if BASELINES[0] not in bl:
        sys.exit(f"REFUSED: {BASELINES[0]} is the published protocol and the column every "
                 f"other baseline is checked against, so it cannot be left out")
    want = [h for h in HEADS if h["key"] in a.heads.split(",")]

    print(f"out      = {OUT_DIR}")
    print(f"folds    = {folds}")
    print(f"heads    = {[h['key'] for h in want]}")
    print(f"baselines= {bl}")
    print(f"protected= {_protected()}")
    print(f"{len(want) * len(folds) * len(bl)} measurements, "
          f"{len(want) * len(folds) * len(bl) * 2 * 4 * (a.faith_steps + 1) * a.faith_n:,} "
          f"forward passes for the curves\n")

    for head in want:
        for fold in folds:
            for baseline in bl:
                print(f"== {head['key']:<8} fold {fold}  baseline {baseline}")
                p = one(head, fold, baseline, a.seed, a.faith_n, a.faith_steps)
                z = np.load(p, allow_pickle=True)
                v = np.asarray(z["faith_head"], dtype=np.float64)
                print(f"      whole INS-DEL {v[2]:+.4f}   within INS-DEL {v[5]:+.4f}")
                if baseline == BASELINES[0]:
                    check_published(head, fold, p)
                print()

    print("done. add the folds up with:")
    print(f"  python {os.path.basename(sys.argv[0])} --collect {OUT_DIR}")


if __name__ == "__main__":
    main()
