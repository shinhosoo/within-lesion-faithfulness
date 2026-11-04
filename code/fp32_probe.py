"""Every accuracy the paper reports, computed twice: once as the runs computed it,
under torch.amp.autocast, and once in single precision.

The point is to find which reported values move at the precision they are printed
to, four decimals, before anything in the manuscript is touched. Nothing is
reimplemented here. Each target calls the same functions its own experiment
called, with autocast switched off for the second pass, so a difference can only
come from the precision.

NOTHING IS WRITTEN WHERE IT WOULD MATTER

Before any project module is imported, the ways results get written are replaced
with functions that refuse: np.savez, np.savez_compressed,
np.lib.format.open_memmap, torch.save and pandas.DataFrame.to_csv.

Creating a directory and opening a file for writing are refused inside the
protected roots only, which are the working directory and whatever
HAM_OUTPUT_ROOT, HAM_SPLIT_ROOT and HAM_DATA_ROOT point at. Everything outside
them is left alone, because the libraries in this stack make temporary
directories under /tmp and caches under the home directory, and refusing those
stops the run for no gain. __pycache__ and .pyc are exempt even inside the
protected roots, since importing the project writes them and blocking that
breaks the import itself.

improve.residual.folds_table would write cv_folds.csv if it were missing, so its
presence is checked first and the run stops rather than regenerating the folds.

SPLITTING THE WORK OVER TWO GPUS

HAM_GPU picks the device, as in the experiments, and --folds picks which folds
this process does. Each process prints one TALLY line per row per fold, and
--combine adds the lines up, so the split costs no accuracy in the result:

    HAM_GPU=0 python fp32_probe.py --target s4 --folds 0,1,2 | tee s4.gpu0.log
    HAM_GPU=1 python fp32_probe.py --target s4 --folds 3,4   | tee s4.gpu1.log
    python fp32_probe.py --combine s4.gpu0.log s4.gpu1.log

Supplementary Table S2 is one pass over the test set with no folds, so it is not
split and needs no combining.

Supplementary Table S3's other backbones go through the wrapper the runs used,
which rebinds the backbone and the output root before anything imports them:

    HAM_GPU=0 python -m improve.bb_run --backbone convnext_tiny \\
        --out outputs_convnext_tiny -- fp32_probe --target s3 --folds 0,1,2

Place this file in the project root, beside SkinCancer/ and improve/.
"""
import argparse
import builtins
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())

import torch  # noqa: E402


# ------------------------------------------------------------- no writing

def _refuse(what):
    def f(*a, **k):
        raise RuntimeError(f"fp32_probe is read-only and a code path tried to {what}")
    return f


np.savez = _refuse("np.savez")
np.savez_compressed = _refuse("np.savez_compressed")
np.lib.format.open_memmap = _refuse("open a memmap for writing")
torch.save = _refuse("torch.save")
pd.DataFrame.to_csv = _refuse("write a csv")

def _protected_roots():
    """Where this script must not write: the project and the data it reads."""
    roots = [os.path.abspath(os.getcwd())]
    for v in ("HAM_OUTPUT_ROOT", "HAM_SPLIT_ROOT", "HAM_DATA_ROOT", "HAM_MASK_DIR"):
        q = os.environ.get(v)
        if q:
            roots.append(os.path.abspath(q))
    return roots


def _exempt(path):
    """Python's own import artefacts, which blocking would break the import over."""
    ap = os.path.abspath(path)
    return ap.endswith(".pyc") or "__pycache__" in ap.split(os.sep)


def _inside_protected(path):
    if _exempt(path):
        return False
    ap = os.path.abspath(path)
    return any(ap == r or ap.startswith(r + os.sep) for r in _protected_roots())


_real_open = builtins.open


def _open(file, mode="r", *a, **k):
    if any(c in str(mode) for c in "wxa+") and _inside_protected(file):
        raise RuntimeError(f"fp32_probe writes nothing under the project or the data "
                           f"roots, and a code path tried to open {file} with mode {mode}")
    return _real_open(file, mode, *a, **k)


builtins.open = _open

_real_makedirs, _real_mkdir = os.makedirs, os.mkdir


def _makedirs(name, mode=0o777, exist_ok=False):
    """os.makedirs, refusing the protected roots.

    The real os.makedirs is not reused, because its own body calls the module-level
    os.mkdir and os.makedirs, which are the names replaced here, so delegating to it
    comes straight back and recurses until the stack ends.
    """
    ap = os.path.abspath(name)
    if os.path.isdir(ap):
        return
    if _inside_protected(ap):
        raise RuntimeError(f"fp32_probe creates no folders under the project or the data "
                           f"roots, and a code path tried to make {name}")
    parent = os.path.dirname(ap)
    if parent and parent != ap and not os.path.isdir(parent):
        _makedirs(parent, mode, exist_ok=True)
    try:
        _real_mkdir(ap, mode)
    except FileExistsError:
        if not exist_ok:
            raise


def _mkdir(name, mode=0o777, *a, **k):
    ap = os.path.abspath(name)
    if _inside_protected(ap):
        raise RuntimeError(f"fp32_probe creates no folders under the project or the data "
                           f"roots, and a code path tried to make {name}")
    return _real_mkdir(ap, mode, *a, **k)


os.makedirs = _makedirs
os.mkdir = _mkdir


# --------------------------------------------------------- precision switch

_AUTOCAST = torch.amp.autocast


class _NoAutocast:
    """Stands in for torch.amp.autocast so nested helpers run in single precision.

    The reused code paths enter autocast in several places, some inside
    improve.zoom and improve.hier rather than here, and every one of them calls
    it as torch.amp.autocast, so replacing the attribute reaches all of them.
    """

    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False

    def __call__(self, fn):
        return fn


class precision:
    def __init__(self, amp):
        self.amp = amp

    def __enter__(self):
        torch.amp.autocast = _AUTOCAST if self.amp else _NoAutocast
        return self

    def __exit__(self, *a):
        torch.amp.autocast = _AUTOCAST
        return False


def both(fn):
    """Run fn under each precision and return (mixed, single)."""
    out = []
    for amp in (True, False):
        with precision(amp):
            out.append(fn())
    return out[0], out[1]


def tally(target, row, fold, n, wrong_mixed, wrong_single, flips):
    print(f"TALLY {target} {row} fold={fold} n={n} wrong_mixed={wrong_mixed} "
          f"wrong_single={wrong_single} flips={flips}")


def conf(target, row, fold, which, y, pred, k=7):
    """One fold's confusion matrix, so the combiner can pool a macro F1.

    Accuracy pools from counts alone, but F1 does not, and Supplementary Table S3
    prints a Macro F1 column with no fold-to-fold spread, so it has to be computed
    over the folds together.
    """
    m = np.zeros((k, k), dtype=np.int64)
    for t, p in zip(np.asarray(y).ravel(), np.asarray(pred).ravel()):
        m[int(t), int(p)] += 1
    flat = ",".join(str(int(v)) for v in m.ravel())
    print(f"CONF {target} {row} fold={fold} which={which} m={flat}")


def macro_f1(m):
    """Macro F1 from a pooled confusion matrix, over the classes that occur."""
    f1 = []
    for i in range(m.shape[0]):
        tp = m[i, i]
        fp = m[:, i].sum() - tp
        fn = m[i, :].sum() - tp
        if m[i, :].sum() == 0 and m[:, i].sum() == 0:
            continue
        d = 2 * tp + fp + fn
        f1.append(0.0 if d == 0 else 2 * tp / d)
    return float(np.mean(f1)) if f1 else float("nan")


def verdict(name, mixed, single, note="", digits=6):
    d = single - mixed
    tag = "MOVES at 4 dp" if round(mixed, 4) != round(single, 4) else "same at 4 dp"
    print(f"  {name:<40} mixed={mixed:.{digits}f}  single={single:.{digits}f}  "
          f"d={d:+.6f}  {tag}{note}")
    # A value sitting on the rounding boundary cannot be transcribed from six decimals,
    # and this column goes into the manuscript, so say so rather than let it be guessed.
    for label, v in (("mixed", mixed), ("single", single)):
        frac = abs(v) * 1e4 % 1
        if abs(frac - 0.5) < 1e-6:
            print(f"  {'':<40} {label} sits on the 4-dp rounding boundary: {v:.10f}")


# ------------------------------------------------------------------ guards

def preflight(folds):
    from SkinCancer.configs.data_config import DATA_ROOT, METADATA_CSV, PROJECT_ROOT, SPLIT_ROOT
    root = os.environ.get("HAM_OUTPUT_ROOT", os.path.join(PROJECT_ROOT, "outputs"))
    print(f"PROJECT_ROOT = {PROJECT_ROOT}   (the folder holding SkinCancer/)")
    print(f"SPLIT_ROOT   = {SPLIT_ROOT}")
    print(f"DATA_ROOT    = {DATA_ROOT}")
    print(f"OUTPUT_ROOT  = {root}")
    print(f"HAM_GPU      = {os.environ.get('HAM_GPU', '(unset, so device 0)')}")
    print(f"folds        = {folds}")
    need = [os.path.join(SPLIT_ROOT, "global_train.csv"),
            os.path.join(SPLIT_ROOT, "global_val.csv"),
            METADATA_CSV,
            root,
            os.path.join(root, "residual", "cv_folds.csv")]
    bad = [p for p in need if not os.path.exists(p)]
    for p in need:
        print(f"  {'ok     ' if os.path.exists(p) else 'MISSING'}  {p}")
    if bad:
        print("\nSet HAM_SPLIT_ROOT, HAM_DATA_ROOT and HAM_OUTPUT_ROOT to the paths the runs")
        print("used. cv_folds.csv must be the stored one: this script will not regenerate it,")
        print("because the fold assignment has to be the one the published numbers used.")
        sys.exit(1)
    print(f"  nothing will be written under: {', '.join(_protected_roots())}")
    print("  temporary files elsewhere are left to the libraries\n")


# ----------------------------------------------------------------- targets

def t_s2(ckpts):
    """Supplementary Table S2: the single-stage classifier on the held-out test set.

    The table prints mean and standard deviation over the classifier seeds, not over
    folds, so one checkpoint per seed is needed. improve.hier takes the seeded ones
    through --flat_ckpt and leaves no naming convention behind, so they are passed in
    with --s2-ckpts. With one checkpoint the spread is not computed and the row is
    labelled as a single seed.
    """
    import timm

    from SkinCancer.configs.data_config import FINAL_CLASS_NAMES
    from SkinCancer.configs.model_config import BACKBONE_NAME
    from SkinCancer.configs.training_config import DEVICE, OUTPUT_ROOT
    from SkinCancer.data.loaders import make_loader
    from SkinCancer.data.split_io import build_image_path_map
    from SkinCancer.data.transforms import build_val_transform
    from SkinCancer.eval.metrics import calculate_detailed_multiclass_metrics
    from improve.hier import load_global, run_loader

    paths = []
    for c in ckpts:
        q = c if os.path.isabs(c) else os.path.join(OUTPUT_ROOT, c)
        if os.path.exists(q):
            paths.append(q)
        else:
            print(f"S2  no checkpoint at {q}, skipped")
    if not paths:
        print("S2  nothing to do, pass --s2-ckpts with the seeded checkpoints")
        return
    df = load_global("test", build_image_path_map())
    print(f"\nS2  single-stage {BACKBONE_NAME} on the test set, n={len(df)}, "
          f"{len(paths)} checkpoint(s)")
    if len(paths) < 5:
        print("    the table's spread is over five classifier seeds, so fewer than five "
              "checkpoints\n    gives the cells but not the spread the table prints")

    # Precision is PPV and Recall is Sensitivity in the metric function's naming.
    WANT = [("Precision", "PPV"), ("Recall", "Sensitivity"), ("F1-score", "F1")]
    cells = {True: {}, False: {}}
    accs = {True: [], False: []}
    flips = 0

    for q in paths:
        def run():
            m = timm.create_model(BACKBONE_NAME, pretrained=False, num_classes=7).to(DEVICE)
            m.load_state_dict(torch.load(q, map_location=DEVICE, weights_only=True),
                              strict=True)
            m.eval()
            loader = make_loader(df, "final_label", build_val_transform(), shuffle=False)
            (P,), y = run_loader([(m, lambda z: torch.softmax(z, 1))], loader)
            return np.asarray(P), np.asarray(y)

        (Pm, y), (Ps, _) = both(run)
        flips += int((Pm.argmax(1) != Ps.argmax(1)).sum())
        for amp, P in ((True, Pm), (False, Ps)):
            accs[amp].append(float((P.argmax(1) == y).mean()))
            ov, pc = calculate_detailed_multiclass_metrics(y, P, FINAL_CLASS_NAMES)
            for cls in pc:
                for shown, key in WANT:
                    cells[amp].setdefault((cls, shown), []).append(float(pc[cls][key]))
            for shown, key in WANT:
                cells[amp].setdefault(("Macro", shown), []).append(float(ov[key]))
        print(f"  {os.path.basename(q)} done")

    def ms(v):
        v = np.asarray(v, float)
        return v.mean(), (v.std(ddof=1) if len(v) > 1 else float("nan"))

    am, as_ = ms(accs[True]), ms(accs[False])
    verdict("overall accuracy (mean over checkpoints)", am[0], as_[0])
    print(f"  {'':<40} sd: mixed={am[1]:.4f}  single={as_[1]:.4f}")
    print(f"  {'':<40} predictions that differ, summed over checkpoints: {flips}")

    moved = []
    for k in cells[True]:
        a, b = ms(cells[True][k]), ms(cells[False][k])
        tags = []
        if round(a[0], 3) != round(b[0], 3):
            tags.append(f"mean {a[0]:.3f} -> {b[0]:.3f}")
        if len(paths) > 1 and round(a[1], 3) != round(b[1], 3):
            tags.append(f"sd {a[1]:.3f} -> {b[1]:.3f}")
        if tags:
            moved.append((k, tags))
    print(f"  S2 cells that move at the printed 3 decimals: {len(moved)} of {len(cells[True])}")
    for (cls, shown), tags in moved:
        print(f"    {cls:<8} {shown:<10} {'; '.join(tags)}")

def t_s3(folds, seeds):
    """Supplementary Table S3: this backbone's cross-validated accuracy, per classifier seed.

    load_flat is inlined only so the seed can vary, which its signature does not allow.
    """
    from SkinCancer.configs.model_config import BACKBONE_NAME
    from SkinCancer.configs.training_config import DEVICE, OUTPUT_ROOT
    from SkinCancer.data.loaders import make_loader
    from SkinCancer.data.split_io import build_image_path_map
    from SkinCancer.data.transforms import build_val_transform
    from improve.residual import ResidualModel, get_data

    pm = build_image_path_map()
    print(f"\nS3  {BACKBONE_NAME}, root {OUTPUT_ROOT}")
    for seed in seeds:
        for fold in folds:
            ck = os.path.join(OUTPUT_ROOT, "residual", f"flat_f{fold}_s{seed}.pth")
            if not os.path.exists(ck):
                print(f"  seed {seed} fold {fold}: no checkpoint at {ck}, skipped")
                continue
            _, va, _ = get_data(fold, pm)

            def run():
                m = ResidualModel("flat", pretrained=False).to(DEVICE)
                m.load_state_dict(torch.load(ck, map_location=DEVICE, weights_only=True),
                                  strict=True)
                m.eval()
                net = m.net
                out, ys = [], []
                with torch.no_grad():
                    for img, lbl in make_loader(va, "label", build_val_transform(),
                                                shuffle=False):
                        with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                            lg = net(img.to(DEVICE))
                        out.append(lg.float().cpu().numpy())
                        ys.append(np.asarray(lbl))
                return np.concatenate(out).argmax(1), np.concatenate(ys).astype(int)

            (pr_m, y), (pr_s, _) = both(run)
            row = f"{BACKBONE_NAME}_seed{seed}"
            tally("s3", row, fold, len(y), int((pr_m != y).sum()),
                  int((pr_s != y).sum()), int((pr_m != pr_s).sum()))
            conf("s3", row, fold, "mixed", y, pr_m)
            conf("s3", row, fold, "single", y, pr_s)


def t_s4(folds):
    """Supplementary Table S4: the two lesion-restricted conditions, stored checkpoints."""
    import timm
    from torch.utils.data import DataLoader

    from SkinCancer.configs.model_config import BACKBONE_NAME
    from SkinCancer.configs.training_config import BATCH_SIZE, DEVICE, OUTPUT_ROOT
    from SkinCancer.data.split_io import build_image_path_map
    from improve.aug2 import LesionAug
    from improve.residual import get_data

    OUT = os.path.join(OUTPUT_ROOT, "residual")
    pm = build_image_path_map()
    print("\nS4  lesion-restricted training, the stored checkpoints re-evaluated")
    for aug in ("lcrop", "bgdim"):
        for fold in folds:
            ck = os.path.join(OUT, f"{aug}_f{fold}_s42.pth")
            if not os.path.exists(ck):
                print(f"  {aug} fold {fold}: no checkpoint at {ck}, skipped")
                continue
            _, va, _ = get_data(fold, pm)

            def run():
                m = timm.create_model(BACKBONE_NAME, pretrained=False,
                                      num_classes=7).to(DEVICE)
                m.load_state_dict(torch.load(ck, map_location=DEVICE, weights_only=True),
                                  strict=True)
                m.eval()
                vl = DataLoader(LesionAug(va, aug, False), batch_size=BATCH_SIZE,
                                shuffle=False, num_workers=4, pin_memory=True)
                P, Y = [], []
                with torch.no_grad():
                    for x, lbl in vl:
                        with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                            P.append(torch.softmax(m(x.to(DEVICE)).float(), 1).cpu().numpy())
                        Y.append(np.asarray(lbl))
                return np.concatenate(P).argmax(1), np.concatenate(Y).astype(int)

            (pr_m, y), (pr_s, _) = both(run)
            tally("s4", aug, fold, len(y), int((pr_m != y).sum()), int((pr_s != y).sum()),
                  int((pr_m != pr_s).sum()))


def t_s15(folds):
    """Supplementary Table S15: the base pass and the mask-box margins.

    Only the mask-box condition is computed, because that is what the table
    reports. The other condition in improve.zoom2 needs the readout head, which
    this never loads.
    """
    from PIL import Image

    from SkinCancer.configs.training_config import BATCH_SIZE
    from SkinCancer.data.split_io import build_image_path_map
    from improve.localize import load_flat
    from improve.residual import get_data
    from improve.zoom import MASK_DIR, predict, square_box, to_tensor
    from improve.zoom2 import MARGINS

    pm = build_image_path_map()
    keys = ["base"] + [f"mask_m{int(m * 100)}" for m in MARGINS]
    print(f"\nS15  magnification at inference, margins {MARGINS}")

    for fold in folds:
        _, va, _ = get_data(fold, pm)
        net = load_flat(fold)

        def run():
            right = {k: 0 for k in keys}
            for st in range(0, len(va), BATCH_SIZE):
                rows = va.iloc[st:st + BATCH_SIZE]
                imgs = [Image.open(p).convert("RGB") for p in rows["image_path"]]
                y = rows["label"].astype(int).values
                base = [to_tensor(im) for im in imgs]
                pb = predict(net, base)
                right["base"] += int((pb.argmax(1) == y).sum())
                boxes = []
                for i, im in enumerate(imgs):
                    W, H = im.size
                    mp = os.path.join(MASK_DIR, f"{rows.iloc[i]['image_id']}_segmentation.png")
                    m = np.array(Image.open(mp).convert("L")) > 127
                    ys, xs = np.where(m)
                    boxes.append((W, H, (xs.min(), ys.min(), xs.max(), ys.max())
                                  if len(xs) else (0, 0, W, H)))
                for mg in MARGINS:
                    crops = []
                    for i, im in enumerate(imgs):
                        W, H, (x0, y0, x1, y1) = boxes[i]
                        crops.append(to_tensor(im.crop(
                            square_box(x0, y0, x1, y1, W, H, margin=mg))))
                    p = 0.5 * (pb + predict(net, crops))
                    right[f"mask_m{int(mg * 100)}"] += int((p.argmax(1) == y).sum())
            return right

        rm, rs = both(run)
        for k in keys:
            tally("s15", k, fold, len(va), len(va) - rm[k], len(va) - rs[k], -1)
        print(f"  fold {fold} done, n={len(va)}")


# ----------------------------------------------------------------- combine

def combine(paths):
    tpat = re.compile(r"^TALLY (\S+) (\S+) fold=(\S+) n=(\d+) wrong_mixed=(\d+) "
                      r"wrong_single=(\d+) flips=(-?\d+)")
    cpat = re.compile(r"^CONF (\S+) (\S+) fold=(\S+) which=(\S+) m=([\d,]+)")
    acc, mats, perfold = {}, {}, {}
    seen, seen_c = set(), set()
    for p in paths:
        for ln in _real_open(p, encoding="utf-8", errors="replace"):
            ln = ln.strip()
            m = tpat.match(ln)
            if m:
                tgt, row, fold, n, wm, ws, fl = m.groups()
                if (tgt, row, fold) in seen:
                    print(f"  note: {tgt} {row} fold {fold} appears twice, counted once")
                    continue
                seen.add((tgt, row, fold))
                a = acc.setdefault((tgt, row),
                                   {"n": 0, "wm": 0, "ws": 0, "fl": 0, "per": {}})
                a["n"] += int(n); a["wm"] += int(wm); a["ws"] += int(ws)
                if int(fl) >= 0:
                    a["fl"] += int(fl)
                a["per"][fold] = (int(n), int(wm), int(ws))
                continue
            m = cpat.match(ln)
            if m:
                tgt, row, fold, which, flat = m.groups()
                if (tgt, row, fold, which) in seen_c:
                    continue
                seen_c.add((tgt, row, fold, which))
                v = np.array([int(x) for x in flat.split(",")], dtype=np.int64)
                k = int(round(len(v) ** 0.5))
                mats.setdefault((tgt, row, which), np.zeros((k, k), dtype=np.int64))
                mats[(tgt, row, which)] += v.reshape(k, k)
                perfold.setdefault((tgt, row, which), {})[fold] = v.reshape(k, k)
        print(f"  read {p}")
    if not acc:
        print("\nno TALLY lines found in those files")
        return
    print()
    for (tgt, row), a in sorted(acc.items()):
        folds = sorted(a["per"])
        note = "" if len(folds) == 5 else f"   [only folds {','.join(folds)}]"
        verdict(f"{tgt} {row}  pooled", 1 - a["wm"] / a["n"], 1 - a["ws"] / a["n"], note)
        fm = np.array([1 - w / n for n, w, _ in (a["per"][f] for f in folds)])
        fs = np.array([1 - w / n for n, _, w in (a["per"][f] for f in folds)])
        sd = (fm.std(ddof=1), fs.std(ddof=1)) if len(folds) > 1 else (float("nan"),) * 2
        print(f"  {'':<40} the table prints the fold mean: "
              f"mixed={fm.mean():.4f} +- {sd[0]:.4f}   "
              f"single={fs.mean():.4f} +- {sd[1]:.4f}")
        if a["fl"]:
            print(f"  {'':<40} predictions that differ: {a['fl']} of {a['n']}")
        km, ks = mats.get((tgt, row, "mixed")), mats.get((tgt, row, "single"))
        if km is not None and ks is not None:
            verdict(f"{tgt} {row}  macro F1 pooled", macro_f1(km), macro_f1(ks), note)
            pm_ = perfold.get((tgt, row, "mixed"), {})
            ps_ = perfold.get((tgt, row, "single"), {})
            if pm_ and ps_:
                fm = np.array([macro_f1(pm_[f]) for f in sorted(pm_)])
                fs = np.array([macro_f1(ps_[f]) for f in sorted(ps_)])
                verdict(f"{tgt} {row}  macro F1 fold mean", float(fm.mean()),
                        float(fs.mean()), note, digits=10)
                sd = (fm.std(ddof=1), fs.std(ddof=1)) if len(fm) > 1 else (float("nan"),) * 2
                print(f"  {'':<40} macro F1 sd: mixed={sd[0]:.4f}  single={sd[1]:.4f}")
    print("\nA row marked 'same at 4 dp' needs no edit. A row that moves needs its table")
    print("cell, and every sentence derived from it, recomputed.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="all", choices=["s2", "s3", "s4", "s15", "all"])
    ap.add_argument("--folds", default="0,1,2,3,4", help="which folds this process does")
    ap.add_argument("--seeds", default="42,43,44",
                    help="classifier seeds for S3, as flat_f{fold}_s{seed}.pth")
    ap.add_argument("--s2-ckpts", default="onestage_7class.pth",
                    help="comma-separated single-stage checkpoints, one per classifier "
                         "seed, absolute or relative to OUTPUT_ROOT; Supplementary "
                         "Table S2's spread is over these")
    ap.add_argument("--combine", nargs="+", metavar="LOG",
                    help="add up the TALLY lines of earlier runs and stop")
    a = ap.parse_args()

    if a.combine:
        combine(a.combine)
        return

    folds = [int(x) for x in a.folds.split(",") if x.strip() != ""]
    seeds = [int(x) for x in a.seeds.split(",") if x.strip() != ""]
    preflight(folds)

    from SkinCancer.configs.training_config import DEVICE
    print(f"device = {DEVICE}")

    jobs = {"s2": lambda: t_s2([c.strip() for c in a.s2_ckpts.split(",") if c.strip()]),
            "s4": lambda: t_s4(folds),
            "s15": lambda: t_s15(folds), "s3": lambda: t_s3(folds, seeds)}
    order = ["s4", "s15", "s3", "s2"] if a.target == "all" else [a.target]
    for k in order:
        try:
            jobs[k]()
        except Exception as e:                   # one target failing is not the others'
            print(f"\n{k}: failed, {type(e).__name__}: {e}")
    print("\nPer-fold counts are on the TALLY lines above. Add the two GPUs' logs with")
    print("  python fp32_probe.py --combine <log> <log>")


if __name__ == "__main__":
    main()
