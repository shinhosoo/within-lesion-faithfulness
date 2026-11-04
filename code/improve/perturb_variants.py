"""Does the ranking of explanation maps depend on how the perturbation baseline is built?

    python -m improve.perturb_variants --fold 0
    python -m improve.perturb_variants --fold 0 --baselines blur20,blur5,mean --n 100
    python -m improve.perturb_variants --fold 0 --steps-list 5,10,20

Insertion and deletion both replace part of the image with a baseline, and the published
protocol here uses one baseline: a Gaussian blur with k = 51 and sigma = 20, from
improve.faithful.blur. That choice is a free parameter of the measure, not a property of the
maps, and a measure whose verdict changes with it says less than it appears to. This script
holds everything else fixed and varies the baseline, so the only question it answers is
whether the ordering of the maps survives the choice.

What it does. For each batch it builds the reference maps once -- Grad-CAM at four depths,
Integrated Gradients, the lesion mask, the centered Gaussian, a random map -- and then runs
the published curve() and curve_within() against those same maps under each baseline in turn.
The maps are therefore bit-identical across baselines, so any movement in the scores comes
from the perturbation and from nothing else. Integrated Gradients is the one map that is
itself defined against a baseline; by default its path still starts from the published blur,
for the same reason. --ig-follows-baseline lifts that, and then the IG row measures a
different map under a different measure and is not comparable across baselines.

What it does not do. Agreement across baselines does not make the measure correct, and
disagreement does not make any one baseline right. Each baseline removes something different
-- blur removes texture and keeps layout, a constant colour removes both, noise replaces one
with another -- and a classifier may be sensitive to that difference for reasons that have
nothing to do with the explanation. The result to report is the rank correlation between
baselines, which answers whether the comparisons in this study depend on the choice, not
which baseline measures faithfulness.

Masks are required: curve_within ranks positions inside the lesion, so HAM_MASK_DIR must
point at the Tschandl segmentations. The script refuses to start without them.

Cost. Each map costs 4 * (steps + 1) forward passes per image per variant: two directions
over the whole image and two inside the lesion. The defaults -- eight maps, six baselines,
ten steps, 400 images -- are therefore about 845k forward passes for one fold, which is
larger than any single run in this study. The printed header gives the map and variant
counts before the loop starts, and --n, --maps and --baselines all cut it down; the
smallest useful run is the published blur against one alternative on a hundred images.
"""

import argparse
import os

import numpy as np
import torch

from SkinCancer.configs.model_config import IMG_SIZE
from SkinCancer.configs.training_config import DEVICE, OUTPUT_ROOT
from SkinCancer.data.loaders import make_loader
from SkinCancer.data.split_io import build_image_path_map
from SkinCancer.data.transforms import build_val_transform
from SkinCancer.utils.seed import set_seed

from improve.head_swap import MASK_DIR, center_map, curve_within, feats, load_flat, source_id, up
from improve.localize import gradcam, metrics
from improve.residual import eval_split
from improve.within_refs2 import LABEL, integrated_gradients

OUT = os.path.join(OUTPUT_ROOT, "headswap")
MAPS = ("gc1", "gc2", "gc3", "gc4", "ig", "mask", "center", "random")

# The published baseline. blur(x, k=51, s=20.0) in improve.faithful truncates the kernel at
# 1.25 sigma each side, which is where k = 51 comes from for sigma = 20. The blur variants
# below keep that ratio so that each one is the published kernel rescaled rather than a
# different kernel, and blur20 reproduces improve.faithful.blur exactly.
PUBLISHED = "blur20"
BASELINES = ("blur20", "blur5", "blur40", "mean", "imagenet", "noise")
BDESC = {
    "blur20": "Gaussian blur, sigma 20 (published)",
    "blur5": "Gaussian blur, sigma 5",
    "blur40": "Gaussian blur, sigma 40",
    "mean": "per-image channel mean",
    "imagenet": "ImageNet mean colour",
    "noise": "noise at each image's channel mean and sd",
}


def _blur_k(s):
    """The published truncation, 1.25 sigma each side. s = 20 gives k = 51."""
    return 2 * int(round(1.25 * s)) + 1


def build_baseline(name, x, gen):
    """The replacement image for one baseline. x is the batch in normalized space.

    Every tensor here lives after transforms.Normalize, so "zero" is the ImageNet mean
    colour rather than black, and a per-channel mean is the mean of the normalized image.
    Working in that space is what improve.faithful.blur already does, and the blur is linear,
    so blurring before or after normalization is the same operation.
    """
    from improve.faithful import blur

    if name == "blur20":
        return blur(x)
    if name == "blur5":
        return blur(x, k=_blur_k(5.0), s=5.0)
    if name == "blur40":
        return blur(x, k=_blur_k(40.0), s=40.0)
    if name == "mean":
        return x.mean(dim=(2, 3), keepdim=True).expand_as(x).contiguous()
    if name == "imagenet":
        return torch.zeros_like(x)
    if name == "noise":
        mu = x.mean(dim=(2, 3), keepdim=True)
        sd = x.std(dim=(2, 3), keepdim=True)
        return mu + sd * torch.randn(x.shape, generator=gen, device=x.device, dtype=x.dtype)
    raise SystemExit(f"unknown baseline {name!r}; choose from {', '.join(BASELINES)}")


def require_masks(n_min=1):
    """curve_within has nothing to rank without the lesion masks, so stop here rather than
    report a column of zeros that looks like a result."""
    if not MASK_DIR:
        raise SystemExit("HAM_MASK_DIR is unset. This script measures the within-lesion "
                         "ordering and cannot run without the Tschandl segmentations.")
    if not os.path.isdir(MASK_DIR):
        raise SystemExit(f"HAM_MASK_DIR points at {MASK_DIR!r}, which is not a directory. "
                         "Set it to the unpacked Tschandl segmentations.")
    found = sum(1 for f in os.listdir(MASK_DIR) if f.endswith("_segmentation.png"))
    if found < n_min:
        raise SystemExit(f"{MASK_DIR!r} holds {found} *_segmentation.png files. The "
                         "within-lesion measure needs the masks; nothing to run.")
    return found


def spearman(a, b):
    """Rank correlation with its two-sided p-value, over the maps measured in both columns."""
    from scipy.stats import spearmanr

    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan"), float("nan"), int(m.sum())
    r, p = spearmanr(a[m], b[m])
    return float(r), float(p), int(m.sum())


def run(a):
    from PIL import Image

    from improve.faithful import curve

    n_masks = require_masks()
    set_seed(a.seed)
    os.makedirs(OUT, exist_ok=True)
    ev = eval_split(a.fold, build_image_path_map()).head(a.n).reset_index(drop=True)
    net = load_flat(a.fold)
    cen = torch.tensor(center_map(), dtype=torch.float32, device=DEVICE)
    rng = np.random.default_rng(a.seed)
    gen = torch.Generator(device=DEVICE).manual_seed(a.seed + 20_000)
    names = a.maps
    variants = [(b, s) for b in a.baselines for s in a.steps_list]

    print(f"[perturb_variants] fold {a.fold}, {len(ev)} images, {n_masks} masks under {MASK_DIR}")
    print(f"[perturb_variants] {len(names)} maps x {len(variants)} variants "
          f"({len(a.baselines)} baselines x {len(a.steps_list)} step counts)")
    print(f"[perturb_variants] IG path baseline: "
          f"{'follows each variant' if a.ig_follows_baseline else 'held at the published blur'}")
    print()

    acc = {v: {k: {j: [] for j in ("del", "ins", "wdel", "wins")} for k in names}
           for v in variants}
    rows, seen, used, miss = [], 0, 0, 0

    store = {}

    def _mk(key):
        def _hook(mod, inp, out):
            if out.requires_grad:
                out.retain_grad()
                store[key] = out
        return _hook

    hooks = [net.stages[i].register_forward_hook(_mk(f"s{i + 1}")) for i in range(3)]
    for img, _ in make_loader(ev, "label", build_val_transform(), shuffle=False):
        x0 = img.to(DEVICE)
        B = x0.shape[0]
        xg = x0.clone().requires_grad_(True)
        with torch.enable_grad():
            f, _s2, _ = feats(net, xg)
            f.retain_grad()
            logits = net.forward_head(f)
            cls = logits.argmax(1)
            logits.gather(1, cls[:, None]).sum().backward()
            G = {"gc4": gradcam(f.detach(), f.grad, nhwc=True)}
            for i in range(3):
                h = store[f"s{i + 1}"]
                G[f"gc{i + 1}"] = gradcam(h.detach(), h.grad, nhwc=(h.shape[-1] != h.shape[-2]))
        net.zero_grad(set_to_none=True)
        cls = cls.detach()

        mk, ok = [], []
        for j in range(B):
            iid = ev.iloc[seen + j]["image_id"]
            mp = os.path.join(MASK_DIR, f"{iid}_segmentation.png")
            if os.path.exists(mp):
                arr = np.array(Image.open(mp).convert("L")
                               .resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)) > 127
            else:
                arr = np.zeros((IMG_SIZE, IMG_SIZE), bool)
                miss += 1
            mk.append(arr.astype(np.float32))
            ok.append(bool(0.0 < arr.mean() < 1.0))
        seen += B
        mask = torch.tensor(np.stack(mk), device=DEVICE)
        keep = np.array(ok)

        x = x0.detach()
        # The maps are built once and reused by every variant, so the scores below differ
        # only in the perturbation. The published blur is what IG starts from unless asked
        # otherwise, and it is also what the overlap row is computed from.
        pub = build_baseline(PUBLISHED, x, gen)
        M = {k: up(G[k]).detach() for k in ("gc1", "gc2", "gc3", "gc4")}
        M["ig"] = integrated_gradients(net, x, pub, cls, a.ig_steps).detach()
        M["mask"] = mask + 1e-6 * torch.rand_like(mask)
        M["center"] = cen.expand(B, -1, -1)
        M["random"] = torch.tensor(rng.random((B, IMG_SIZE, IMG_SIZE)),
                                   dtype=torch.float32, device=DEVICE)

        Mn = {k: M[k].cpu().numpy() for k in names}
        for j in range(B):
            rows.append(metrics(np.stack([Mn[k][j] for k in names]), mk[j] > 0.5))

        for bname in a.baselines:
            base = pub if bname == PUBLISHED else build_baseline(bname, x, gen)
            Mb = M
            if a.ig_follows_baseline and "ig" in names and bname != PUBLISHED:
                Mb = dict(M)
                Mb["ig"] = integrated_gradients(net, x, base, cls, a.ig_steps).detach()
            with torch.no_grad():
                for steps in a.steps_list:
                    v = (bname, steps)
                    for nm in names:
                        for mode in ("del", "ins"):
                            acc[v][nm][mode].append(
                                curve(net, x, base, Mb[nm], cls, steps, mode))
                        if keep.any():
                            w = curve_within(net, x, base, Mb[nm], mask, cls, steps)
                            acc[v][nm]["wdel"].append(w["del"][keep])
                            acc[v][nm]["wins"].append(w["ins"][keep])
        used += int(keep.sum())
    for h in hooks:
        h.remove()
    if miss:
        print(f"  [warning] {miss} masks not found under {MASK_DIR}")

    R = np.stack(rows)
    iou = np.array([float(np.nanmean(R[:, i, 2])) for i in range(len(names))])
    whole = np.full((len(variants), len(names)), np.nan)
    within = np.full((len(variants), len(names)), np.nan)
    for vi, v in enumerate(variants):
        for ni, nm in enumerate(names):
            d = {j: np.concatenate(acc[v][nm][j]).mean(1).mean() if acc[v][nm][j] else np.nan
                 for j in acc[v][nm]}
            whole[vi, ni] = d["ins"] - d["del"]
            within[vi, ni] = d["wins"] - d["wdel"]

    vlab = [f"{b}, {s} steps" for b, s in variants]
    # The value format below is fixed at nine significant characters, so the header has to be
    # at least as wide as the longest variant name or the columns run together.
    W = max(14, max(len(v) for v in vlab) + 2)
    print("    overlap is the same under every variant by construction, since the baseline")
    print(f"    enters the curves and not the maps. IoU over {len(ev)} images:")
    print()
    for ni, nm in enumerate(names):
        print(f"    {LABEL[nm]:<30}{iou[ni]:>9.4f}")
    print()
    for title, A in (("whole-image INS-DEL", whole), ("within-lesion INS-DEL", within)):
        print(f"    {title}")
        print(f"    {'map':<30}" + "".join(f"{s:>{W}}" for s in vlab))
        for ni, nm in enumerate(names):
            print(f"    {LABEL[nm]:<30}" + "".join(f"{A[vi, ni]:>+{W}.4f}"
                                                   for vi in range(len(variants))))
        print()

    # The column everything is compared against is the published protocol: the published
    # blur at ten steps when ten steps were asked for, and otherwise the published blur at
    # the first step count given.
    ref_steps = 10 if 10 in a.steps_list else a.steps_list[0]
    ref = variants.index((PUBLISHED, ref_steps))
    print(f"    rank correlation of the map ordering against {vlab[ref]}, over the "
          f"{len(names)} maps")
    print(f"    {'variant':<26}{'whole rho':>12}{'p':>10}{'within rho':>13}{'p':>10}")
    for vi in range(len(variants)):
        if vi == ref:
            continue
        rw, pw, _ = spearman(whole[ref], whole[vi])
        ri, pi, _ = spearman(within[ref], within[vi])
        print(f"    {vlab[vi]:<26}{rw:>+12.3f}{pw:>10.3f}{ri:>+13.3f}{pi:>10.3f}")
    print()
    print("    A high rho means the comparisons this study draws between maps do not depend")
    print("    on the baseline. It does not mean the baseline is the right one, and a low rho")
    print("    does not identify which baseline to prefer. Each one removes something")
    print("    different from the image, so the classifier may react to that difference for")
    print("    reasons unrelated to the maps.")
    print()

    path = os.path.join(OUT, f"perturb_variants_f{a.fold}.npz")
    if a.no_save:
        path = "(not saved: --no-save)"
    else:
        np.savez(path, names=np.array(names), baselines=np.array(a.baselines),
                 steps_list=np.array(a.steps_list),
                 variant_baseline=np.array([b for b, _ in variants]),
                 variant_steps=np.array([s for _, s in variants]),
                 whole=whole, within=within, iou=iou, metrics=R,
                 n=len(ev), n_within=used, ig_steps=a.ig_steps,
                 ig_follows_baseline=bool(a.ig_follows_baseline),
                 source_sha1=source_id())
    print(f"    {len(ev)} images, {used} with a usable lesion area")
    print(f"    produced by perturb_variants.py {source_id()}")
    print(f"    saved: {path}")


def _csv(valid, what):
    def parse(s):
        got = tuple(x.strip() for x in s.split(",") if x.strip())
        bad = [x for x in got if x not in valid]
        if bad:
            raise SystemExit(f"--{what}: {', '.join(bad)} not available. "
                             f"Choose from {', '.join(valid)}.")
        return got
    return parse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--baselines", type=_csv(BASELINES, "baselines"), default=BASELINES,
                    help="comma-separated; " + "; ".join(f"{k} = {v}" for k, v in BDESC.items()))
    ap.add_argument("--maps", type=_csv(MAPS, "maps"), default=MAPS,
                    help="comma-separated reference maps to measure")
    ap.add_argument("--steps-list", type=lambda s: tuple(int(x) for x in s.split(",") if x),
                    default=(10,), help="step counts to measure, comma-separated; the "
                                        "published protocol uses 10")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=400,
                    help="images per fold, as improve.faithful and improve.within_refs2 use")
    ap.add_argument("--ig-steps", type=int, default=32, help="path steps for Integrated Gradients")
    ap.add_argument("--ig-follows-baseline", action="store_true",
                    help="rebuild Integrated Gradients against each baseline instead of "
                         "holding its path at the published blur; the IG row is then not "
                         "comparable across baselines, because both the map and the measure "
                         "have changed")
    ap.add_argument("--no-save", action="store_true",
                    help="measure and print without writing the npz, for a smoke test")
    a = ap.parse_args()
    if PUBLISHED not in a.baselines:
        raise SystemExit(f"--baselines must include {PUBLISHED}, which is the published "
                         "protocol and the column every other variant is compared against.")
    if not a.steps_list:
        raise SystemExit("--steps-list is empty.")
    run(a)


if __name__ == "__main__":
    main()
