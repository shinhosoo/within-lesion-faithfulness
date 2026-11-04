"""More reference maps on the within-lesion test: Grad-CAM at four depths, Integrated

    python -m improve.within_refs2 --fold 0
    python -m improve.within_refs2 --fold 0 --ig-steps 16
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

OUT = os.path.join(OUTPUT_ROOT, "headswap")
BASE_NAMES = ("gc1", "gc2", "gc3", "gc4", "ig", "mask", "center", "random")
NAMES = BASE_NAMES
LABEL = {"gc1": "Grad-CAM (stage-1)", "gc2": "Grad-CAM (stage-2)",
         "gc3": "Grad-CAM (stage-3)", "gc4": "Grad-CAM (stage-4)",
         "ig": "Integrated Gradients", "mask": "Lesion mask as map",
         "center": "Centered Gaussian", "random": "Random map"}

# The maps worth shuffling are the ones that score above the centered Gaussian on the
SHUF_DEFAULT = ("gc2", "gc3", "gc4", "ig", "center")
for _k in SHUF_DEFAULT:
    LABEL[_k + "_shuf"] = LABEL[_k] + ", shuffled inside the lesion"


def shuffle_within(m, msk, rng):
    """Permute each image's map values among the positions inside its lesion."""
    import torch as _t
    out = m.clone()
    for j in range(m.shape[0]):
        sel = msk[j] > 0.5
        n = int(sel.sum().item())
        if n < 2:
            continue
        vals = m[j][sel]
        idx = _t.as_tensor(rng.permutation(n), dtype=_t.long, device=vals.device)
        out[j][sel] = vals[idx]
    return out


def integrated_gradients(net, x, base, cls, steps):
    """IG for the classifier's predicted class, along a straight path from the blurred image."""
    total = torch.zeros_like(x)
    for i in range(steps):
        a = (i + 0.5) / steps
        xi = (base + a * (x - base)).detach().requires_grad_(True)
        with torch.enable_grad():
            f, _s2, _ = feats(net, xi)
            z = net.forward_head(f)
            z.gather(1, cls[:, None]).sum().backward()
        total += xi.grad.detach()
        net.zero_grad(set_to_none=True)
    return torch.relu(((x - base) * total / steps).sum(1))


def run(a):
    from PIL import Image

    from improve.faithful import blur, curve

    set_seed(a.seed)
    os.makedirs(OUT, exist_ok=True)
    ev = eval_split(a.fold, build_image_path_map()).head(a.n).reset_index(drop=True)
    net = load_flat(a.fold)
    cen = torch.tensor(center_map(), dtype=torch.float32, device=DEVICE)
    cen_np = center_map()
    rng = np.random.default_rng(a.seed)
    # Assigning NAMES here makes it local to run() for the whole function, which is why the
    # global is read through BASE_NAMES rather than by this name.
    NAMES = (BASE_NAMES + tuple(k + "_shuf" for k in a.shuffle)) if a.shuffle else BASE_NAMES
    # The permutations draw from their own generator, not from rng. rng also builds the "random"
    srng = np.random.default_rng(a.seed + 10_000)
    acc = {k: {j: [] for j in ("del", "ins", "wdel", "wins")} for k in NAMES}
    rows, seen, used, miss = [], 0, 0, 0

    # One hook per intermediate stage. The guard matters: the hooks stay registered while curve()
    store = {}

    def _mk(key):
        def _hook(mod, inp, out):
            if out.requires_grad:
                out.retain_grad()
                store[key] = out
        return _hook

    hooks = [net.stages[i].register_forward_hook(_mk(f"s{i + 1}")) for i in range(3)]
    loader = make_loader(ev, "label", build_val_transform(), shuffle=False)
    for img, _ in loader:
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
                G[f"gc{i + 1}"] = gradcam(h.detach(), h.grad,
                                          nhwc=(h.shape[-1] != h.shape[-2]))
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
        base = blur(x)
        ig = integrated_gradients(net, x, base, cls, a.ig_steps)
        M = {k: up(G[k]).detach() for k in ("gc1", "gc2", "gc3", "gc4")}
        M["ig"] = ig.detach()
        # The mask is uniform inside the lesion, so the within-lesion ordering it implies is
        M["mask"] = mask + 1e-6 * torch.rand_like(mask)
        M["center"] = cen.expand(B, -1, -1)
        M["random"] = torch.tensor(rng.random((B, IMG_SIZE, IMG_SIZE)),
                                   dtype=torch.float32, device=DEVICE)
        for _k in (a.shuffle or ()):
            M[_k + "_shuf"] = shuffle_within(M[_k], mask, srng)

        Mn = {k: M[k].cpu().numpy() for k in NAMES}
        for j in range(B):
            rows.append(metrics(np.stack([Mn[k][j] for k in NAMES]), mk[j] > 0.5))

        with torch.no_grad():
            for nm in NAMES:
                for mode in ("del", "ins"):
                    acc[nm][mode].append(curve(net, x, base, M[nm], cls, a.steps, mode))
                if keep.any():
                    w = curve_within(net, x, base, M[nm], mask, cls, a.steps)
                    acc[nm]["wdel"].append(w["del"][keep])
                    acc[nm]["wins"].append(w["ins"][keep])
        used += int(keep.sum())
    for h in hooks:
        h.remove()
    if miss:
        print(f"  [warning] {miss} masks not found under {MASK_DIR}")

    R = np.stack(rows)
    out = {}
    for nm in NAMES:
        v = {j: np.concatenate(acc[nm][j]).mean(1).mean() for j in acc[nm]}
        out[nm] = (float(v["ins"] - v["del"]), float(v["wins"] - v["wdel"]))
    iou = {nm: float(np.nanmean(R[:, i, 2])) for i, nm in enumerate(NAMES)}
    path = os.path.join(OUT, f"refs2_shuf_f{a.fold}.npz" if a.shuffle
                        else f"refs2_f{a.fold}.npz")
    if a.no_save:
        # A smoke test runs on a handful of images and must not leave a file behind: the queue
        path = "(not saved: --no-save)"
    else:
        np.savez(path, names=np.array(NAMES), metrics=R,
                 faith=np.array([out[n][0] for n in NAMES]),
                 within=np.array([out[n][1] for n in NAMES]),
                 iou=np.array([iou[n] for n in NAMES]),
                 n=a.n, n_within=used, ig_steps=a.ig_steps, source_sha1=source_id())

    print(f"[within_refs2] fold {a.fold}, {seen} images, {used} with a usable lesion area")
    print(f"[within_refs2] produced by within_refs2.py {source_id()}, IG {a.ig_steps} steps")
    print()
    print(f"    {'map':<24}{'IoU':>9}{'Ins-Del':>11}{'within':>11}")
    for nm in NAMES:
        print(f"    {LABEL[nm]:<24}{iou[nm]:>9.4f}{out[nm][0]:>+11.4f}{out[nm][1]:>+11.4f}")
    print()
    if a.shuffle:
        print("    The shuffle control. Permuting a map's values among the positions inside the")
        print("    lesion leaves the overlap measures arithmetically unchanged and destroys the")
        print("    within-lesion ordering, so energy and IoU below must be zero to machine")
        print("    precision. A non-zero energy or IoU difference is a bug in the shuffle, not a")
        print("    result. Pointing may differ on a handful of images where a plateau ties the")
        print("    global maximum across the lesion boundary.")
        print()
        print(f"    {'map':<26}{'d pointing':>12}{'d energy':>11}{'d IoU':>10}"
              f"{'within':>10}{'-> shuffled':>13}")
        # Pointing and IoU must be bit-identical: pointing asks whether the arg max lies
        EPS32 = float(np.finfo(np.float32).eps)
        E_TOL = 100 * EPS32
        hard = soft = 0.0
        for _k in a.shuffle:
            i0, i1 = NAMES.index(_k), NAMES.index(_k + "_shuf")
            d = [float(np.nanmax(np.abs(R[:, i0, c] - R[:, i1, c]))) for c in range(3)]
            nflip = int(np.nansum(R[:, i0, 0] != R[:, i1, 0]))
            hard = max(hard, d[0], d[2])
            soft = max(soft, d[1])
            print(f"    {LABEL[_k]:<26}{d[0]:>12.2e}{d[1]:>11.2e}{d[2]:>10.2e}"
                  f"{out[_k][1]:>+10.4f}{out[_k + '_shuf'][1]:>+13.4f}"
                  + (f"   ({nflip} pointing flips)" if nflip else ""))
        print()
        if hard == 0.0 and soft < E_TOL:
            print(f"    pointing and IoU are bit-identical and energy agrees to "
                  f"{soft / EPS32:.1f} float32 epsilon,")
            print("    which is the reordering of the sum and not a property of the maps. The")
            print("    overlap measures are blind to the within-lesion ordering by construction,")
            print("    so the within column above is measuring something they cannot see.")
        elif hard != 0.0:
            print(f"    FAILED: pointing or IoU moved by {hard:.2e}. Both are exact under an")
            print("    in-lesion permutation, so this is a bug in the shuffle, not a result.")
        else:
            print(f"    FAILED: energy moved by {soft:.2e}, which is {soft / EPS32:.0f} float32")
            print("    epsilon and too large for summation order alone. Check that the")
            print("    permutation stays inside the mask.")
        print()
    print("    Checks, in order. The centered Gaussian must read +0.2038 and +0.0165 on fold 0,")
    print("    as head_swap.py and within_refs.py both report. The lesion-mask row must read an")
    print("    IoU of 1.0000 and a within value near zero, which it cannot avoid: it is uniform")
    print("    inside the lesion, so it carries no ordering there. Only then are the Grad-CAM")
    print("    depths and the IG row worth reading.")
    print(f"    saved: {path}")


def _check_shuffle(names):
    bad = [k for k in names if k not in BASE_NAMES]
    if bad:
        raise SystemExit(f"--shuffle: {', '.join(bad)} is not a map this script builds. "
                         f"Choose from {', '.join(BASE_NAMES)}.")
    flat = [k for k in names if k in ("mask", "random")]
    if flat:
        raise SystemExit(f"--shuffle: {', '.join(flat)} carries no within-lesion ordering to "
                         f"destroy, so shuffling it measures nothing. Drop it.")
    return tuple(names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--shuffle", nargs="?", const=",".join(SHUF_DEFAULT), default=None,
                    type=lambda s: tuple(x for x in s.split(",") if x),
                    help="also measure each named map with its values permuted among the "
                         "positions inside the lesion; bare --shuffle uses "
                         + ", ".join(SHUF_DEFAULT))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=400, help="images per fold, as improve.faithful uses")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--ig-steps", type=int, default=32, help="path steps for Integrated Gradients")
    ap.add_argument("--no-save", action="store_true",
                    help="measure and print without writing the npz, for a smoke test")
    a = ap.parse_args()
    if a.shuffle:
        a.shuffle = _check_shuffle(a.shuffle)
    run(a)


if __name__ == "__main__":
    main()
