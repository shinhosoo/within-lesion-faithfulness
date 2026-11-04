"""Measure the reference maps on the within-lesion test: Grad-CAM, a random map, the Gaussian.

    python -m improve.within_refs --fold 0
    python -m improve.within_refs --fold 0 --n 400
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
from improve.localize import gradcam
from improve.residual import eval_split

OUT = os.path.join(OUTPUT_ROOT, "headswap")
NAMES = ("gc2", "gc4", "center", "random")


def run(a):
    from PIL import Image

    from improve.faithful import blur, curve

    set_seed(a.seed)
    os.makedirs(OUT, exist_ok=True)
    ev = eval_split(a.fold, build_image_path_map()).head(a.n).reset_index(drop=True)
    net = load_flat(a.fold)
    cen = torch.tensor(center_map(), dtype=torch.float32, device=DEVICE)
    rng = np.random.default_rng(a.seed)
    acc = {k: {j: [] for j in ("del", "ins", "wdel", "wins")} for k in NAMES}
    seen, used, miss = 0, 0, 0

    # Grad-CAM at stage 2 reads the hooked stage-1 output, exactly as improve.localize does, so
    # the maps measured here are the maps the paper's Table 1 row was computed from.
    store = {}

    def _hook(mod, inp, out):
        # The hook stays registered while curve() runs the backbone again under no_grad, where
        if out.requires_grad:
            out.retain_grad()
            store["s2"] = out

    hook = net.stages[1].register_forward_hook(_hook)
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
            g4 = gradcam(f.detach(), f.grad, nhwc=True)
            h2 = store["s2"]
            g2 = gradcam(h2.detach(), h2.grad, nhwc=(h2.shape[-1] != h2.shape[-2]))
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
        M = {"gc2": up(g2).detach(), "gc4": up(g4).detach(),
             "center": cen.expand(B, -1, -1),
             "random": torch.tensor(rng.random((B, IMG_SIZE, IMG_SIZE)),
                                    dtype=torch.float32, device=DEVICE)}
        with torch.no_grad():
            for nm in NAMES:
                for mode in ("del", "ins"):
                    acc[nm][mode].append(curve(net, x, base, M[nm], cls, a.steps, mode))
                if keep.any():
                    w = curve_within(net, x, base, M[nm], mask, cls, a.steps)
                    acc[nm]["wdel"].append(w["del"][keep])
                    acc[nm]["wins"].append(w["ins"][keep])
        used += int(keep.sum())
    hook.remove()
    if miss:
        print(f"  [warning] {miss} masks not found under {MASK_DIR}")

    out = {}
    for nm in NAMES:
        v = {j: np.concatenate(acc[nm][j]).mean(1).mean() for j in acc[nm]}
        out[nm] = (float(v["ins"] - v["del"]), float(v["wins"] - v["wdel"]))
    path = os.path.join(OUT, f"refs_f{a.fold}.npz")
    np.savez(path, names=np.array(NAMES),
             faith=np.array([out[n][0] for n in NAMES]),
             within=np.array([out[n][1] for n in NAMES]),
             n=a.n, n_within=used, source_sha1=source_id())

    print(f"[within_refs] fold {a.fold}, {seen} images, {used} with a usable lesion area")
    print(f"[within_refs] produced by within_refs.py {source_id()}")
    print()
    print(f"    {'map':<28}{'Ins-Del':>11}{'within':>11}")
    label = {"gc2": "Grad-CAM (stage-2)", "gc4": "Grad-CAM (stage-4)",
             "center": "Centered Gaussian", "random": "Random map"}
    for nm in NAMES:
        print(f"    {label[nm]:<28}{out[nm][0]:>+11.4f}{out[nm][1]:>+11.4f}")
    print()
    print("    The centered Gaussian is the check: it knows nothing about the model, so its")
    print("    within value has to sit near zero for the column to mean anything. Grad-CAM is")
    print("    the row that matters, being the method a reader already uses.")
    print(f"    saved: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=400, help="images per fold, as improve.faithful uses")
    ap.add_argument("--steps", type=int, default=10)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
