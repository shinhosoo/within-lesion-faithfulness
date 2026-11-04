"""Overlap for the reference maps over the whole held-out fold, matching how every other

    python -m improve.refs_overlap --fold 0
    python -m improve.refs_overlap --fold 0 --ig-steps 32
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

from improve.head_swap import MASK_DIR, center_map, feats, load_flat, source_id, up
from improve.localize import gradcam, metrics
from improve.residual import eval_split
from improve.within_refs2 import LABEL, NAMES, integrated_gradients

OUT = os.path.join(OUTPUT_ROOT, "headswap")


def run(a):
    from PIL import Image

    from improve.faithful import blur

    set_seed(a.seed)
    os.makedirs(OUT, exist_ok=True)
    ev = eval_split(a.fold, build_image_path_map())
    if a.limit:
        ev = ev.head(a.limit).reset_index(drop=True)
    net = load_flat(a.fold)
    cen_np = center_map()
    rng = np.random.default_rng(a.seed)
    rows, seen, miss = [], 0, 0

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
                G[f"gc{i + 1}"] = gradcam(h.detach(), h.grad,
                                          nhwc=(h.shape[-1] != h.shape[-2]))
        net.zero_grad(set_to_none=True)
        cls = cls.detach()

        mk = []
        for j in range(B):
            iid = ev.iloc[seen + j]["image_id"]
            mp = os.path.join(MASK_DIR, f"{iid}_segmentation.png")
            if os.path.exists(mp):
                arr = np.array(Image.open(mp).convert("L")
                               .resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)) > 127
            else:
                arr = np.zeros((IMG_SIZE, IMG_SIZE), bool)
                miss += 1
            mk.append(arr)
        seen += B

        x = x0.detach()
        base = blur(x)
        mask_t = torch.tensor(np.stack(mk).astype(np.float32), device=DEVICE)
        M = {k: up(G[k]).detach() for k in ("gc1", "gc2", "gc3", "gc4")}
        M["ig"] = integrated_gradients(net, x, base, cls, a.ig_steps).detach()
        # The same jitter within_refs2.py uses, kept for consistency even though overlap is
        # indifferent to it: the area-matched threshold lands below the in-mask values either way.
        M["mask"] = mask_t + 1e-6 * torch.rand_like(mask_t)
        M["center"] = torch.tensor(cen_np, dtype=torch.float32,
                                   device=DEVICE).expand(B, -1, -1)
        M["random"] = torch.tensor(rng.random((B, IMG_SIZE, IMG_SIZE)),
                                   dtype=torch.float32, device=DEVICE)
        Mn = {k: M[k].cpu().numpy() for k in NAMES}
        for j in range(B):
            rows.append(metrics(np.stack([Mn[k][j] for k in NAMES]), mk[j]))
    for h in hooks:
        h.remove()
    if miss:
        print(f"  [warning] {miss} masks not found under {MASK_DIR}")

    R = np.stack(rows)
    ov = {nm: [float(np.nanmean(R[:, i, c])) for c in range(3)] for i, nm in enumerate(NAMES)}
    tag = f"refs_ov_f{a.fold}" + (f"_lim{a.limit}" if a.limit else "")
    path = os.path.join(OUT, f"{tag}.npz")
    np.savez(path, names=np.array(NAMES), metrics=R, n=seen,
             ig_steps=a.ig_steps, source_sha1=source_id())

    print(f"[refs_overlap] fold {a.fold}, {seen} images (the whole held-out fold)")
    print(f"[refs_overlap] produced by refs_overlap.py {source_id()}, IG {a.ig_steps} steps")
    print()
    print(f"    {'map':<24}{'Pointing':>10}{'Energy':>9}{'IoU':>9}")
    for nm in NAMES:
        print(f"    {LABEL[nm]:<24}{ov[nm][0]:>10.4f}{ov[nm][1]:>9.4f}{ov[nm][2]:>9.4f}")
    print()
    print("    The check: Grad-CAM at the second stage and the centered Gaussian are in Table 1,")
    print("    at an IoU of 0.201 and 0.656. On 400 images they read 0.207 and 0.659; computed")
    print("    over the whole fold they have to read the published values, and the lesion-mask")
    print("    row has to read 1.0000 on all three measures.")
    print(f"    saved: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ig-steps", type=int, default=32,
                    help="path steps for Integrated Gradients; must match the faithfulness run")
    ap.add_argument("--limit", type=int, default=0, help="smoke test on the first n images")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
