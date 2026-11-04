"""Threshold-free pixel-level agreement, for a saved explanation head.

    python -m improve.pixel_ap_head --fold 0 --head s3_mlp
    python -m improve.pixel_ap_head --fold 0 --head s3_mlp_sup0.2
    python -m improve.pixel_ap_head --fold 0 --head mlp --stage 2
"""

import argparse
import os

import numpy as np
import torch
from PIL import Image

from SkinCancer.configs.model_config import IMG_SIZE
from SkinCancer.configs.training_config import DEVICE
from SkinCancer.data.loaders import make_loader
from SkinCancer.data.split_io import build_image_path_map
from SkinCancer.data.transforms import build_val_transform
from SkinCancer.utils.seed import set_seed

from improve.deep_head import StageView
from improve.head_swap import (MASK_DIR, OUT, build_head, center_map, feats, load_flat, probe,
                               source_id, up)
from improve.localize import gradcam
from improve.pixel_ap import ap
from improve.residual import eval_split


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--head", required=True,
                   help="the stem of a saved head, e.g. s3_mlp or s3_mlp_sup0.2")
    p.add_argument("--stage", type=int, default=0, choices=[0, 2, 3, 4],
                   help="which stage the head reads; 0 reads it from the stem's s<N>_ prefix")
    p.add_argument("--map", default=None, choices=["relu", "softmax"],
                   help="read the saved head's map as this quantity instead of the "
                        "default for its stem; the archive name carries the mode so "
                        "the original measurement is never overwritten")
    p.add_argument("--limit", type=int, default=0, help="first N images only, for a quick check")
    a = p.parse_args()
    set_seed(a.seed)

    stage = a.stage
    if stage == 0:
        stage = int(a.head[1]) if a.head.startswith("s") and a.head[1].isdigit() else 2
    # The map mode has to match the run that saved the checkpoint, as everywhere else: a
    map_explicit = a.map is not None
    mode = a.map or ("softmax" if "_sup" in a.head else "relu")
    name = a.head.split("_sup", 1)[0].split("_", 1)[-1] if a.head.startswith("s") else \
        a.head.split("_sup", 1)[0]

    ev = eval_split(a.fold, build_image_path_map())
    if a.limit:
        ev = ev.head(a.limit).reset_index(drop=True)

    raw = load_flat(a.fold)
    net = StageView(raw, stage) if stage != 2 else raw
    s_dim, final_dim, hw = probe(net)
    head = build_head(name, net, s_dim, final_dim, 7, 4, False, mode).to(DEVICE)
    ck = os.path.join(OUT, f"head_{a.head}_f{a.fold}.pth")
    if not os.path.exists(ck):
        raise SystemExit(f"no checkpoint at {ck}")
    head.load_state_dict(torch.load(ck, map_location=DEVICE, weights_only=True), strict=True)
    head.eval()
    print(f"[pixel_ap_head] fold {a.fold}  head {a.head}  stage {stage}  "
          f"map {s_dim}x{hw[0]}x{hw[1]}  mode {mode}")
    print(f"[pixel_ap_head] produced by pixel_ap_head.py {source_id()}")

    cen = center_map()
    names = ["head", "gc4", "gc2", "center"]
    rows, ids, areas, k = [], [], [], 0

    store = {}

    def _hook(mod, inp, out):
        if out.requires_grad:
            out.retain_grad()
            store["s2"] = out

    hook = raw.stages[1].register_forward_hook(_hook)
    for img, _lbl in make_loader(ev, "label", build_val_transform(), shuffle=False):
        x = img.to(DEVICE)
        xg = x.clone().requires_grad_(True)
        with torch.enable_grad():
            f, _s, _ = feats(raw, xg)
            f.retain_grad()
            z = raw.forward_head(f)
            z.gather(1, z.argmax(1)[:, None]).sum().backward()
            gc4 = gradcam(f.detach(), f.grad, nhwc=True)
            h2 = store["s2"]
            gc2 = gradcam(h2.detach(), h2.grad, nhwc=(h2.shape[-1] != h2.shape[-2]))
        raw.zero_grad(set_to_none=True)

        with torch.no_grad():
            f2, s2, _ = feats(net, x)
            _, sal = head(net.forward_head(f2, pre_logits=True).float(),
                          s2.float())
            hm = up(sal.float())
        maps = [hm.cpu().numpy(), up(gc4).cpu().numpy(), up(gc2).cpu().numpy()]
        for i in range(x.shape[0]):
            iid = ev.iloc[k]["image_id"]
            k += 1
            mp = os.path.join(MASK_DIR, f"{iid}_segmentation.png")
            mask = np.array(Image.open(mp).convert("L")
                            .resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)) > 127
            y = mask.reshape(-1)
            rows.append([ap(y, m[i].reshape(-1)) for m in maps] + [ap(y, cen.reshape(-1))])
            ids.append(iid)
            areas.append(float(mask.mean()))
    hook.remove()

    R = np.array(rows, dtype=np.float64)
    ar = np.array(areas)
    tag = (f"{a.head}"
           + (f"_map{a.map}" if map_explicit else "")
           + f"_f{a.fold}"
           + (f"_lim{a.limit}" if a.limit else ""))
    path = os.path.join(OUT, f"pixelap_{tag}.npz")
    np.savez(path, ap=R, names=np.array(names), image_id=np.array(ids), lesion_area=ar,
             stage=stage, source_sha1=source_id())

    print(f"\n[pixel_ap_head] n={len(ids)}  ->  {path}")
    print(f"{'map':<10}{'mean AP':>10}{'s.d.':>9}")
    for j, nm in enumerate(names):
        print(f"{nm:<10}{np.nanmean(R[:, j]):>10.4f}{np.nanstd(R[:, j]):>9.4f}")
    print(f"{'random':<10}{ar.mean():>10.4f}{ar.std():>9.4f}"
          f"   (analytic: a random map scores the lesion area fraction)")
    print("\n    The Grad-CAM and centered-Gaussian rows depend on the backbone and not on the")
    print("    head, so they must come out the same for every head of the same fold; that is")
    print("    the check that this path produces the quantity improve.pixel_ap produces.")


if __name__ == "__main__":
    main()
