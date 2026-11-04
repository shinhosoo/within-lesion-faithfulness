"""Zoom margin sweep, inference only. Finds how much background is best to keep."""
import argparse
import os

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from SkinCancer.configs.training_config import BATCH_SIZE, DEVICE, OUTPUT_ROOT
from SkinCancer.data.split_io import build_image_path_map
from SkinCancer.utils.seed import set_seed
from improve.localize import load_flat
from improve.residual import get_data
from improve.zoom import MASK_DIR, box_from_map, predict, square_box, to_tensor

OUT = os.path.join(OUTPUT_ROOT, "zoom")
os.makedirs(OUT, exist_ok=True)
MARGINS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--readout", default="sup0.2_bias_local")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    set_seed(a.seed)
    _, va, _ = get_data(a.fold, build_image_path_map())
    if a.limit:
        va = va.head(a.limit).reset_index(drop=True)
    net = load_flat(a.fold)
    br, s2_dim = build_sup(net, mode="bias", local=True)
    br.load_state_dict(torch.load(os.path.join(OUTPUT_ROOT, "localize",
                                               f"readout_{a.readout}_f{a.fold}.pth"),
                                  map_location=DEVICE, weights_only=True), strict=True)
    br.eval()
    keys = ["base"] + [f"{s}_m{int(m*100)}" for s in ("mask",) for m in MARGINS]
    P = {k: [] for k in keys}
    Y = []
    print(f"[zoom2] fold {a.fold} n={len(va)} margins={MARGINS}")

    for st in tqdm(range(0, len(va), BATCH_SIZE), leave=False):
        rows = va.iloc[st:st + BATCH_SIZE]
        imgs = [Image.open(p).convert("RGB") for p in rows["image_path"]]
        base = [to_tensor(im) for im in imgs]
        att = attention(net, br, s2_dim, base)
        pb = predict(net, base)
        P["base"].append(pb); Y.append(rows["label"].astype(int).values)
        boxes = []
        for i, im in enumerate(imgs):
            W, H = im.size
            mp = os.path.join(MASK_DIR, f"{rows.iloc[i]['image_id']}_segmentation.png")
            m = np.array(Image.open(mp).convert("L")) > 127
            ys, xs = np.where(m)
            mb = (xs.min(), ys.min(), xs.max(), ys.max()) if len(xs) else (0, 0, W, H)
            lb = box_from_map(att[i], W, H)      # box already includes a 20% margin, so only its center and size are used
            boxes.append((W, H, mb, lb))
        for src in ("mask",):
            for mg in MARGINS:
                crops = []
                for i, im in enumerate(imgs):
                    W, H, mb, lb = boxes[i]
                    x0, y0, x1, y1 = mb if src == "mask" else lb
                    crops.append(to_tensor(im.crop(square_box(x0, y0, x1, y1, W, H, margin=mg))))
                P[f"{src}_m{int(mg*100)}"].append(0.5 * (pb + predict(net, crops)))

    Y = np.concatenate(Y)
    f = os.path.join(OUT, f"zoom2_f{a.fold}_s{a.seed}.npz")
    np.savez(f, y_true=Y, image_id=va["image_id"].values, **{k: np.concatenate(v) for k, v in P.items()})
    b = np.concatenate(P["base"]).argmax(1)
    print(f"  {'condition':<14}{'Acc':>9}{'vs base':>10}")
    for k in keys:
        p = np.concatenate(P[k]).argmax(1)
        print(f"  {k:<14}{(p==Y).mean():>9.4f}{(p==Y).mean()-(b==Y).mean():>+10.4f}")
    print(f"  saved: {f}")


if __name__ == "__main__":
    main()
