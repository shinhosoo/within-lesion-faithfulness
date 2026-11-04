"""Re-input of a zoomed lesion region (inference only). Compares base / rand / center /
maskbox, the last from the ground-truth lesion box."""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as Fn
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm import tqdm

from SkinCancer.configs.model_config import IMG_SIZE
from SkinCancer.configs.training_config import BATCH_SIZE, DEVICE, OUTPUT_ROOT
from SkinCancer.data.split_io import build_image_path_map
from SkinCancer.utils.seed import set_seed
from improve.localize import load_flat
from improve.residual import get_data

OUT = os.path.join(OUTPUT_ROOT, "zoom")
os.makedirs(OUT, exist_ok=True)
MASK_DIR = os.environ.get("HAM_MASK_DIR", "masks_ham10000")
NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
COND = ["base", "rand", "center", "maskbox"]


def to_tensor(img):
    return NORM(TF.to_tensor(img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)))


def square_box(x0, y0, x1, y1, W, H, margin=0.2):
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    s = max(x1 - x0, y1 - y0) * (1 + margin)
    s = max(s, 32)
    x0, y0 = max(0, cx - s / 2), max(0, cy - s / 2)
    x1, y1 = min(W, cx + s / 2), min(H, cy + s / 2)
    return int(x0), int(y0), int(max(x1, x0 + 8)), int(max(y1, y0 + 8))


def box_from_map(m, W, H, frac=0.5):
    """Bounding box of the top region of a map (square grid), in original image coordinates."""
    g = m.shape[0]
    sel = m >= frac * m.max()
    ys, xs = np.where(sel)
    if len(xs) == 0:
        return 0, 0, W, H
    return square_box(xs.min() / g * W, ys.min() / g * H, (xs.max() + 1) / g * W, (ys.max() + 1) / g * H, W, H)


@torch.no_grad()
def predict(net, batch):
    x = torch.stack(batch).to(DEVICE)
    with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
        f, _, _ = feats(net, x)
        z = net.forward_head(f)
    return torch.softmax(z.float(), 1).cpu().numpy()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    set_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    _, va, _ = get_data(a.fold, build_image_path_map())
    if a.limit:
        va = va.head(a.limit).reset_index(drop=True)
    net = load_flat(a.fold)
    print(f"[zoom] fold {a.fold}  n={len(va)}")

    P = {c: [] for c in COND}
    Y = []
    for st in tqdm(range(0, len(va), BATCH_SIZE), leave=False):
        rows = va.iloc[st:st + BATCH_SIZE]
        imgs = [Image.open(p).convert("RGB") for p in rows["image_path"]]
        base = [to_tensor(im) for im in imgs]
        crops = {c: [] for c in ["rand", "center", "maskbox"]}
        for i, im in enumerate(imgs):
            W, H = im.size
            s = min(W, H) * rng.uniform(0.6, 0.9)
            x0 = rng.uniform(0, W - s); y0 = rng.uniform(0, H - s)
            crops["rand"].append(to_tensor(im.crop((int(x0), int(y0), int(x0 + s), int(y0 + s)))))
            c = min(W, H) * 0.8
            crops["center"].append(to_tensor(im.crop(square_box((W - c) / 2, (H - c) / 2,
                                                                (W + c) / 2, (H + c) / 2, W, H, 0.0))))
            mp = os.path.join(MASK_DIR, f"{rows.iloc[i]['image_id']}_segmentation.png")
            m = np.array(Image.open(mp).convert("L")) > 127
            ys, xs = np.where(m)
            bb = square_box(xs.min(), ys.min(), xs.max(), ys.max(), W, H) if len(xs) else (0, 0, W, H)
            crops["maskbox"].append(to_tensor(im.crop(bb)))
        pb = predict(net, base)
        pc = {c: predict(net, v) for c, v in crops.items()}
        P["base"].append(pb)
        for c in ["rand", "center", "maskbox"]:
            P[c].append(0.5 * (pb + pc[c]))
        Y.append(rows["label"].astype(int).values)

    Y = np.concatenate(Y)
    f = os.path.join(OUT, f"zoom_f{a.fold}_s{a.seed}.npz")
    np.savez(f, y_true=Y, image_id=va["image_id"].values,
             **{c: np.concatenate(P[c]) for c in COND})
    print(f"  {'condition':<10}{'Acc':>9}{'vs base':>10}")
    b = np.concatenate(P["base"]).argmax(1)
    for c in COND:
        p = np.concatenate(P[c]).argmax(1)
        print(f"  {c:<10}{(p==Y).mean():>9.4f}{(p==Y).mean()-(b==Y).mean():>+10.4f}")
    print(f"  saved: {f}")


if __name__ == "__main__":
    main()
