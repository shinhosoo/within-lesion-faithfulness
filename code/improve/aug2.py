"""Lesion-based training augmentation: lcrop (random-scale crop centered on the lesion), bgdim (background blur and dim)."""
import argparse
import os
import random

import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

from SkinCancer.configs.model_config import BACKBONE_NAME, IMG_SIZE
from SkinCancer.configs.training_config import (
    ACCUMULATION_STEPS, BATCH_SIZE, DEVICE, EPOCHS_STAGE1, LEARNING_RATE, OUTPUT_ROOT, WEIGHT_DECAY,
)
from SkinCancer.data.split_io import build_image_path_map
from SkinCancer.utils.seed import set_seed
from improve.residual import get_data, inv_freq, macro_auc
from improve.zoom import MASK_DIR, square_box

OUT = os.path.join(OUTPUT_ROOT, "residual")
NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


class LesionAug(Dataset):
    def __init__(self, df, aug, train, p=0.7):
        self.p_ = df["image_path"].tolist(); self.y = df["label"].astype(int).tolist()
        self.ids = df["image_id"].tolist(); self.aug, self.train, self.p = aug, train, p

    def __len__(self):
        return len(self.p_)

    def bbox(self, iid, W, H):
        m = np.array(Image.open(os.path.join(MASK_DIR, f"{iid}_segmentation.png")).convert("L")) > 127
        ys, xs = np.where(m)
        return ((xs.min(), ys.min(), xs.max(), ys.max()) if len(xs) else (0, 0, W, H)), m

    def __getitem__(self, i):
        im = Image.open(self.p_[i]).convert("RGB")
        W, H = im.size
        if self.train and random.random() < self.p:
            bb, m = self.bbox(self.ids[i], W, H)
            if self.aug == "lcrop":
                im = im.crop(square_box(*bb, W, H, margin=random.uniform(0.2, 1.5)))
            else:                                    # bgdim: blur and darken outside the lesion
                mk = Image.fromarray((m * 255).astype(np.uint8)).resize(im.size, Image.NEAREST)
                bg = im.filter(ImageFilter.GaussianBlur(random.uniform(2, 6)))
                bg = TF.adjust_brightness(bg, random.uniform(0.5, 0.9))
                im = Image.composite(im, bg, mk)
        im = im.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        if self.train:
            if random.random() < 0.5:
                im = TF.hflip(im)
            if random.random() < 0.2:
                im = TF.vflip(im)
            im = TF.rotate(im, random.uniform(-30, 30), interpolation=TF.InterpolationMode.BILINEAR)
            im = transforms.ColorJitter(brightness=0.1, contrast=0.1)(im)
        return NORM(TF.to_tensor(im)), self.y[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aug", required=True, choices=["lcrop", "bgdim"])
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=EPOCHS_STAGE1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-pretrained", action="store_true")
    a = ap.parse_args()
    set_seed(a.seed)
    tr, va, _ = get_data(a.fold, build_image_path_map())
    if a.limit:
        tr = tr.groupby("label").head(a.limit // 7 + 1).reset_index(drop=True)
    y = tr["label"].astype(int).values
    ce = nn.CrossEntropyLoss(weight=inv_freq(np.bincount(y, minlength=7)))
    model = timm.create_model(BACKBONE_NAME, pretrained=not a.no_pretrained, num_classes=7).to(DEVICE)
    if hasattr(model, "set_grad_checkpointing"):
        model.set_grad_checkpointing(True)
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
    tl = DataLoader(LesionAug(tr, a.aug, True), batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    vl = DataLoader(LesionAug(va, a.aug, False), batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    print(f"[aug2:{a.aug}_f{a.fold}_s{a.seed}] train={len(tr)} val={len(va)} (no augmentation on val)")

    ck, best = os.path.join(OUT, f"{a.aug}_f{a.fold}_s{a.seed}.pth"), -1.0
    for ep in range(a.epochs):
        model.train(); tot = 0.0
        opt.zero_grad(set_to_none=True)
        for i, (x, lbl) in enumerate(tl):
            with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                loss = ce(model(x.to(DEVICE)), lbl.to(DEVICE).long())
            scaler.scale(loss / ACCUMULATION_STEPS).backward()
            if (i + 1) % ACCUMULATION_STEPS == 0:
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            tot += float(loss)
        model.eval(); P, Y = [], []
        with torch.no_grad():
            for x, lbl in vl:
                with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                    P.append(torch.softmax(model(x.to(DEVICE)).float(), 1).cpu().numpy())
                Y.append(np.asarray(lbl))
        P, Y = np.concatenate(P), np.concatenate(Y).astype(int)
        auc = macro_auc(P, Y)
        print(f"Epoch {ep+1:02d}/{a.epochs} | Loss {tot/max(len(tl),1):.4f} | Val AUC {auc:.4f} "
              f"Acc {(P.argmax(1)==Y).mean():.4f}")
        if auc > best:
            best = auc; torch.save(model.state_dict(), ck)
    model.load_state_dict(torch.load(ck, map_location=DEVICE, weights_only=True), strict=True)
    model.eval(); P, Y = [], []
    with torch.no_grad():
        for x, lbl in vl:
            with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                P.append(torch.softmax(model(x.to(DEVICE)).float(), 1).cpu().numpy())
            Y.append(np.asarray(lbl))
    P, Y = np.concatenate(P), np.concatenate(Y).astype(int)
    f = os.path.join(OUT, f"{a.aug}_f{a.fold}_s{a.seed}_val_predictions.npz")
    np.savez(f, y_prob=P, y_true=Y, image_id=va["image_id"].values)
    print(f"[aug2:{a.aug}] fold {a.fold} val acc={(P.argmax(1)==Y).mean():.4f} saved: {f}")


if __name__ == "__main__":
    main()
