"""The image, label and lesion-mask dataset the mask-supervised runs train on.

Masks enter the alignment term and the rule that chooses the epoch. They are not needed to
produce a map at inference. The supervised training itself lives in improve.head_swap.
"""
import os
import random

import numpy as np
import torch
import torch.nn.functional as Fn
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

from SkinCancer.configs.model_config import IMG_SIZE

MASK_DIR = os.environ.get("HAM_MASK_DIR", "masks_ham10000")
NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
JIT = transforms.ColorJitter(brightness=0.1, contrast=0.1)


class MaskSet(Dataset):
    """Image, label and 40x40 lesion mask, with the same geometric augmentation applied."""

    def __init__(self, df, train, grid=40):
        self.p = df["image_path"].tolist(); self.y = df["label"].astype(int).tolist()
        self.ids = df["image_id"].tolist(); self.train = train; self.g = grid

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        img = Image.open(self.p[i]).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        mp = os.path.join(MASK_DIR, f"{self.ids[i]}_segmentation.png")
        msk = Image.open(mp).convert("L").resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
        if self.train:
            if random.random() < 0.5:
                img, msk = TF.hflip(img), TF.hflip(msk)
            if random.random() < 0.2:
                img, msk = TF.vflip(img), TF.vflip(msk)
            ang = random.uniform(-30, 30)
            img = TF.rotate(img, ang, interpolation=TF.InterpolationMode.BILINEAR)
            msk = TF.rotate(msk, ang, interpolation=TF.InterpolationMode.NEAREST)
            img = JIT(img)
        x = NORM(TF.to_tensor(img))
        m = torch.tensor(np.array(msk) > 127, dtype=torch.float32)[None, None]
        m = Fn.adaptive_avg_pool2d(m, self.g)[0, 0].reshape(-1)
        return x, self.y[i], m


