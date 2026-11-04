"""Deletion/insertion test: measures whether a map points at the model real evidence (no extra training). Supports sup/supg readouts."""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as Fn
from PIL import Image
from tqdm import tqdm

from SkinCancer.configs.model_config import IMG_SIZE
from SkinCancer.configs.training_config import DEVICE, OUTPUT_ROOT
from SkinCancer.data.loaders import make_loader
from SkinCancer.data.split_io import build_image_path_map
from SkinCancer.data.transforms import build_val_transform
from SkinCancer.utils.seed import set_seed

OUT = os.path.join(OUTPUT_ROOT, "faithful")
os.makedirs(OUT, exist_ok=True)


def blur(x, k=51, s=20.0):
    c = torch.arange(k, device=x.device, dtype=x.dtype) - (k - 1) / 2
    g = torch.exp(-c ** 2 / (2 * s ** 2)); g = (g / g.sum()).view(1, 1, 1, k)
    y = Fn.conv2d(Fn.pad(x, (k // 2,) * 2 + (0, 0), mode="reflect"), g.expand(3, 1, 1, k), groups=3)
    return Fn.conv2d(Fn.pad(y, (0, 0) + (k // 2,) * 2, mode="reflect"),
                     g.transpose(2, 3).expand(3, 1, k, 1), groups=3)


@torch.no_grad()
def curve(net, img, base, maps, cls, steps, mode):
    """maps [B,H,W] -> per-step predicted probability [B, steps+1]"""
    B = img.shape[0]
    flat = maps.reshape(B, -1)
    order = flat.argsort(dim=1, descending=True)
    rank = torch.empty_like(order); ar = torch.arange(flat.shape[1], device=img.device)
    rank.scatter_(1, order, ar.expand(B, -1))
    out = []
    for i in range(steps + 1):
        n = int(flat.shape[1] * i / steps)
        sel = (rank < n).view(B, 1, IMG_SIZE, IMG_SIZE).float()
        x = img * (1 - sel) + base * sel if mode == "del" else base * (1 - sel) + img * sel
        with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
            p = torch.softmax(net.forward_head(net.forward_intermediates(x, intermediates_only=False)[0]).float(), 1)
        out.append(p.gather(1, cls[:, None]).squeeze(1).cpu())
    return torch.stack(out, 1).numpy()


