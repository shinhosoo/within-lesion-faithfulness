"""Shared helpers for the localization measurements: the frozen fold backbone, stage-2
features, Grad-CAM, upsampling, the overlap metrics and the centered Gaussian reference."""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from PIL import Image
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from SkinCancer.configs.model_config import IMG_SIZE
from SkinCancer.configs.training_config import DEVICE, OUTPUT_ROOT
from SkinCancer.data.loaders import make_loader
from SkinCancer.data.split_io import build_image_path_map
from SkinCancer.data.transforms import build_train_transform, build_val_transform
from SkinCancer.utils.seed import set_seed
from improve.residual import eval_split, get_data, inv_freq

OUT = os.path.join(OUTPUT_ROOT, "localize")
os.makedirs(OUT, exist_ok=True)
MASK_DIR = os.environ.get("HAM_MASK_DIR", "masks_ham10000")


def load_flat(fold):
    m = ResidualModel("flat", pretrained=False).to(DEVICE)
    m.load_state_dict(torch.load(os.path.join(OUTPUT_ROOT, "residual", f"flat_f{fold}_s42.pth"),
                                 map_location=DEVICE, weights_only=True), strict=True)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m.net


def feats(net, x):
    f, st = net.forward_intermediates(x, intermediates_only=False)
    s2 = st[1][0] if isinstance(st[1], (list, tuple)) else st[1]
    return f, s2, st


def gradcam(act, grad, nhwc):
    if nhwc:
        w = grad.mean(dim=(1, 2)); cam = (act * w[:, None, None, :]).sum(-1)
    else:
        w = grad.mean(dim=(2, 3)); cam = (act * w[:, :, None, None]).sum(1)
    return Fn.relu(cam)


def up(m):
    return Fn.interpolate(m[:, None].float(), size=(IMG_SIZE, IMG_SIZE), mode="bilinear",
                          align_corners=False)[:, 0]


def metrics(maps, mask):
    out = []
    area = mask.mean()
    for m in maps:
        m = m - m.min()
        flat = m.reshape(-1)
        point = float(mask.reshape(-1)[flat.argmax()])
        energy = float((m * mask).sum() / max(m.sum(), 1e-12))
        if area <= 0 or area >= 1:
            iou = np.nan
        else:
            thr = np.quantile(flat, 1 - area)
            sel = m >= thr
            iou = float((sel & mask).sum() / max((sel | mask).sum(), 1))
        out.append((point, energy, iou))
    return np.array(out)


def center_map():
    yy, xx = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]
    c, s = (IMG_SIZE - 1) / 2, IMG_SIZE * 0.25
    return np.exp(-((xx - c) ** 2 + (yy - c) ** 2) / (2 * s ** 2))


