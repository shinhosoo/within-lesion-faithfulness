"""Threshold-free pixel-level agreement between a saliency map and the lesion mask.

    python -m experiments.pixel_ap --fold 0
    for K in 0 1 2 3 4; do python -m experiments.pixel_ap --fold $K; done
"""

import argparse
import os

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from SkinCancer.configs.model_config import IMG_SIZE
from SkinCancer.configs.training_config import DEVICE
from SkinCancer.data.loaders import make_loader
from SkinCancer.data.split_io import build_image_path_map
from SkinCancer.data.transforms import build_val_transform



def ap(y, s):
    """Average precision with proper tie handling."""
    o = np.argsort(-s, kind="mergesort")
    y = y[o].astype(np.int64)
    s = s[o]
    npos = int(y.sum())
    if npos == 0 or npos == len(y):
        return float("nan")
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    last = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]
    tp, fp = tp[last], fp[last]
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / npos
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


