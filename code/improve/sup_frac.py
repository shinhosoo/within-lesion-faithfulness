"""How many masks are needed: attention supervision on only a fraction of the training samples.

    python -m improve.sup_frac --stage train --fold 0 --frac 0.25 --lam 0.2
    python -m improve.sup_frac --stage eval --fold 0 --frac 0.25
"""
import argparse
import hashlib
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

import improve.localize_sup as S
from SkinCancer.configs.training_config import BATCH_SIZE, DEVICE
from SkinCancer.data.split_io import build_image_path_map
from SkinCancer.utils.seed import set_seed
from improve.residual import get_data, inv_freq


class MaskSetFrac(S.MaskSet):
    """Returns (x, y, m, keep). Only samples with keep=1 take part in the attention loss."""

    def __init__(self, df, train, frac, grid=40):
        super().__init__(df, train, grid)
        self.keep = [int(hashlib.md5(str(i).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < frac
                     for i in self.ids]

    def __getitem__(self, i):
        x, y, m = super().__getitem__(i)
        return x, y, m, float(self.keep[i])


