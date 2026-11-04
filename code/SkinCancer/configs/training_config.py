import os

# CUDA_VISIBLE_DEVICES must be set before torch is imported
# precedence: HAM_GPU > the shell's CUDA_VISIBLE_DEVICES > 0
_gpu = os.environ.get("HAM_GPU")
if _gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _gpu
elif "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch

from SkinCancer.configs.data_config import PROJECT_ROOT

# where checkpoints, predictions and CI outputs are saved (default: <root>/outputs)
OUTPUT_ROOT = os.environ.get("HAM_OUTPUT_ROOT", os.path.join(PROJECT_ROOT, "outputs"))
os.makedirs(OUTPUT_ROOT, exist_ok=True)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
ACCUMULATION_STEPS = 4
EPOCHS_STAGE1 = 20
LEARNING_RATE = 5e-5
WEIGHT_DECAY = float(os.environ.get("HAM_WEIGHT_DECAY", 0.05))  # NOTE: Supplementary Table S1 says 5e-5
SEED = 42

BOOTSTRAP_N = 10000
