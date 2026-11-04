import os

import numpy as np
import torch

# per-batch tensor logging is off by default; enable with export SKINCANCER_DEBUG=1
VERBOSE = os.environ.get("SKINCANCER_DEBUG", "0") == "1"


def debug_tensor(name, tensor):
    if not VERBOSE:
        return
    if isinstance(tensor, torch.Tensor):
        print(
            f"[Debug][Tensor] {name}: "
            f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device}"
        )
    elif isinstance(tensor, (list, tuple)):
        shapes = [
            tuple(t.shape) if isinstance(t, torch.Tensor) else type(t).__name__
            for t in tensor
        ]
        print(f"[Debug][Tensor] {name}: type={type(tensor).__name__}, items={shapes}")
    else:
        print(f"[Debug][Tensor] {name}: type={type(tensor).__name__}")


def debug_feature_map(name, tensor):
    if not VERBOSE:
        return
    if isinstance(tensor, torch.Tensor) and tensor.dim() == 4:
        print(
            f"[Debug][FeatureMap] {name}: "
            f"B={tensor.shape[0]}, C={tensor.shape[1]}, H={tensor.shape[2]}, W={tensor.shape[3]}"
        )
    else:
        debug_tensor(name, tensor)


def debug_dataframe(name, df, label_col=None):
    print(f"\n[Debug][DataFrame] {name}")
    print(f"  rows={len(df)}, columns={list(df.columns)}")
    if "image_path" in df.columns:
        existing = df["image_path"].map(lambda p: os.path.exists(p) if isinstance(p, str) else False).sum()
        print(f"  valid_image_paths={existing}/{len(df)}")
        print(f"  first_image_path={df['image_path'].iloc[0] if len(df) > 0 else 'N/A'}")
    if "diagnosis" in df.columns:
        print("  diagnosis_distribution:")
        print(df["diagnosis"].value_counts(dropna=False).sort_index().to_string())
    if label_col is not None and label_col in df.columns:
        print(f"  {label_col}_distribution:")
        print(df[label_col].value_counts(dropna=False).sort_index().to_string())


def debug_loader(name, loader):
    print(
        f"[Debug][Loader] {name}: "
        f"samples={len(loader.dataset)}, batches={len(loader)}, "
        f"batch_size={loader.batch_size}, num_workers={loader.num_workers}, "
        f"pin_memory={loader.pin_memory}"
    )


def get_current_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


def debug_model_summary(name, model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[Debug][Model] {name}: "
        f"total_params={total_params:,}, trainable_params={trainable_params:,}"
    )


def debug_class_weights(name, weights, class_names=None):
    weights_np = weights.detach().cpu().numpy() if isinstance(weights, torch.Tensor) else np.asarray(weights)
    print(f"[Debug][ClassWeights] {name}")
    for i, w in enumerate(weights_np.flatten()):
        cls = class_names[i] if class_names is not None and i < len(class_names) else str(i)
        print(f"  {cls}: {float(w):.6f}")


def debug_cuda_memory(prefix):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        print(f"[Debug][CUDA] {prefix}: allocated={allocated:.2f}MB, reserved={reserved:.2f}MB")
