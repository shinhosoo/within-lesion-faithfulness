"""The single-stage seven-class baseline.

    python -m ablation.onestage # train and evaluate
    python -m ablation.onestage --eval-only # evaluate an existing checkpoint
"""

import argparse
import os

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from SkinCancer.configs.data_config import (
    FINAL7_TEST_CSV, FINAL_CLASS_NAMES, SPLIT_ROOT, STAGE2_FINAL_MAPPING,
)
from SkinCancer.configs.model_config import BACKBONE_NAME, IMG_SIZE
from SkinCancer.configs.training_config import (
    ACCUMULATION_STEPS, BATCH_SIZE, BOOTSTRAP_N, DEVICE, EPOCHS_STAGE1,
    LEARNING_RATE, OUTPUT_ROOT, SEED, WEIGHT_DECAY,
)
from SkinCancer.data.dataset import LocalSkinCancerDataset
from SkinCancer.data.split_io import build_image_path_map
from SkinCancer.data.transforms import build_train_transform, build_val_transform
from SkinCancer.eval.bootstrap import bootstrap_macro_ci
from SkinCancer.eval.metrics import calculate_detailed_multiclass_metrics
from SkinCancer.utils.seed import set_seed

CKPT = os.path.join(OUTPUT_ROOT, "onestage_7class.pth")
PREDS = os.path.join(OUTPUT_ROOT, "onestage_7class_predictions.npz")
CI = os.path.join(OUTPUT_ROOT, "onestage_7class_macro_95ci.csv")


def load(csv_path, path_map):
    df = pd.read_csv(csv_path)
    if "diagnosis" not in df.columns and "dx" in df.columns:
        df = df.rename(columns={"dx": "diagnosis"})
    df["diagnosis"] = df["diagnosis"].str.lower()
    df["label"] = df["diagnosis"].map(STAGE2_FINAL_MAPPING)
    df["image_path"] = df["image_id"].map(path_map)
    df = df.dropna(subset=["label", "image_path"]).reset_index(drop=True)
    return df


def make_loader(df, transform, shuffle):
    ds = LocalSkinCancerDataset(
        df["image_path"].tolist(), df["label"].astype(float).tolist(), transform
    )
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=4, pin_memory=True)


@torch.no_grad()
def predict(model, loader):
    model.eval()
    P, T = [], []
    for img, lbl in loader:
        with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
            out = model(img.to(DEVICE))
        P.extend(torch.softmax(out, 1).float().cpu().numpy())
        T.extend(lbl.numpy().flatten())
    return np.array(P), np.array(T).astype(int)


def macro_auc(P, T):
    from sklearn.metrics import roc_auc_score
    p = np.asarray(P, dtype=np.float64)
    p = p / p.sum(axis=1, keepdims=True)
    try:
        return roc_auc_score(T, p, multi_class="ovr", average="macro")
    except Exception:
        return float((p.argmax(1) == T).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--epochs", type=int, default=EPOCHS_STAGE1)
    args = ap.parse_args()

    set_seed(SEED)
    path_map = build_image_path_map()
    tr = load(os.path.join(SPLIT_ROOT, "global_train.csv"), path_map)
    va = load(os.path.join(SPLIT_ROOT, "global_val.csv"), path_map)
    te = load(FINAL7_TEST_CSV, path_map)
    print(f"[1-stage] train={len(tr)} val={len(va)} test={len(te)}")

    model = timm.create_model(BACKBONE_NAME, pretrained=not args.eval_only, num_classes=7).to(DEVICE)
    if hasattr(model, "set_grad_checkpointing"):
        model.set_grad_checkpointing(True)

    if not args.eval_only:
        counts = tr["label"].value_counts().sort_index().values
        w = torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float32, device=DEVICE)
        print("[1-stage] class_weights = " + ", ".join(
            f"{c}:{v:.3f}" for c, v in zip(FINAL_CLASS_NAMES, w.tolist())))
        criterion = nn.CrossEntropyLoss(weight=w)
        opt = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")

        tl = make_loader(tr, build_train_transform(), True)
        vl = make_loader(va, build_val_transform(), False)
        best = -1.0
        for ep in range(args.epochs):
            model.train()
            total = 0.0
            opt.zero_grad(set_to_none=True)
            for i, (img, lbl) in enumerate(tl):
                with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                    loss = criterion(model(img.to(DEVICE)), lbl.to(DEVICE).long())
                scaler.scale(loss / ACCUMULATION_STEPS).backward()
                if (i + 1) % ACCUMULATION_STEPS == 0:
                    scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                total += loss.item()
            P, T = predict(model, vl)
            auc = macro_auc(P, T)
            print(f"Epoch {ep+1:02d}/{args.epochs} | Loss: {total/max(len(tl),1):.4f} | Val Macro-AUC: {auc:.4f}")
            if auc > best:
                best = auc
                torch.save(model.state_dict(), CKPT)
                print(f"  -> checkpoint saved (val AUC {best:.4f})")

    model.load_state_dict(torch.load(CKPT, map_location=DEVICE, weights_only=True), strict=True)
    P, T = predict(model, make_loader(te, build_val_transform(), False))
    np.savez(PREDS, y_prob=P, y_true=T, y_pred=P.argmax(1), image_id=te["image_id"].values)

    overall, per_class = calculate_detailed_multiclass_metrics(T, P, FINAL_CLASS_NAMES)
    print("\n" + "=" * 62)
    print("   single-stage seven-class baseline")
    print("=" * 62)
    for k in ["Accuracy", "Sensitivity", "Specificity", "AUC", "F1", "PPV", "NPV"]:
        print(f"   {k:<14} | {overall[k]:.4f}")
    print("\n   [Per-Class]")
    for c in FINAL_CLASS_NAMES:
        m = per_class[c]
        print(f"   {c.upper():<6} prec={m['PPV']*100:6.2f}%  rec={m['Sensitivity']*100:6.2f}%  "
              f"f1={m['F1']*100:6.2f}%  n={m['Support']}")

    ci = bootstrap_macro_ci(T, P, FINAL_CLASS_NAMES, n_boot=BOOTSTRAP_N, seed=SEED)
    ci.to_csv(CI, index=False)
    print(f"\n   wrote {PREDS}")


if __name__ == "__main__":
    main()
