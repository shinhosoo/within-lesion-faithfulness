"""Train 1-stage once per seed. Does not overwrite outputs/onestage_7class.pth."""
import argparse, os
import numpy as np, timm, torch, torch.nn as nn, torch.optim as optim
from SkinCancer.configs.data_config import FINAL7_TEST_CSV, SPLIT_ROOT
from SkinCancer.configs.model_config import BACKBONE_NAME
from SkinCancer.configs.training_config import (
    ACCUMULATION_STEPS, DEVICE, EPOCHS_STAGE1, LEARNING_RATE, OUTPUT_ROOT, WEIGHT_DECAY)
from SkinCancer.data.split_io import build_image_path_map
from SkinCancer.data.transforms import build_train_transform, build_val_transform
from SkinCancer.utils.seed import set_seed
from ablation.onestage import load, macro_auc, make_loader, predict

OUT = os.path.join(OUTPUT_ROOT, "hier"); os.makedirs(OUT, exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=EPOCHS_STAGE1)
    a = ap.parse_args(); set_seed(a.seed)
    ckpt = os.path.join(OUT, f"flat_s{a.seed}.pth")
    pm = build_image_path_map()
    tr = load(os.path.join(SPLIT_ROOT, "global_train.csv"), pm)
    va = load(os.path.join(SPLIT_ROOT, "global_val.csv"), pm)
    te = load(FINAL7_TEST_CSV, pm)
    print(f"[flat s{a.seed}] train={len(tr)} val={len(va)} test={len(te)}")
    model = timm.create_model(BACKBONE_NAME, pretrained=True, num_classes=7).to(DEVICE)
    if hasattr(model, "set_grad_checkpointing"): model.set_grad_checkpointing(True)
    counts = tr["label"].value_counts().sort_index().values
    w = torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float32, device=DEVICE)
    criterion = nn.CrossEntropyLoss(weight=w)
    opt = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
    tl = make_loader(tr, build_train_transform(), True)
    vl = make_loader(va, build_val_transform(), False)
    best = -1.0
    for ep in range(a.epochs):
        model.train(); total = 0.0; opt.zero_grad(set_to_none=True)
        for i, (img, lbl) in enumerate(tl):
            with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                loss = criterion(model(img.to(DEVICE)), lbl.to(DEVICE).long())
            scaler.scale(loss / ACCUMULATION_STEPS).backward()
            if (i + 1) % ACCUMULATION_STEPS == 0:
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            total += loss.item()
        P, T = predict(model, vl); auc = macro_auc(P, T)
        print(f"Epoch {ep+1:02d}/{a.epochs} | Loss: {total/max(len(tl),1):.4f} | Val Macro-AUC: {auc:.4f}")
        if auc > best:
            best = auc; torch.save(model.state_dict(), ckpt)
            print(f"  -> checkpoint saved (val AUC {best:.4f})")
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True), strict=True)
    for split, df in [("val", va), ("test", te)]:
        P, T = predict(model, make_loader(df, build_val_transform(), False))
        f = os.path.join(OUT, f"flat_s{a.seed}_{split}_predictions.npz")
        np.savez(f, y_prob=P, y_true=T, image_id=df["image_id"].values)
        print(f"[flat s{a.seed}] {split} acc={(P.argmax(1)==T).mean():.4f} saved: {f}")

if __name__ == "__main__":
    main()
