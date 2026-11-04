"""The single-stage classifier the explanation heads read from, trained under lesion-grouped 5-fold CV."""
import argparse, os
import numpy as np, pandas as pd, timm, torch, torch.nn as nn, torch.optim as optim
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from tqdm import tqdm
from SkinCancer.configs.data_config import FINAL7_TEST_CSV, SPLIT_ROOT
from SkinCancer.configs.model_config import BACKBONE_NAME
from SkinCancer.configs.training_config import ACCUMULATION_STEPS, DEVICE, EPOCHS_STAGE1, LEARNING_RATE, OUTPUT_ROOT, WEIGHT_DECAY
from SkinCancer.data.loaders import make_loader
from SkinCancer.data.split_io import build_image_path_map
from SkinCancer.data.transforms import build_train_transform, build_val_transform
from SkinCancer.utils.seed import set_seed

OUT = os.path.join(OUTPUT_ROOT, "residual"); os.makedirs(OUT, exist_ok=True)
FINAL7 = {"nv": 0, "mel": 1, "bkl": 2, "df": 3, "vasc": 4, "bcc": 5, "akiec": 6}
N_FOLDS = 5


class ResidualModel(nn.Module):
    """The backbone and its seven-class classifier. The explanation heads read the stage-2
    feature map and the pooled feature of this model with its weights frozen, so nothing is
    stacked on top of it here."""

    def __init__(self, kind, pretrained=True):
        super().__init__()
        if kind != "flat":
            raise ValueError(kind)
        self.kind = kind
        self.net = timm.create_model(BACKBONE_NAME, pretrained=pretrained, num_classes=7)
        if hasattr(self.net, "set_grad_checkpointing"):
            self.net.set_grad_checkpointing(True)

    def forward(self, x):
        z = self.net(x)
        return {"logits": z, "flat": z}


def _read(csv, pm):
    df = pd.read_csv(csv)
    if "diagnosis" not in df.columns and "dx" in df.columns:
        df = df.rename(columns={"dx": "diagnosis"})
    df["diagnosis"] = df["diagnosis"].str.lower()
    df["image_path"] = df["image_id"].map(pm)
    df["label"] = df["diagnosis"].map(FINAL7)
    return df.dropna(subset=["image_path", "label"]).reset_index(drop=True)


def folds_table(pm):
    f = os.path.join(OUT, "cv_folds.csv")
    if os.path.exists(f):
        return pd.read_csv(f)
    df = pd.concat([_read(os.path.join(SPLIT_ROOT, "global_train.csv"), pm),
                    _read(os.path.join(SPLIT_ROOT, "global_val.csv"), pm)], ignore_index=True)
    df = df.sort_values(["lesion_id", "image_id"]).reset_index(drop=True)
    df["fold"] = -1
    for k, (_, vi) in enumerate(GroupKFold(N_FOLDS).split(df, groups=df["lesion_id"])):
        df.loc[vi, "fold"] = k
    df[["image_id", "lesion_id", "diagnosis", "fold"]].to_csv(f, index=False)
    return df[["image_id", "lesion_id", "diagnosis", "fold"]]


def get_data(fold, pm):
    if fold < 0:
        return (_read(os.path.join(SPLIT_ROOT, "global_train.csv"), pm),
                _read(os.path.join(SPLIT_ROOT, "global_val.csv"), pm), _read(FINAL7_TEST_CSV, pm))
    ft = folds_table(pm)
    full = pd.concat([_read(os.path.join(SPLIT_ROOT, "global_train.csv"), pm),
                      _read(os.path.join(SPLIT_ROOT, "global_val.csv"), pm)], ignore_index=True)
    full = full.merge(ft[["image_id", "fold"]], on="image_id")
    tr = full[full["fold"] != fold].reset_index(drop=True)
    va = full[full["fold"] == fold].reset_index(drop=True)
    assert not set(tr["lesion_id"]) & set(va["lesion_id"]), "lesion leakage"
    return tr, va, None


def inv_freq(counts):
    counts = np.maximum(np.asarray(counts, float), 1)
    return torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float32, device=DEVICE)


def macro_auc(P, y):
    p = np.asarray(P, np.float64); p = p / p.sum(1, keepdims=True)
    try:
        return roc_auc_score(y, p, multi_class="ovr", average="macro")
    except Exception:
        return float((p.argmax(1) == y).mean())


@torch.no_grad()
def infer(model, loader):
    model.eval(); P, Pf, Y = [], [], []
    for img, lbl in tqdm(loader, leave=False):
        with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
            out = model(img.to(DEVICE))
        P.append(torch.softmax(out["logits"].float(), 1).cpu().numpy())
        Pf.append(torch.softmax(out["flat"].float(), 1).cpu().numpy())
        Y.append(lbl.numpy().flatten())
    return np.concatenate(P), np.concatenate(Pf), np.concatenate(Y).astype(int)


def run(a):
    set_seed(a.seed)
    tag = f"{a.model}_f{a.fold}_s{a.seed}" if a.fold >= 0 else f"{a.model}_full_s{a.seed}"
    pm = build_image_path_map()
    tr, va, te = get_data(a.fold, pm)
    y = tr["label"].astype(int).values
    ce = nn.CrossEntropyLoss(weight=inv_freq(np.bincount(y, minlength=7)))
    print(f"[residual:{tag}] model={a.model} fold={a.fold} train={len(tr)} val={len(va)} "
          f"test={0 if te is None else len(te)}")
    model = ResidualModel(a.model, pretrained=not a.no_pretrained).to(DEVICE)
    extra = sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in model.net.parameters())
    print(f"[residual:{tag}] params backbone+flat={sum(p.numel() for p in model.net.parameters()):,} extra={extra:,}")
    opt = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
    tl = make_loader(tr, "label", build_train_transform(), shuffle=True)
    vl = make_loader(va, "label", build_val_transform(), shuffle=False)
    if a.limit:
        tl = make_loader(tr.groupby("label").head(a.limit // 7 + 1).reset_index(drop=True),
                         "label", build_train_transform(), shuffle=True)
    ckpt, best = os.path.join(OUT, f"{tag}.pth"), -1.0
    epochs = a.epochs or EPOCHS_STAGE1
    for ep in range(epochs):
        model.train(); tot = 0.0
        opt.zero_grad(set_to_none=True)
        for i, (img, lbl) in enumerate(tl):
            img, lbl = img.to(DEVICE), lbl.to(DEVICE).long()
            with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                out = model(img)
                loss = ce(out["logits"], lbl)
            scaler.scale(loss / ACCUMULATION_STEPS).backward()
            if (i + 1) % ACCUMULATION_STEPS == 0:
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            tot += float(loss)
        P, Pf, yv = infer(model, vl)
        auc, auc_f = macro_auc(P, yv), macro_auc(Pf, yv)
        print(f"Epoch {ep+1:02d}/{epochs} | Loss {tot/max(len(tl),1):.4f} | Val AUC {auc:.4f} "
              f"(flat head {auc_f:.4f}) Acc {(P.argmax(1)==yv).mean():.4f}")
        if auc > best:
            best = auc; torch.save(model.state_dict(), ckpt)
            print(f"  -> checkpoint saved (val AUC {best:.4f})")
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True), strict=True)
    for sp, df in [("val", va)] + ([("test", te)] if te is not None else []):
        P, Pf, yy = infer(model, make_loader(df, "label", build_val_transform(), shuffle=False))
        f = os.path.join(OUT, f"{tag}_{sp}_predictions.npz")
        np.savez(f, y_prob=P, y_prob_flat=Pf, y_true=yy, image_id=df["image_id"].values)
        print(f"[residual:{tag}] {sp} acc={(P.argmax(1)==yy).mean():.4f} saved: {f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="flat", choices=["flat"])
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-pretrained", action="store_true")
    run(ap.parse_args())


# ---------------------------------------------------------------------------------------
from sklearn.model_selection import GroupShuffleSplit as _GroupShuffleSplit

INNER_VAL = float(os.environ.get("HAM_INNER_VAL", "0") or 0)
_get_data_single_level = get_data


def get_data(fold, pm):
    tr, va, te = _get_data_single_level(fold, pm)
    if fold < 0 or INNER_VAL <= 0:
        return tr, va, te
    gss = _GroupShuffleSplit(n_splits=1, test_size=INNER_VAL, random_state=1000 + fold)
    i_tr, i_va = next(gss.split(tr, groups=tr["lesion_id"]))
    inner_tr = tr.iloc[i_tr].reset_index(drop=True)
    inner_va = tr.iloc[i_va].reset_index(drop=True)
    for x, y, what in ((inner_tr, inner_va, "inner train/val"),
                       (inner_tr, va, "inner train/held-out"),
                       (inner_va, va, "inner val/held-out")):
        assert not set(x["lesion_id"]) & set(y["lesion_id"]), f"lesion overlap in {what}"
    print(f"[nested] fold {fold}: inner train {len(inner_tr)}, inner val {len(inner_va)}, "
          f"held out {len(va)}  (selection on the inner val only)")
    return inner_tr, inner_va, va


def eval_split(fold, pm):
    """The split the results are reported on: always the held-out fold."""
    _, va, report = get_data(fold, pm)
    return report if report is not None else va


if __name__ == "__main__":
    main()
