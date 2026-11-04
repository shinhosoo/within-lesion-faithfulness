"""Train a saliency head on the classifier's own occlusion sensitivity inside the lesion.

    python -m improve.distil --stage targets --fold 0
    python -m improve.distil --stage train --fold 0
    python -m improve.distil --stage train --fold 0 --lam-attr 1.0 --lam-mask 0.0
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader

from SkinCancer.configs.model_config import IMG_SIZE
from SkinCancer.configs.training_config import BATCH_SIZE, DEVICE, OUTPUT_ROOT
from SkinCancer.data.loaders import make_loader
from SkinCancer.data.split_io import build_image_path_map
from SkinCancer.data.transforms import build_val_transform
from SkinCancer.utils.seed import set_seed

from improve.head_swap import (build_head, counts, eval_head_live, faith_of, feats, load_flat,
                               probe, source_id)
from improve.localize_sup import JIT, MASK_DIR, NORM, MaskSet
from improve.residual import eval_split, get_data, inv_freq

OUT = os.path.join(OUTPUT_ROOT, "distil")
TGT = os.path.join(OUT, "targets")


# --------------------------------------------------------------------------------------

def grid_cells(mask, n, mask_free=False):
    """The n x n cells to occlude, as (y0, y1, x0, x1) at image resolution."""
    step = IMG_SIZE // n
    out = []
    for i in range(n):
        for j in range(n):
            y0, x0 = i * step, j * step
            y1 = IMG_SIZE if i == n - 1 else y0 + step
            x1 = IMG_SIZE if j == n - 1 else x0 + step
            if mask_free or mask[y0:y1, x0:x1].any():
                out.append((y0, y1, x0, x1))
    return out


@torch.no_grad()
def importance(net, x, masks, grid, blur_fn, g=48, mask_free=False):
    """A g x g importance target per image: the probability lost when each cell is blurred."""
    B = x.shape[0]
    base = blur_fn(x)
    with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
        f = net.forward_intermediates(x, intermediates_only=False)[0]
        p_full = torch.softmax(net.forward_head(f).float(), 1)
    cls = p_full.argmax(1)
    p0 = p_full.gather(1, cls[:, None]).squeeze(1)

    tgt = torch.zeros(B, IMG_SIZE, IMG_SIZE, device=x.device)
    for b in range(B):
        m = masks[b]
        cells = grid_cells(m.cpu().numpy() > 0.5, grid, mask_free)
        if not cells:
            continue
        # one batched pass over this image's cells
        xs = x[b : b + 1].repeat(len(cells), 1, 1, 1).clone()
        for k, (y0, y1, x0, x1) in enumerate(cells):
            xs[k, :, y0:y1, x0:x1] = base[b, :, y0:y1, x0:x1]
        drops = []
        for s in range(0, len(cells), 32):
            chunk = xs[s : s + 32]
            with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                ff = net.forward_intermediates(chunk, intermediates_only=False)[0]
                p = torch.softmax(net.forward_head(ff).float(), 1)
            drops.append(p0[b] - p[:, cls[b]])
        drops = torch.cat(drops)
        for k, (y0, y1, x0, x1) in enumerate(cells):
            tgt[b, y0:y1, x0:x1] = drops[k]
    tgt = torch.relu(tgt)
    if not mask_free:
        tgt = tgt * masks                                       # inside the lesion only
    tgt = torch.nn.functional.adaptive_avg_pool2d(tgt[:, None], g)[:, 0]
    return tgt


def make_targets(a):
    """Compute and cache the targets for one fold. They depend only on the frozen backbone."""
    from improve.faithful import blur

    os.makedirs(TGT, exist_ok=True)
    pm = build_image_path_map()
    tr, va, _ = get_data(a.fold, pm)
    net = load_flat(a.fold)
    for split, df in (("train", tr), ("val", va)):
        path = os.path.join(TGT, f"f{a.fold}_{split}{'_free' if a.mask_free else ''}.npz")
        if os.path.exists(path) and not a.rebuild:
            print(f"  [{split}] {os.path.basename(path)} exists; skipping")
            continue
        ids, maps, k, miss = [], [], 0, 0
        loader = make_loader(df, "label", build_val_transform(), shuffle=False)
        for img, _ in loader:
            x = img.to(DEVICE)
            mk = []
            for j in range(x.shape[0]):
                iid = df.iloc[k + j]["image_id"]
                mp = os.path.join(MASK_DIR, f"{iid}_segmentation.png")
                if os.path.exists(mp):
                    arr = np.array(Image.open(mp).convert("L")
                                   .resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)) > 127
                else:
                    arr = np.zeros((IMG_SIZE, IMG_SIZE), bool)
                    miss += 1
                mk.append(arr.astype(np.float32))
            masks = torch.tensor(np.stack(mk), device=DEVICE)
            t = importance(net, x, masks, a.grid, blur, a.target_grid, a.mask_free)
            for j in range(x.shape[0]):
                ids.append(df.iloc[k + j]["image_id"])
            maps.append(t.cpu().numpy().astype(np.float16))
            k += x.shape[0]
            if k % 800 == 0:
                print(f"  [{split}] {k}/{len(df)}")
        M = np.concatenate(maps)
        np.savez(path, image_id=np.array(ids), target=M, grid=a.grid,
                 source_sha1=source_id())
        frac = float((M.reshape(len(M), -1).sum(1) > 0).mean())
        print(f"  [{split}] {len(ids)} targets, {frac:.1%} non-empty"
              + (f", {miss} masks missing" if miss else "") + f" -> {path}")


# --------------------------------------------------------------------------------------

class TargetSet(MaskSet):
    """MaskSet plus the cached importance target, carried through the same augmentation."""

    def __init__(self, df, train, targets, grid=48):
        super().__init__(df, train, grid=grid)
        self.tg = targets                                       # image_id -> g x g float array

    def __getitem__(self, i):
        img = Image.open(self.p[i]).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        mp = os.path.join(MASK_DIR, f"{self.ids[i]}_segmentation.png")
        msk = Image.open(mp).convert("L").resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
        t = self.tg[self.ids[i]]
        tim = Image.fromarray(np.asarray(t, dtype=np.float32), mode="F")
        if self.train:
            if random.random() < 0.5:
                img, msk, tim = TF.hflip(img), TF.hflip(msk), TF.hflip(tim)
            if random.random() < 0.2:
                img, msk, tim = TF.vflip(img), TF.vflip(msk), TF.vflip(tim)
            ang = random.uniform(-30, 30)
            img = TF.rotate(img, ang, interpolation=TF.InterpolationMode.BILINEAR)
            msk = TF.rotate(msk, ang, interpolation=TF.InterpolationMode.NEAREST)
            tim = TF.rotate(tim, ang, interpolation=TF.InterpolationMode.BILINEAR)
            img = JIT(img)
        x = NORM(TF.to_tensor(img))
        m = torch.tensor(np.array(msk) > 127, dtype=torch.float32)[None, None]
        m = torch.nn.functional.adaptive_avg_pool2d(m, self.g)[0, 0].reshape(-1)
        tt = torch.tensor(np.array(tim), dtype=torch.float32).reshape(-1)
        return x, self.y[i], m, tt


def ce_to(att, tgt):
    """Cross-entropy from a map to a target, both normalized over positions."""
    s = tgt.sum(1, keepdim=True)
    live = (s > 1e-8).squeeze(1)
    if not live.any():
        return att.sum() * 0
    a = att[live] / att[live].sum(1, keepdim=True).clamp_min(1e-12)
    t = tgt[live] / s[live]
    return -(t * torch.log(a.clamp_min(1e-12))).sum(1).mean()


def train(a):
    from sklearn.metrics import roc_auc_score

    set_seed(a.seed)
    os.makedirs(OUT, exist_ok=True)
    pm = build_image_path_map()
    tr, va, _ = get_data(a.fold, pm)
    ev = eval_split(a.fold, pm)
    net = load_flat(a.fold)
    s2_dim, final_dim, hw = probe(net)

    tg = {}
    for split in ("train", "val"):
        p = os.path.join(TGT, f"f{a.fold}_{split}{'_free' if a.mask_free else ''}.npz")
        if not os.path.exists(p):
            raise SystemExit(f"no targets at {p}; run --stage targets --fold {a.fold} first")
        with np.load(p, allow_pickle=True) as z:
            for iid, m in zip(z["image_id"], z["target"]):
                tg[str(iid)] = m.astype(np.float32)
    print(f"[distil] fold {a.fold}  head {a.head}  lam_attr {a.lam_attr} lam_mask {a.lam_mask}")
    print(f"[distil] produced by distil.py {source_id()}  targets {len(tg)}")

    head = build_head(a.head, net, s2_dim, final_dim, 7, 4, False, "softmax")
    total, mech = counts(head)
    print(f"  [{a.head}] total {total:,} parameters, saliency mechanism {mech:,}")

    y = tr["label"].astype(int).values
    ce = nn.CrossEntropyLoss(weight=inv_freq(np.bincount(y, minlength=7)))
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=5e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
    tl = DataLoader(TargetSet(tr, True, tg, hw[0]), batch_size=BATCH_SIZE, shuffle=True,
                    num_workers=4, pin_memory=True)
    vl = DataLoader(TargetSet(va, False, tg, hw[0]), batch_size=BATCH_SIZE, shuffle=False,
                    num_workers=4, pin_memory=True)

    best, state = float("inf"), None
    for ep in range(a.epochs):
        head.train()
        tc = tm = ta = 0.0
        nb = 0
        for img, lbl, m, t in tl:
            img, lbl = img.to(DEVICE), lbl.to(DEVICE).long()
            m, t = m.to(DEVICE), t.to(DEVICE)
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                f, s2, _ = feats(net, img)
                macro = net.forward_head(f, pre_logits=True)
                micro = s2
            with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                z, sal = head(macro.float(), micro.float(), lbl)
                att = sal.reshape(sal.shape[0], -1).float()
                l_cls = ce(z, lbl)
                l_mask = ce_to(att, m)
                l_attr = ce_to(att, t)
                loss = l_cls + a.lam_mask * l_mask + a.lam_attr * l_attr
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tc += float(l_cls); tm += float(l_mask); ta += float(l_attr); nb += 1
        # The epoch is chosen on the validation attribution loss, because the map's agreement
        # with the measured importance is what this run is for.
        head.eval()
        va_attr, P, Y = [], [], []
        with torch.no_grad():
            for img, lbl, _m, t in vl:
                t = t.to(DEVICE)
                with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                    f, s2, _ = feats(net, img.to(DEVICE))
                    z, sal = head(net.forward_head(f, pre_logits=True).float(),
                                  s2.float(), lbl.to(DEVICE).long())
                va_attr.append(float(ce_to(sal.reshape(sal.shape[0], -1).float(), t)))
                P.append(torch.softmax(z.float(), 1).cpu().numpy())
                Y.append(lbl.numpy())
        P, Y = np.concatenate(P), np.concatenate(Y).astype(int)
        v = float(np.mean(va_attr))
        try:
            auc = roc_auc_score(Y, P / P.sum(1, keepdims=True), multi_class="ovr",
                                average="macro")
        except Exception:                                          # noqa: BLE001
            auc = float((P.argmax(1) == Y).mean())
        if v < best:
            best = v
            state = {k: q.detach().clone() for k, q in head.state_dict().items()}
        print(f"    epoch {ep + 1:02d}/{a.epochs}  cls {tc / nb:.4f}  mask {tm / nb:.4f}  "
              f"attr {ta / nb:.4f}  val attr {v:.4f}  val AUC {auc:.4f}"
              + ("  <-" if v == best else ""))
    if state is not None:
        head.load_state_dict(state)

    tag = (f"{a.head}_attr{a.lam_attr:g}_mask{a.lam_mask:g}"
           + ("_free" if a.mask_free else "") + f"_f{a.fold}")
    torch.save(head.state_dict(), os.path.join(OUT, f"head_{tag}.pth"))
    R, ids, labels, fpred, rpred, areas = eval_head_live(head, net, s2_dim, ev)
    fa, n_fa, n_w = faith_of(net, head, s2_dim, ev, a.faith_n, 10)
    path = os.path.join(OUT, f"ds_{tag}.npz")
    np.savez(path, metrics=R, names=np.array([a.head, "center"]), image_id=ids, label=labels,
             flat_pred=fpred, readout_pred=rpred, lesion_area=areas,
             params_total=total, params_mechanism=mech, readout_val_attr=best,
             faith_head=np.array(fa["head"]), faith_center=np.array(fa["center"]),
             faith_n=a.faith_n, faith_n_within=n_w, source_sha1=source_id())
    mine = [float(np.nanmean(R[:, 0, c])) for c in range(3)]
    ref = [float(np.nanmean(R[:, 1, c])) for c in range(3)]
    print()
    print(f"    fold {a.fold}, {len(ids)} images")
    print(f"    {'map':<22}{'Pointing':>11}{'Energy':>9}{'IoU':>9}{'Ins-Del':>10}{'within':>10}")
    print(f"    {tag:<22}{mine[0]:>11.4f}{mine[1]:>9.4f}{mine[2]:>9.4f}"
          f"{fa['head'][2]:>+10.4f}{fa['head'][5]:>+10.4f}")
    print(f"    {'center':<22}{ref[0]:>11.4f}{ref[1]:>9.4f}{ref[2]:>9.4f}"
          f"{fa['center'][2]:>+10.4f}{fa['center'][5]:>+10.4f}")
    print()
    print("    The number this run exists for is `within`. A map trained on the classifier's")
    print("    own occlusion sensitivity either reproduces it out of fold, which says the")
    print("    quantity is learnable and the mask was simply the wrong signal, or it does not,")
    print("    which says it is not predictable from these features at all.")
    print(f"    saved: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["targets", "train"])
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--head", default="mlp", choices=["mlp", "cam", "lowrank", "ssf"])
    ap.add_argument("--grid", type=int, default=6, help="n x n occlusion cells over the image")
    ap.add_argument("--target-grid", type=int, default=48, help="stored target resolution")
    ap.add_argument("--lam-attr", type=float, default=1.0, help="weight on the importance term")
    ap.add_argument("--lam-mask", type=float, default=0.2, help="weight on the mask term")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--faith-n", type=int, default=400)
    ap.add_argument("--mask-free", action="store_true",
                    help="occlude the whole image, not only the lesion, so that no annotation "
                         "enters the target; masks then appear only in the evaluation")
    ap.add_argument("--rebuild", action="store_true", help="recompute cached targets")
    a = ap.parse_args()
    (make_targets if a.stage == "targets" else train)(a)


if __name__ == "__main__":
    main()
