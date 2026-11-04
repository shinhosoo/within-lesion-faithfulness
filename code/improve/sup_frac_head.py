"""How many masks the supervision needs, for the replacement explanation head.

    python -m improve.sup_frac_head --fold 0 --frac 1.0 --limit 64 # smoke test
    python -m improve.sup_frac_head --fold 0 --frac 0.25
    python -m improve.sup_frac_head --fold 0 --frac 0.25 --stage 2
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn

from SkinCancer.configs.training_config import BATCH_SIZE, DEVICE
from SkinCancer.data.split_io import build_image_path_map
from SkinCancer.utils.seed import set_seed

from improve.deep_head import StageView
from improve.head_swap import (OUT, build_head, check_splits, counts, eval_head_live, faith_of,
                               feats, inv_freq, load_flat, probe, source_id, seed_out)
from improve.residual import eval_split, get_data
from improve.sup_frac import MaskSetFrac


def train(head, net, s_dim, tr, va, a, grid):
    from sklearn.metrics import roc_auc_score
    from torch.utils.data import DataLoader

    from improve.localize_sup import MaskSet

    y = tr["label"].astype(int).values
    ce = nn.CrossEntropyLoss(weight=inv_freq(np.bincount(y, minlength=7)))
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=5e-5)
    ds = MaskSetFrac(tr, True, a.frac, grid=grid)
    tl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    vl = DataLoader(MaskSet(va, False, grid=grid), batch_size=BATCH_SIZE, shuffle=False,
                    num_workers=4, pin_memory=True)
    n_sup = int(sum(ds.keep))
    print(f"  [frac {a.frac:g}] masks used on {n_sup} of {len(tr)} training images "
          f"({100 * n_sup / max(len(tr), 1):.1f}%)")

    best, state = -1.0, None
    for ep in range(a.epochs):
        head.train()
        tc = ta = 0.0
        used, nb = 0, 0
        for img, lbl, m, keep in tl:
            img, lbl = img.to(DEVICE), lbl.to(DEVICE).long()
            m, keep = m.to(DEVICE), keep.to(DEVICE).float()
            with torch.no_grad():
                f, s2, _ = feats(net, img)
                macro = net.forward_head(f, pre_logits=True).float()
                micro = s2.float()
            z, sal = head(macro, micro)
            att = sal.reshape(sal.shape[0], -1).float()
            att = att / att.sum(1, keepdim=True).clamp_min(1e-12)
            tgt = m / m.sum(1, keepdim=True).clamp_min(1e-12)
            # Per-sample, then averaged over the selected subset only: the unselected images
            per = -(tgt * torch.log(att.clamp_min(1e-12))).sum(1)
            l_att = (per * keep).sum() / keep.sum().clamp_min(1.0)
            l_cls = ce(z, lbl)
            loss = l_cls + a.lam * l_att
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tc += float(l_cls); ta += float(l_att); used += int(keep.sum()); nb += 1

        head.eval()
        P, Y, vatt = [], [], []
        with torch.no_grad():
            for img, lbl, m in vl:
                m = m.to(DEVICE)
                f, s2, _ = feats(net, img.to(DEVICE))
                z, sal = head(net.forward_head(f, pre_logits=True).float(),
                              s2.float(), lbl.to(DEVICE).long())
                att = sal.reshape(sal.shape[0], -1).float()
                att = att / att.sum(1, keepdim=True).clamp_min(1e-12)
                t = m / m.sum(1, keepdim=True).clamp_min(1e-12)
                vatt.append(float(-(t * torch.log(att.clamp_min(1e-12))).sum(1).mean()))
                P.append(torch.softmax(z.float(), 1).cpu().numpy())
                Y.append(lbl.numpy())
        P, Y = np.concatenate(P), np.concatenate(Y).astype(int)
        try:
            auc = roc_auc_score(Y, P / P.sum(1, keepdims=True), multi_class="ovr",
                               average="macro")
        except Exception:                                              # noqa: BLE001
            auc = float((P.argmax(1) == Y).mean())
        att_val = float(np.mean(vatt))
        # Selected on the validation alignment loss, as every other supervised row in this study
        score = -att_val if a.select == "att" else auc
        if score > best:
            best, state = score, {k: v.detach().clone() for k, v in head.state_dict().items()}
        if not a.quiet:
            print(f"    epoch {ep + 1:02d}/{a.epochs}  cls {tc / max(nb,1):.4f}  "
                  f"att {ta / max(nb,1):.4f} (on {used} samples)  val att {att_val:.4f}  "
                  f"val AUC {auc:.4f}" + ("  <-" if score == best else ""))
    if state is not None:
        head.load_state_dict(state)
    return best


def run(a):
    set_seed(a.seed)
    pm = build_image_path_map()
    tr, va, _ = get_data(a.fold, pm)
    ev = eval_split(a.fold, pm)
    if a.limit:
        tr = tr.groupby("label").head(max(2, a.limit // 7)).reset_index(drop=True)
        va = va.head(a.limit).reset_index(drop=True)
        ev = ev.head(a.limit).reset_index(drop=True)
    check_splits(tr, va, ev, a.limit)

    raw = load_flat(a.fold)
    net = StageView(raw, a.stage) if a.stage != 2 else raw
    s_dim, final_dim, hw = probe(net)
    if hw[0] != hw[1]:
        raise SystemExit(f"the mask grid assumes a square map, got {hw}")

    head = build_head(a.head, net, s_dim, final_dim, 7, 4, False, "softmax").to(DEVICE)
    total, mech = counts(head)
    stem = f"s{a.stage}_{a.head}_frac{a.frac:g}_sup{a.lam:g}"
    print(f"[sup_frac_head] fold {a.fold} seed {a.seed}  train={len(tr)} val={len(va)} "
          f"eval={len(ev)}  stage-{a.stage} map {s_dim}x{hw[0]}x{hw[1]}  macro {final_dim}")
    print(f"[sup_frac_head] produced by sup_frac_head.py {source_id()}")
    print(f"  [{stem}] total {total:,} parameters, saliency mechanism {mech:,}, lambda {a.lam:g}")

    t0 = time.time()
    best = train(head, net, s_dim, tr, va, a, hw[0])

    tag = f"{stem}_f{a.fold}" + (f"_lim{a.limit}" if a.limit else "")
    out = seed_out(a.seed)
    torch.save(head.state_dict(), os.path.join(out, f"head_{tag}.pth"))

    R, ids, labels, fpred, rpred, areas = eval_head_live(head, net, s_dim, ev)
    fa, n_fa, n_w = faith_of(net, head, s_dim, ev, a.faith_n, a.faith_steps)
    path = os.path.join(out, f"hs_{tag}.npz")
    np.savez(path, metrics=R, names=np.array([stem, "center"]), image_id=ids, label=labels,
             flat_pred=fpred, readout_pred=rpred, lesion_area=areas,
             params_total=total, params_mechanism=mech, readout_val_acc=best,
             stage=a.stage, frac=a.frac, lam=a.lam, select=a.select, source_sha1=source_id(),
             faith_head=np.array(fa["head"]), faith_center=np.array(fa["center"]),
             faith_n=a.faith_n, faith_n_within=n_w)

    mine = [float(np.nanmean(R[:, 0, c])) for c in range(3)]
    print()
    print(f"    {'map':<28}{'Pointing':>10}{'Energy':>9}{'IoU':>9}{'Ins-Del':>10}{'within':>10}")
    print(f"    {stem:<28}{mine[0]:>10.4f}{mine[1]:>9.4f}{mine[2]:>9.4f}"
          f"{fa['head'][2]:>+10.4f}{fa['head'][5]:>+10.4f}")
    print()
    if abs(a.frac - 1.0) < 1e-9:
        print("    This is the frac=1.0 check: every image supervised, so it has to reproduce the")
        print("    mask-supervised row of the supplementary table, an area-matched IoU of 0.872")
        print("    for the third stage. If it does not, the fractions below it mean nothing.")
    else:
        print("    Read against frac=1.0 of this same head, not against the published table: that")
        print("    table's rows are the localize-then-verify attention and reach 0.856 to 0.882.")
    print(f"    {time.time() - t0:.0f}s   saved: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--frac", type=float, required=True,
                    help="fraction of the training split whose masks enter the loss")
    ap.add_argument("--stage", type=int, default=3, choices=[2, 3, 4],
                    help="which backbone stage the head reads")
    ap.add_argument("--head", default="mlp", choices=["mlp", "cam", "lowrank", "ssf"])
    ap.add_argument("--lam", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--limit", type=int, default=0, help="smoke test on a few images per class")
    ap.add_argument("--select", default="att", choices=["att", "auc"])
    ap.add_argument("--faith-n", type=int, default=400)
    ap.add_argument("--faith-steps", type=int, default=10)
    ap.add_argument("--quiet", action="store_true")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
