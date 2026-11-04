"""H2: the mask decides where, the classifier's gradient decides how much.

    python -m improve.grad_distil --fold 0 --limit 64 # smoke test
    python -m improve.grad_distil --fold 0
    python -m improve.grad_distil --fold 0 --teacher gc4
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from SkinCancer.configs.training_config import BATCH_SIZE, DEVICE, OUTPUT_ROOT
from SkinCancer.data.split_io import build_image_path_map
from SkinCancer.utils.seed import set_seed

from improve.head_swap import (OUT, build_head, check_splits, counts, eval_head_live, faith_of,
                               feats, inv_freq, load_flat, probe, source_id)
from improve.localize import gradcam
from improve.residual import eval_split, get_data

TEACHERS = ("gc2", "gc3", "gc4")


def teacher_map(net, img, cls, which, store, grid):
    """The teacher's saliency map for this batch, at the mask's resolution, detached."""
    x = img.clone().requires_grad_(True)
    with torch.enable_grad():
        f, _s2, _ = feats(net, x)
        f.retain_grad()
        z = net.forward_head(f)
        z.gather(1, cls[:, None]).sum().backward()
        if which == "gc4":
            g = gradcam(f.detach(), f.grad, nhwc=True)
        else:
            h = store[which]
            g = gradcam(h.detach(), h.grad, nhwc=(h.shape[-1] != h.shape[-2]))
    net.zero_grad(set_to_none=True)
    g = Fn.interpolate(g.detach()[:, None].float(), size=(grid, grid),
                       mode="bilinear", align_corners=False)[:, 0]
    return g.reshape(g.shape[0], -1)


def ent_of(tgt):
    """Mean entropy of the normalized target, in nats. The diagnosis this method turns on."""
    s = tgt.sum(1, keepdim=True)
    ok = s[:, 0] > 1e-8
    if not bool(ok.any()):
        return float("nan")
    t = tgt[ok] / s[ok].clamp_min(1e-12)
    return float(-(t * torch.log(t.clamp_min(1e-12))).sum(1).mean())


def ce_to(att, tgt):
    """Cross-entropy from a normalized map to a normalized target, skipping empty targets."""
    s = tgt.sum(1, keepdim=True)
    ok = s[:, 0] > 1e-8
    if not bool(ok.any()):
        return att.sum() * 0.0, 0
    t = tgt[ok] / s[ok].clamp_min(1e-12)
    a = att[ok]
    return -(t * torch.log(a.clamp_min(1e-12))).sum(1).mean(), int(ok.sum())


def train(head, net, s2_dim, tr, va, a, grid):
    from sklearn.metrics import roc_auc_score
    from torch.utils.data import DataLoader

    from improve.localize_sup import MaskSet

    y = tr["label"].astype(int).values
    ce = nn.CrossEntropyLoss(weight=inv_freq(np.bincount(y, minlength=7)))
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=5e-5)
    tl = DataLoader(MaskSet(tr, True, grid=grid), batch_size=BATCH_SIZE, shuffle=True,
                    num_workers=4, pin_memory=True)
    vl = DataLoader(MaskSet(va, False, grid=grid), batch_size=BATCH_SIZE, shuffle=False,
                    num_workers=4, pin_memory=True)

    store = {}

    def _mk(key):
        def _hook(mod, inp, out):
            if out.requires_grad:
                out.retain_grad()
                store[key] = out
        return _hook

    # stages[1] is the second stage and stages[2] the third, which is how improve.localize reads
    hooks = [net.stages[i].register_forward_hook(_mk(f"gc{i + 1}")) for i in (1, 2)]

    best, state = -1.0, None
    for ep in range(a.epochs):
        head.train()
        tc = tg = tm = hg = hm = 0.0
        nb = 0
        for img, lbl, m in tl:
            img, lbl, m = img.to(DEVICE), lbl.to(DEVICE).long(), m.to(DEVICE)
            with torch.no_grad():
                f, s2, _ = feats(net, img)
                cls = net.forward_head(f).argmax(1)
                macro = net.forward_head(f, pre_logits=True).float()
                micro = s2.float()
            gt = teacher_map(net, img, cls, a.teacher, store, grid)

            z, sal = head(macro, micro)
            att = sal.reshape(sal.shape[0], -1).float()
            att = att / att.sum(1, keepdim=True).clamp_min(1e-12)
            # The mask gates the teacher: the product is zero outside the lesion, so the support
            # is the mask's and the ordering inside it is the gradient's.
            l_grad, _n = ce_to(att, gt * m)
            with torch.no_grad():
                l_mask, _ = ce_to(att, m)
                # The entropies, not the losses, say whether the gradient target carries an
                hg += ent_of(gt * m); hm += ent_of(m)
            l_cls = ce(z, lbl)
            loss = l_cls + a.lam * l_grad
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tc += float(l_cls); tg += float(l_grad); tm += float(l_mask); nb += 1

        head.eval()
        P, Y, vg, vm = [], [], [], []
        for img, lbl, m in vl:
            img, m = img.to(DEVICE), m.to(DEVICE)
            with torch.no_grad():
                f, s2, _ = feats(net, img)
                cls = net.forward_head(f).argmax(1)
                macro = net.forward_head(f, pre_logits=True).float()
                micro = s2.float()
            gt = teacher_map(net, img, cls, a.teacher, store, grid)
            with torch.no_grad():
                z, sal = head(macro, micro, lbl.to(DEVICE).long())
                att = sal.reshape(sal.shape[0], -1).float()
                att = att / att.sum(1, keepdim=True).clamp_min(1e-12)
                vg.append(float(ce_to(att, gt * m)[0]))
                vm.append(float(ce_to(att, m)[0]))
                P.append(torch.softmax(z.float(), 1).cpu().numpy())
                Y.append(lbl.numpy())
        P, Y = np.concatenate(P), np.concatenate(Y).astype(int)
        try:
            auc = roc_auc_score(Y, P / P.sum(1, keepdims=True), multi_class="ovr",
                               average="macro")
        except Exception:                                              # noqa: BLE001
            auc = float((P.argmax(1) == Y).mean())
        g_val, m_val = float(np.mean(vg)), float(np.mean(vm))
        # The epoch is chosen on the quantity being supervised, as head_swap.py's --select att
        score = -g_val
        if score > best:
            best, state = score, {k: v.detach().clone() for k, v in head.state_dict().items()}
        if not a.quiet:
            print(f"    epoch {ep + 1:02d}/{a.epochs}  cls {tc / max(nb,1):.4f}  "
                  f"grad {tg / max(nb,1):.4f}  (mask {tm / max(nb,1):.4f})  "
                  f"H[grad] {hg / max(nb,1):.3f}  H[mask] {hm / max(nb,1):.3f}  "
                  f"val grad {g_val:.4f}  val AUC {auc:.4f}"
                  + ("  <-" if score == best else ""))
    for h in hooks:
        h.remove()
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
    net = load_flat(a.fold)
    s2_dim, final_dim, hw = probe(net)
    if hw[0] != hw[1]:
        raise SystemExit(f"the mask grid assumes a square map, got {hw}")

    head = build_head(a.head, net, s2_dim, final_dim, 7, 4, False, "softmax")
    total, mech = counts(head)
    stem = f"grad{a.teacher}_{a.head}_sup{a.lam:g}"
    print(f"[grad_distil] fold {a.fold} seed {a.seed}  train={len(tr)} val={len(va)} "
          f"eval={len(ev)}  stage-2 map {s2_dim}x{hw[0]}x{hw[1]}  macro {final_dim}")
    print(f"[grad_distil] produced by grad_distil.py {source_id()}")
    print(f"  [{stem}] teacher {a.teacher}, lambda {a.lam:g}, "
          f"total {total:,} parameters, saliency mechanism {mech:,}")

    t0 = time.time()
    best = train(head, net, s2_dim, tr, va, a, hw[0])
    tag = f"{stem}_f{a.fold}" + (f"_lim{a.limit}" if a.limit else "")
    os.makedirs(OUT, exist_ok=True)
    torch.save(head.state_dict(), os.path.join(OUT, f"head_{tag}.pth"))

    R, ids, labels, fpred, rpred, areas = eval_head_live(head, net, s2_dim, ev)
    fa, n_fa, n_w = faith_of(net, head, s2_dim, ev, a.faith_n, a.faith_steps)
    path = os.path.join(OUT, f"hs_{tag}.npz")
    np.savez(path, metrics=R, names=np.array([stem, "center"]), image_id=ids, label=labels,
             flat_pred=fpred, readout_pred=rpred, lesion_area=areas,
             params_total=total, params_mechanism=mech, readout_val_acc=float("nan"),
             teacher=a.teacher, lam=a.lam, val_grad_ce=-best, source_sha1=source_id(),
             faith_head=np.array(fa["head"]), faith_center=np.array(fa["center"]),
             faith_n=a.faith_n, faith_n_within=n_w)

    mine = [float(np.nanmean(R[:, 0, c])) for c in range(3)]
    ref = [float(np.nanmean(R[:, 1, c])) for c in range(3)]
    print()
    print(f"    {'map':<26}{'Pointing':>10}{'Energy':>9}{'IoU':>9}{'Ins-Del':>10}{'within':>10}")
    print(f"    {stem:<26}{mine[0]:>10.4f}{mine[1]:>9.4f}{mine[2]:>9.4f}"
          f"{fa['head'][2]:>+10.4f}{fa['head'][5]:>+10.4f}")
    print(f"    {'center':<26}{ref[0]:>10.4f}{ref[1]:>9.4f}{ref[2]:>9.4f}"
          f"{fa['center'][2]:>+10.4f}{fa['center'][5]:>+10.4f}")
    print()
    print("    Read it against two rows already measured, on the same folds and protocol:")
    print("    the same head with a uniform-inside-the-lesion target reaches IoU 0.818 and a")
    print(f"    within-lesion value of -0.012, and the teacher {a.teacher} reaches an IoU of")
    print("    0.17 to 0.23 with a within-lesion value of +0.07 to +0.14. The claim this run")
    print("    tests is a map with both, not a map that beats its teacher.")
    print(f"    {time.time() - t0:.0f}s   saved: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--head", default="mlp", choices=["mlp", "cam", "lowrank", "ssf"],
                    help="the mechanism; mlp is the row this is compared against")
    ap.add_argument("--teacher", default="gc3", choices=TEACHERS,
                    help="which Grad-CAM depth supplies the ordering inside the lesion")
    ap.add_argument("--lam", type=float, default=0.2,
                    help="weight on the alignment term, as the mask-supervised rows use")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--limit", type=int, default=0, help="smoke test on a few images per class")
    ap.add_argument("--faith-n", type=int, default=400)
    ap.add_argument("--faith-steps", type=int, default=10)
    ap.add_argument("--quiet", action="store_true")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
