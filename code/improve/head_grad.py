"""Read a trained head's own gradient as its map, instead of the attention weight it reports.

    python -m improve.head_grad --fold 0
    python -m improve.head_grad --fold 0 --heads mlp,mlp_sup0.2
"""

import argparse
import os

import numpy as np
import torch

from SkinCancer.configs.model_config import IMG_SIZE
from SkinCancer.configs.training_config import DEVICE, OUTPUT_ROOT
from SkinCancer.data.loaders import make_loader
from SkinCancer.data.split_io import build_image_path_map
from SkinCancer.data.transforms import build_val_transform
from SkinCancer.utils.seed import set_seed

from improve.head_swap import (MASK_DIR, OUT as HS_OUT, build_head, center_map, curve_within,
                               feats, load_flat, probe, source_id)
from improve.localize import gradcam, metrics, up
from improve.residual import eval_split

OUT = os.path.join(OUTPUT_ROOT, "headswap")
DISTIL_OUT = os.path.join(OUTPUT_ROOT, "distil")
KEYS = ("attn", "grad", "center")


def resolve(spec, fold):
    """Checkpoint, head name and map mode for one head, as head_swap.py wrote them."""
    if spec.startswith("distil:"):
        tag = spec.split(":", 1)[1]
        return (os.path.join(DISTIL_OUT, f"head_{tag}_f{fold}.pth"), tag.split("_", 1)[0],
                "softmax", os.path.join(DISTIL_OUT, f"ds_{tag}_f{fold}.npz"))
    mode = "softmax" if "_sup" in spec else "relu"
    return (os.path.join(HS_OUT, f"head_{spec}_f{fold}.pth"), spec.split("_sup", 1)[0], mode,
            os.path.join(HS_OUT, f"hs_{spec}_f{fold}.npz"))


class NoGradient(Exception):
    """Raised when a head's logit carries no gradient back to the stage-2 features."""


def _corr(a, b):
    """Per-image Pearson correlation between two flattened maps, averaged over the batch."""
    a = a - a.mean(1, keepdim=True)
    b = b - b.mean(1, keepdim=True)
    d = (a.norm(dim=1) * b.norm(dim=1)).clamp_min(1e-12)
    return float(((a * b).sum(1) / d).mean())


def saved_whole(path):
    """The whole-image Ins-Del already recorded for this head and fold, for the check."""
    if not os.path.exists(path):
        return float("nan")
    d = np.load(path, allow_pickle=True)
    if "faith_head" not in d:
        return float("nan")
    return float(d["faith_head"][2])


def measure(net, head, s2_dim, ev, mask_of, n_img, steps):
    from improve.faithful import blur, curve

    cen = torch.tensor(center_map(), dtype=torch.float32, device=DEVICE)
    cen_np = center_map()
    sub = ev.head(n_img).reset_index(drop=True)
    jobs = ("del", "ins", "wdel", "wins")
    acc = {k: {j: [] for j in jobs} for k in KEYS}
    rows, fa_corr, seen, used = [], [], 0, 0
    head.eval()
    for img, _ in make_loader(sub, "label", build_val_transform(), shuffle=False):
        x = img.to(DEVICE)
        B = x.shape[0]
        mk, ok = [], []
        for j in range(B):
            a = mask_of(sub.iloc[seen + j]["image_id"])
            mk.append(a)
            ok.append(bool(0.0 < a.mean() < 1.0))
        seen += B
        mask = torch.tensor(np.stack(mk), dtype=torch.float32, device=DEVICE)
        keep = np.array(ok)

        with torch.no_grad():
            f, s2, _ = feats(net, x)
            cls = net.forward_head(f).argmax(1)
            macro = net.forward_head(f, pre_logits=True).float()
            micro = s2.float()

        # The backbone stays frozen and out of the graph; only the head is differentiated, and
        mg = micro.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            z, sal = head(macro.detach(), mg, cls)
            z.gather(1, cls[:, None]).sum().backward()
            if mg.grad is None:
                # A head whose logit does not depend differentiably on the stage-2 input has no
                raise NoGradient
            g = gradcam(mg.detach(), mg.grad, nhwc=False)
        head.zero_grad(set_to_none=True)

        M = {"attn": up(sal.detach().float()).detach(),
             "grad": up(g.detach().float()).detach(),
             "center": cen.expand(B, -1, -1)}

        # If the two readouts were nearly the same map there would be nothing to compare, so the
        fa_corr.append(_corr(M["attn"].reshape(B, -1), M["grad"].reshape(B, -1)))
        Mn = {k: M[k].cpu().numpy() for k in ("attn", "grad")}
        for j in range(B):
            rows.append(metrics(np.stack([Mn["attn"][j], Mn["grad"][j], cen_np]), mk[j]))

        base = blur(x)
        with torch.no_grad():
            for nm in KEYS:
                for mode in ("del", "ins"):
                    acc[nm][mode].append(curve(net, x, base, M[nm], cls, steps, mode))
                if keep.any():
                    w = curve_within(net, x, base, M[nm], mask, cls, steps)
                    acc[nm]["wdel"].append(w["del"][keep])
                    acc[nm]["wins"].append(w["ins"][keep])
        used += int(keep.sum())

    R = np.stack(rows)                                  # images x 3 maps x 3 overlap measures
    fa = {}
    for nm in KEYS:
        v = {j: np.concatenate(acc[nm][j]).mean(1).mean() for j in acc[nm]}
        fa[nm] = (float(v["ins"] - v["del"]), float(v["wins"] - v["wdel"]))
    return R, fa, len(sub), used, float(np.mean(fa_corr))


def run(a):
    from PIL import Image

    set_seed(a.seed)
    os.makedirs(OUT, exist_ok=True)
    specs = [s.strip() for s in a.heads.split(",") if s.strip()]
    ev = eval_split(a.fold, build_image_path_map())
    net = load_flat(a.fold)
    s2_dim, final_dim, _hw = probe(net)

    cache = {}

    def mask_of(iid):
        if iid not in cache:
            mp = os.path.join(MASK_DIR, f"{iid}_segmentation.png")
            cache[iid] = (np.array(Image.open(mp).convert("L")
                                   .resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)) > 127
                          if os.path.exists(mp) else np.zeros((IMG_SIZE, IMG_SIZE), bool))
        return cache[iid]

    print(f"[head_grad] fold {a.fold}, {len(specs)} heads, {a.faith_n} images")
    print(f"[head_grad] produced by head_grad.py {source_id()}")
    out = {}
    for spec in specs:
        ck, name, mode, prev = resolve(spec, a.fold)
        if not os.path.exists(ck):
            print(f"  [{spec}] no checkpoint at {ck}; skipping")
            continue
        head = build_head(name, net, s2_dim, final_dim, 7, 4, False, mode)
        head.load_state_dict(torch.load(ck, map_location=DEVICE, weights_only=True), strict=True)
        try:
            R, fa, n, n_w, corr = measure(net, head, s2_dim, ev, mask_of, a.faith_n, a.steps)
        except NoGradient:
            print(f"  [{spec}] its logit carries no gradient to the stage-2 features; skipping")
            continue
        ref = saved_whole(prev)
        ok = "" if np.isnan(ref) else ("  [match]" if abs(fa["attn"][0] - ref) < 2e-3
                                       else f"  [MISMATCH vs {ref:+.4f}]")
        print(f"  [{spec}] map={mode}  attention Ins-Del {fa['attn'][0]:+.4f}{ok}"
              f"  corr(attention, gradient) {corr:+.3f}")
        iou = [float(np.nanmean(R[:, i, 2])) for i in range(3)]
        out[spec] = dict(R=R, fa=fa, iou=iou, n=n, n_w=n_w, corr=corr)
        np.savez(os.path.join(OUT, f"hg_{spec.replace(':', '_')}_f{a.fold}.npz"),
                 metrics=R, names=np.array(KEYS),
                 faith=np.array([out[spec]["fa"][k][0] for k in KEYS]),
                 within=np.array([out[spec]["fa"][k][1] for k in KEYS]),
                 corr=corr, faith_n=a.faith_n, faith_n_within=n_w, source_sha1=source_id())

    if not out:
        raise SystemExit("no head had a checkpoint for this fold; nothing measured")
    print()
    print(f"    {'head':<16}{'readout':<12}{'IoU':>9}{'Ins-Del':>11}{'within':>11}")
    for spec, d in out.items():
        for i, k in enumerate(("attn", "grad")):
            label = "attention" if k == "attn" else "gradient"
            print(f"    {spec if i == 0 else '':<16}{label:<12}{d['iou'][i]:>9.4f}"
                  f"{d['fa'][k][0]:>+11.4f}{d['fa'][k][1]:>+11.4f}")
    k = next(iter(out))
    print(f"    {'(reference)':<16}{'centered':<12}{out[k]['iou'][2]:>9.4f}"
          f"{out[k]['fa']['center'][0]:>+11.4f}{out[k]['fa']['center'][1]:>+11.4f}")
    print()
    print("    The attention rows must carry [match] against the value already saved for this")
    print("    fold; without that the gradient rows mean nothing. What the gradient rows decide:")
    print("    a positive within-lesion value says the information was in the head all along and")
    print("    the attention weight is the wrong thing to display, while a negative one says a")
    print("    readout trained for pooling does not encode within-lesion importance at all.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--heads", default="ssf,lowrank,cam,mlp,mlp_sup0.2",
                    help="head_swap stems, or distil:<tag>")
    ap.add_argument("--faith-n", type=int, default=400, help="images, as improve.faithful uses")
    ap.add_argument("--steps", type=int, default=10)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
