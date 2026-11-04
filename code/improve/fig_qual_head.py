#!/usr/bin/env python3
"""The qualitative figure, drawn from the explanation head the paper reports.

    python -m improve.fig_qual_head --fold 0 --out figs/fig_qual_head
    python -m improve.fig_qual_head --fold 0 --gc-stage 2 --n 300
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                           # noqa: E402
import numpy as np                                                        # noqa: E402
import torch                                                              # noqa: E402
from PIL import Image                                                     # noqa: E402

from SkinCancer.configs.model_config import IMG_SIZE                      # noqa: E402
from SkinCancer.configs.training_config import DEVICE                     # noqa: E402
from SkinCancer.data.split_io import build_image_path_map                 # noqa: E402
from SkinCancer.utils.seed import set_seed                                # noqa: E402
from improve.localize import feats, gradcam, load_flat, up                # noqa: E402
from improve.head_swap import OUT, build_head, probe                      # noqa: E402
from improve.deep_head import StageView                                   # noqa: E402
from improve.residual import eval_split                                   # noqa: E402

MASK_DIR = os.environ.get("HAM_MASK_DIR", "masks_ham10000")
CLS = ["NV", "MEL", "BKL", "DF", "VASC", "BCC", "AKIEC"]


def norm01(m):
    m = m - m.min()
    return m / max(m.max(), 1e-12)


def iou_area_matched(m, mask):
    """improve.localize.metrics, for one map."""
    a = mask.mean()
    if a <= 0 or a >= 1:
        return float("nan")
    m = m - m.min()
    thr = np.quantile(m.reshape(-1), 1 - a)
    sel = m >= thr
    return float((sel & mask).sum() / max((sel | mask).sum(), 1))


def center_map():
    yy, xx = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]
    c, s = (IMG_SIZE - 1) / 2, IMG_SIZE * 0.25
    return np.exp(-((xx - c) ** 2 + (yy - c) ** 2) / (2 * s ** 2))


def overlay(ax, img, m=None, mask=None, title="", tag=None):
    ax.imshow(img)
    if m is not None:
        ax.imshow(norm01(m), cmap="jet", alpha=0.45)
    if mask is not None:
        ax.contour(mask.astype(float), levels=[0.5], colors="w", linewidths=1.2)
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=6.0, linespacing=1.15)
    if tag:
        ax.text(0.03, 0.97, tag, transform=ax.transAxes, ha="left", va="top", fontsize=5.4,
                color="white", bbox=dict(fc="black", alpha=0.55, pad=1.1, lw=0))


def load_head(net, s_dim, final_dim, stem, fold, mode):
    head = build_head("mlp", net, s_dim, final_dim, 7, 4, False, mode)
    ck = os.path.join(OUT, f"head_{stem}_f{fold}.pth")
    if not os.path.exists(ck):
        raise SystemExit(f"no checkpoint at {ck}")
    head.load_state_dict(torch.load(ck, map_location=DEVICE, weights_only=True), strict=True)
    head.eval().to(DEVICE)
    return head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stage", type=int, default=3, help="the head's backbone stage")
    ap.add_argument("--gc-stage", type=int, default=3, help="the depth Grad-CAM is read at")
    ap.add_argument("--n", type=int, default=300, help="candidate images to examine")
    ap.add_argument("--out", default="figs/fig_qual_head")
    a = ap.parse_args()
    set_seed(a.seed)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    ev = eval_split(a.fold, build_image_path_map())
    ev = ev.head(a.n).reset_index(drop=True)

    raw = load_flat(a.fold)
    net = StageView(raw, a.stage)
    s_dim, final_dim, hw = probe(net)
    print(f"[fig_qual_head] fold {a.fold}  stage-{a.stage} map {s_dim}x{hw[0]}x{hw[1]}  "
          f"Grad-CAM at stage {a.gc_stage}  {len(ev)} candidates")
    heads = {"unsup": load_head(net, s_dim, final_dim, f"s{a.stage}_mlp", a.fold, "softmax"),
             "sup": load_head(net, s_dim, final_dim, f"s{a.stage}_mlp_sup0.2", a.fold, "softmax")}

    from torchvision import transforms
    from torchvision.transforms import functional as TF
    NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    gauss = center_map()

    rec = []
    for i in range(len(ev)):
        r = ev.iloc[i]
        im = Image.open(r["image_path"]).convert("RGB").resize((IMG_SIZE, IMG_SIZE),
                                                               Image.BILINEAR)
        mk = np.array(Image.open(os.path.join(MASK_DIR, f"{r['image_id']}_segmentation.png"))
                      .convert("L").resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)) > 127
        x = NORM(TF.to_tensor(im))[None].to(DEVICE).requires_grad_(True)

        store = {}
        h = raw.stages[a.gc_stage - 1].register_forward_hook(
            lambda m, i_, o: (o.retain_grad(), store.__setitem__("s", o))[1])
        with torch.enable_grad():
            f, s2, _ = feats(net, x)
            z = net.forward_head(f); pred = int(z.argmax(1))
            z[0, pred].backward()
            hs = store["s"]
            gc = up(gradcam(hs.detach(), hs.grad,
                            nhwc=(hs.shape[-1] != hs.shape[-2])))[0].cpu().numpy()
        h.remove(); raw.zero_grad(set_to_none=True)

        maps = {"gc": gc, "gauss": gauss}
        with torch.no_grad():
            macro = net.forward_head(f.detach(), pre_logits=True)
            micro = s2.detach()
            for key, hd in heads.items():
                _, sal = hd(macro.float(), micro.float())
                v = sal.detach().reshape(-1).cpu().numpy()
                g = int(round(v.size ** 0.5))
                if g * g != v.size:
                    raise SystemExit(f"map of {v.size} positions is not square")
                maps[key] = up(torch.tensor(v.reshape(1, g, g)))[0].cpu().numpy()

        area = float(mk.mean())
        yy, xx = np.where(mk)
        cy, cx = (yy.mean(), xx.mean()) if len(yy) else (IMG_SIZE / 2, IMG_SIZE / 2)
        off = float(np.hypot(cy - IMG_SIZE / 2, cx - IMG_SIZE / 2) / IMG_SIZE)
        ious = {k: iou_area_matched(v, mk) for k, v in maps.items()}
        peak = float(norm01(maps["unsup"]).max() / max(norm01(maps["unsup"]).mean(), 1e-9))
        rec.append(dict(i=i, id=str(r["image_id"]), label=int(r["label"]), pred=pred,
                        area=area, off=off, iou=ious["unsup"], ious=ious, peak=peak,
                        img=np.asarray(im), mask=mk, **maps))

    # improve.fig_qual's selection, unchanged
    df = rec
    mid = [d for d in df if 0.25 <= d["area"] <= 0.55 and d["off"] < 0.08]
    big = max(mid or df, key=lambda d: -abs(d["area"] - 0.4))
    sm = [d for d in df if d["area"] < 0.12 and d["off"] > 0.12]
    small = max(sm or df, key=lambda d: d["off"])
    used = {big["id"], small["id"]}
    cand = [d for d in df if d["id"] not in used and d["area"] >= 0.2]
    cand = sorted(cand, key=lambda d: -d["peak"])[:max(5, len(cand) // 5)]
    plaus = min(cand or df, key=lambda d: d["iou"])
    picks = [("(a) large, central lesion", big),
             ("(b) small, off-center lesion", small),
             ("(c) large lesion, attention elsewhere", plaus)]
    used |= {plaus["id"]}
    wrong = [d for d in df if d["pred"] != d["label"] and d["id"] not in used
             and d["area"] >= 0.15]
    if wrong:
        picks.append(("(d) misclassified case", max(wrong, key=lambda d: d["area"])))

    # Two-line titles and a figure drawn at the width it is printed at: a panel scaled down
    cols = [("image", None), ("lesion\nmask", None), ("centered\nGaussian", "gauss"),
            (f"Grad-CAM\n(stage-{a.gc_stage})", "gc"),
            ("explanation\nhead", "unsup"), ("+ mask\nsupervision", "sup")]
    W = 5.2
    cell = W / len(cols)
    fig, axes = plt.subplots(len(picks), len(cols),
                             figsize=(W, cell * len(picks) + 0.30), dpi=400)
    axes = np.atleast_2d(axes)
    for r_, (lab, d) in enumerate(picks):
        for c_, (title, key) in enumerate(cols):
            t = title if r_ == 0 else ""
            if key is None and c_ == 0:
                overlay(axes[r_, c_], d["img"], title=t,
                        tag=f"{CLS[d['label']]}→{CLS[d['pred']]}")
            elif key is None:
                overlay(axes[r_, c_], d["img"], mask=d["mask"], title=t)
            else:
                overlay(axes[r_, c_], d["img"], m=d[key], mask=d["mask"], title=t,
                        tag=f"IoU {d['ious'][key]:.2f}")
        # The panel letter only. The descriptions are long enough to collide between rows, and
        # they belong in the caption, which can give them in full.
        axes[r_, 0].set_ylabel(lab.split(")")[0] + ")", fontsize=8, rotation=0,
                               labelpad=10, va="center", ha="right")
    fig.tight_layout(pad=0.2, w_pad=0.1, h_pad=0.18)
    for ext in ("png", "pdf"):
        fig.savefig(f"{a.out}.{ext}", bbox_inches="tight")
    print(f"  wrote {a.out}.png / .pdf")

    # where these cases sit in the pool, so the caption can say so
    pool = np.array([d["iou"] for d in df], dtype=float)
    side = {"fold": a.fold, "pool_n": len(df),
            "pool_median_iou": float(np.nanmedian(pool)),
            "pool_below_0.10": int(np.nansum(pool < 0.10)),
            "gc_stage": a.gc_stage,
            "cases": [{"panel": lab, "id": d["id"],
                       "true": CLS[d["label"]], "pred": CLS[d["pred"]],
                       "area": round(d["area"], 4),
                       "iou": {k: (None if np.isnan(v) else round(v, 4))
                               for k, v in d["ious"].items()},
                       "percentile_of_head_iou":
                           round(100.0 * float(np.nanmean(pool < d["iou"])), 1)}
                      for lab, d in picks]}
    json.dump(side, open(f"{a.out}.json", "w"), indent=1)
    print(f"  wrote {a.out}.json")
    print(f"  pool: median head IoU {side['pool_median_iou']:.3f}, "
          f"{side['pool_below_0.10']} of {len(df)} below 0.10")
    for c in side["cases"]:
        print(f"    {c['panel']:<38}{c['id']}  {c['true']}->{c['pred']}  "
              f"head IoU {c['iou']['unsup']}  ({c['percentile_of_head_iou']}th pct)")


if __name__ == "__main__":
    main()
