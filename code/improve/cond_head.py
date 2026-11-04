"""The image-conditional mechanism of the ladder: a linear score whose channel weights come from the image.

    python -m improve.cond_head --fold 0
    python -m improve.cond_head --fold 0 --supervise 0.2
    python -m improve.cond_head --fold 0 --stage 3 --supervise 0.2
    python -m improve.cond_head --fold 0 --limit 64 # smoke test, writes _lim64 files
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn

from SkinCancer.configs.training_config import DEVICE
from SkinCancer.data.split_io import build_image_path_map
from SkinCancer.utils.seed import set_seed

from improve.deep_head import StageView
from improve.head_swap import (OUT, Scaffold, check_splits, counts, eval_head_live, faith_of,
                               load_flat, probe, source_id, train_head_live, train_head_sup, seed_out)
from improve.residual import eval_split, get_data


class CondHead(Scaffold):
    """A linear score over positions whose channel weights are produced from the image."""

    def __init__(self, s2_dim, final_dim, n_out, hidden=32, map_mode="relu"):
        super().__init__(s2_dim, final_dim, n_out)
        self.norm = nn.GroupNorm(1, s2_dim)
        self.gen = nn.Sequential(nn.Linear(final_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, s2_dim))
        self.map_mode = map_mode

    def mechanism(self):
        return list(self.norm.parameters()) + list(self.gen.parameters())

    def forward(self, macro, micro, cls=None):
        w = self.gen(macro)
        raw = torch.einsum("bc,bchw->bhw", w, self.norm(micro))
        B, H, W = raw.shape
        a = torch.softmax(raw.reshape(B, H * W), dim=1)
        pooled = torch.einsum("bchw,bhw->bc", micro, a.reshape(B, H, W))
        shown = a.reshape(B, H, W) if self.map_mode == "softmax" else torch.relu(raw)
        return self.classifier(self.fuse(macro, pooled)), shown


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
    # The two interventions of the hypothesis section are independent, so the cell that tests
    net = StageView(raw, a.stage) if a.stage != 2 else raw
    s2_dim, final_dim, hw = probe(net)
    if a.supervise and hw[0] != hw[1]:
        raise SystemExit(f"the mask grid assumes a square map, got {hw}")

    # The map a supervised run reports is the distribution its loss acts on, and a relu score
    mode = a.map if a.map else ("softmax" if a.supervise else "relu")
    head = CondHead(s2_dim, final_dim, 7, a.hidden, mode).to(DEVICE)
    total, mech = counts(head)
    stem = (f"s{a.stage}_" if a.stage != 2 else "") + "cond" \
        + (f"_sup{a.supervise:g}" if a.supervise else "")
    print(f"[cond_head] fold {a.fold} seed {a.seed}  train={len(tr)} val={len(va)} "
          f"eval={len(ev)}  stage-{a.stage} map {s2_dim}x{hw[0]}x{hw[1]}  macro {final_dim}")
    print(f"[cond_head] produced by cond_head.py {source_id()}")
    print(f"  [{stem}] total {total:,} parameters, saliency mechanism {mech:,}, hidden {a.hidden}"
          + (f", mask supervision lambda={a.supervise:g}" if a.supervise else ""))

    t0 = time.time()
    out = seed_out(a.seed)
    train_tag = f"{stem}_f{a.fold}" + (f"_lim{a.limit}" if a.limit else "")

    if a.measure_only:
        # Re-read a trained head under a map mode its own run did not report. The weights are
        ck = os.path.join(out, f"head_{train_tag}.pth")
        if not os.path.exists(ck):
            raise SystemExit(f"--measure-only needs a trained head at {ck}")
        head.load_state_dict(torch.load(ck, map_location=DEVICE, weights_only=True), strict=True)
        head.eval()
        acc = -1.0
        stem = f"{stem}_map{mode}"
        tag = f"{stem}_f{a.fold}" + (f"_lim{a.limit}" if a.limit else "")
        print(f"  [{stem}] measure-only: loaded {os.path.basename(ck)}, "
              f"reading its map as {mode}")
    else:
        if a.supervise:
            acc = train_head_sup(head, net, s2_dim, tr, va, a.epochs, a.lr, a.supervise,
                                 hw[0], a.quiet, a.select)
        else:
            acc = train_head_live(head, net, s2_dim, tr, va, a.epochs, a.lr, a.quiet)
        # Saved before anything else that could fail, because training is the only expensive
        # part here and a later error must not throw it away.
        tag = train_tag
        torch.save(head.state_dict(), os.path.join(out, f"head_{tag}.pth"))

    R, ids, labels, fpred, rpred, areas = eval_head_live(head, net, s2_dim, ev)
    fa, n_fa, n_w = faith_of(net, head, s2_dim, ev, a.faith_n, a.faith_steps)
    print(f"  [{stem}] Ins-Del {fa['head'][2]:+.4f} over {n_fa} images "
          f"(centered Gaussian {fa['center'][2]:+.4f})")
    print(f"  [{stem}] within-lesion Ins-Del {fa['head'][5]:+.4f} over {n_w} images "
          f"(centered Gaussian {fa['center'][5]:+.4f})")

    path = os.path.join(out, f"hs_{tag}.npz")
    # The stem, not the literal "cond": a stage-3 run has to be distinguishable from the stage-2
    # rows inside the file as well as in its name, or the aggregation reads it as that row.
    np.savez(path, metrics=R, names=np.array([stem, "center"]), image_id=ids, label=labels,
             flat_pred=fpred, readout_pred=rpred, lesion_area=areas,
             params_total=total, params_mechanism=mech, readout_val_acc=acc,
             hidden=a.hidden, map_mode=mode, stage=a.stage, source_sha1=source_id(),
             select=a.select if a.supervise else "",
             faith_head=np.array(fa["head"]), faith_center=np.array(fa["center"]),
             faith_n=a.faith_n, faith_n_within=n_w)

    mine = [float(np.nanmean(R[:, 0, c])) for c in range(3)]
    ref = [float(np.nanmean(R[:, 1, c])) for c in range(3)]
    print()
    print(f"    {'map':<22}{'Pointing':>10}{'Energy':>9}{'IoU':>9}{'Ins-Del':>10}{'within':>10}")
    print(f"    {stem:<22}{mine[0]:>10.4f}{mine[1]:>9.4f}{mine[2]:>9.4f}"
          f"{fa['head'][2]:>+10.4f}{fa['head'][5]:>+10.4f}")
    print(f"    {'center':<22}{ref[0]:>10.4f}{ref[1]:>9.4f}{ref[2]:>9.4f}"
          f"{fa['center'][2]:>+10.4f}{fa['center'][5]:>+10.4f}")
    print()
    print("    The centered Gaussian is the check and must match the value already reported for")
    print("    this fold. Then read `within` against the five mechanisms already measured, all")
    print("    of which are negative, and against Grad-CAM at stage 4, which is positive. A")
    print("    conditional weighting that stays negative says the objective is the constraint,")
    print("    not the architecture.")
    print(f"    {time.time() - t0:.0f}s   saved: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hidden", type=int, default=32, help="width of the weight generator")
    ap.add_argument("--stage", type=int, default=2, choices=[2, 3, 4],
                    help="which backbone stage the head reads; 2 reproduces the reported rows")
    ap.add_argument("--epochs", type=int, default=10, help="as improve.localize uses")
    ap.add_argument("--lr", type=float, default=5e-4, help="as improve.localize uses")
    ap.add_argument("--limit", type=int, default=0, help="smoke test on a few images per class")
    ap.add_argument("--supervise", type=float, default=0.0, metavar="LAM",
                    help="weight on the mask alignment term; 0 trains from the labels alone")
    ap.add_argument("--measure-only", action="store_true",
                    help="load the trained head for this fold instead of training, and measure it; writes to a stem carrying the map mode so the "
                         "training run's archive is never overwritten")
    ap.add_argument("--map", default=None, choices=["relu", "softmax"],
                    help="what the head reports as its map; defaults as head_swap.py defaults")
    ap.add_argument("--select", default="att", choices=["att", "auc"],
                    help="what a supervised run chooses its epoch on")
    ap.add_argument("--faith-n", type=int, default=400)
    ap.add_argument("--faith-steps", type=int, default=10)
    ap.add_argument("--quiet", action="store_true", help="suppress the per-epoch lines")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
