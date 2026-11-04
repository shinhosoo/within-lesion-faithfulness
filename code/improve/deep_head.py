"""H3: the head may be reading the wrong layer.

    python -m improve.deep_head --fold 0 --limit 64 # smoke test
    python -m improve.deep_head --fold 0
    python -m improve.deep_head --fold 0 --supervise 0.2
    python -m improve.deep_head --fold 0 --stage 4
"""

import argparse
import os
import time

import numpy as np
import torch

from SkinCancer.configs.training_config import DEVICE
from SkinCancer.data.split_io import build_image_path_map
from SkinCancer.utils.seed import set_seed

from improve.head_swap import (OUT, build_head, check_splits, counts, eval_head_live, faith_of,
                               load_flat, probe, source_id, train_head_live, train_head_sup, seed_out)
from improve.residual import eval_split, get_data


class StageView:
    """The backbone with one of its later stages presented where the second normally is."""

    def __init__(self, net, stage):
        self._net = net
        self._k = stage - 1                     # stage 3 is index 2, as improve.localize reads it

    def __getattr__(self, name):
        return getattr(self._net, name)

    def forward_intermediates(self, x, **kw):
        f, st = self._net.forward_intermediates(x, **kw)
        st = list(st)
        if self._k >= len(st):
            raise SystemExit(f"this backbone has {len(st)} intermediates; stage "
                             f"{self._k + 1} does not exist")
        st[1] = st[self._k]
        return f, st


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
    net = StageView(raw, a.stage)
    s_dim, final_dim, hw = probe(net)
    if a.supervise and hw[0] != hw[1]:
        raise SystemExit(f"the mask grid assumes a square map, got {hw}")

    # The same rule head_swap.py applies: a supervised run reports the distribution its loss acts
    # on, an unsupervised one a relu score. Breaking it would make this row incomparable.
    mode = a.map if a.map else ("softmax" if a.supervise else "relu")
    head = build_head(a.head, net, s_dim, final_dim, 7, 4, False, mode)
    total, mech = counts(head)
    stem = f"s{a.stage}_{a.head}" + (f"_sup{a.supervise:g}" if a.supervise else "")
    print(f"[deep_head] fold {a.fold} seed {a.seed}  train={len(tr)} val={len(va)} "
          f"eval={len(ev)}  stage-{a.stage} map {s_dim}x{hw[0]}x{hw[1]}  macro {final_dim}")
    print(f"[deep_head] produced by deep_head.py {source_id()}")
    print(f"  [{stem}] total {total:,} parameters, saliency mechanism {mech:,}"
          + (f", mask supervision lambda={a.supervise:g}" if a.supervise else ""))
    print(f"  [{stem}] for reference the stage-2 run of this head reads 192x48x48, so the map "
          f"here is {'coarser' if hw[0] < 48 else 'the same resolution or finer'}")

    t0 = time.time()
    out = seed_out(a.seed)
    train_tag = f"{stem}_f{a.fold}" + (f"_lim{a.limit}" if a.limit else "")

    if a.measure_only:
        # Measure a head that is already trained, under a map mode that may differ from the one
        ck = os.path.join(out, f"head_{train_tag}.pth")
        if not os.path.exists(ck):
            raise SystemExit(f"--measure-only needs a trained head at {ck}")
        head.load_state_dict(torch.load(ck, map_location=DEVICE, weights_only=True), strict=True)
        head.eval()
        acc = -1.0                      # not retrained, so there is no validation accuracy
        stem = f"{stem}_map{mode}"
        tag = f"{stem}_f{a.fold}" + (f"_lim{a.limit}" if a.limit else "")
        print(f"  [{stem}] measure-only: loaded {os.path.basename(ck)}, "
              f"reading its map as {mode}")
    else:
        if a.supervise:
            acc = train_head_sup(head, net, s_dim, tr, va, a.epochs, a.lr, a.supervise,
                                 hw[0], a.quiet, a.select)
        else:
            acc = train_head_live(head, net, s_dim, tr, va, a.epochs, a.lr, a.quiet)
        tag = train_tag
        torch.save(head.state_dict(), os.path.join(out, f"head_{tag}.pth"))

    R, ids, labels, fpred, rpred, areas = eval_head_live(head, net, s_dim, ev)
    fa, n_fa, n_w = faith_of(net, head, s_dim, ev, a.faith_n, a.faith_steps)
    print(f"  [{stem}] Ins-Del {fa['head'][2]:+.4f} over {n_fa} images "
          f"(centered Gaussian {fa['center'][2]:+.4f})")
    print(f"  [{stem}] within-lesion Ins-Del {fa['head'][5]:+.4f} over {n_w} images "
          f"(centered Gaussian {fa['center'][5]:+.4f})")

    path = os.path.join(out, f"hs_{tag}.npz")
    np.savez(path, metrics=R, names=np.array([stem, "center"]), image_id=ids, label=labels,
             flat_pred=fpred, readout_pred=rpred, lesion_area=areas,
             params_total=total, params_mechanism=mech, readout_val_acc=acc,
             stage=a.stage, stage_dim=s_dim, map_grid=hw[0], map_mode=mode,
             source_sha1=source_id(), select=a.select if a.supervise else "",
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
    print("    The row to read this against is the same head at stage 2: unsupervised IoU 0.271")
    print("    with a within-lesion value of -0.126, mask-supervised IoU 0.818 with -0.012. And")
    print("    the ceiling for this depth is a gradient read of the same layer, which for stage")
    print("    3 is +0.137. A head that stays negative here says the depth was not the problem.")
    print(f"    {time.time() - t0:.0f}s   saved: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stage", type=int, default=3, choices=[2, 3, 4],
                    help="which backbone stage the head reads; 2 reproduces the existing rows")
    ap.add_argument("--head", default="mlp", choices=["mlp", "cam", "lowrank", "ssf"])
    ap.add_argument("--epochs", type=int, default=10, help="as improve.localize uses")
    ap.add_argument("--lr", type=float, default=5e-4, help="as improve.localize uses")
    ap.add_argument("--limit", type=int, default=0, help="smoke test on a few images per class")
    ap.add_argument("--supervise", type=float, default=0.0, metavar="LAM",
                    help="weight on the mask alignment term; 0 trains from the labels alone")
    ap.add_argument("--map", default=None, choices=["relu", "softmax"])
    ap.add_argument("--measure-only", action="store_true",
                    help="load the trained head for this fold instead of training, and measure "
                         "it; writes to a stem carrying the map mode so the training run's "
                         "archive is never overwritten")
    ap.add_argument("--select", default="att", choices=["att", "auc"])
    ap.add_argument("--faith-n", type=int, default=400)
    ap.add_argument("--faith-steps", type=int, default=10)
    ap.add_argument("--quiet", action="store_true")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
