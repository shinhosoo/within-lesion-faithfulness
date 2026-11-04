"""Four saliency mechanisms on one frozen backbone, measured against the same lesion masks.

The explanation measurements in the paper train a readout on top of a frozen fold backbone and
score its map against the lesion masks. The question this module asks is whether the result is
a property of any particular mechanism or of the task: if a readout whose saliency mechanism is
three orders of magnitude smaller localises the lesion just as well, then the measured overlap
says nothing about the mechanism.

Four readouts are compared, and they differ only in how a per-position score is produced:

  lowrank    a rank-r linear score, which is the factorisation LoRA uses: s2_dim -> r -> 1. The
             score is content-independent, a fixed learned direction rather than a query derived
             from the image.
  ssf        per-channel scale and shift, the smallest of the parameter-efficient baselines. Its
             map costs nothing extra: the same scales, summed over channels, are the map.
  mlp        a position-wise two-layer score, the smallest head here whose map is not a single
             plane in feature space.
  cam        one map per class, read at the class in question, which is the only head here whose
             map is class-conditional.

None of them has a convolution, a query or an attention head. What they share is only
the scaffolding the rest of the pipeline requires -- the two fusion projections and the
classifier -- so the parameter difference between the rows is the saliency mechanism and
nothing else.

Why this is cheap. The backbone is frozen, and improve.localize nonetheless runs it again on
every image of every epoch. Caching its output once makes the readout training almost free, so
every head and fold finishes in the time one backbone pass takes. The cache is built with the
deterministic validation transform, which removes the training augmentation, so the comparison
is only fair if every head is trained the same way. The live path trains and scores through the
backbone instead of the cache, which is how the shortcut itself is checked.

Nothing here writes where improve.localize writes, and nothing here is imported by it.

    python -m improve.head_swap --count                       # parameters, no GPU, no data
    python -m improve.head_swap --fold 0 --head ssf
    python -m improve.head_swap --fold 0 --head all
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn

from SkinCancer.configs.model_config import BACKBONE_NAME, D_ATTN, IMG_SIZE
from SkinCancer.configs.training_config import DEVICE, OUTPUT_ROOT
from SkinCancer.data.loaders import make_loader
from SkinCancer.data.split_io import build_image_path_map
from SkinCancer.data.transforms import build_train_transform, build_val_transform
from SkinCancer.utils.seed import set_seed

# center_map and metrics come through improve.localize, which defines them where the overlap
# measures are not yet a separate module and re-exports them where they are, so this import
# works on either layout and scores with the same two functions the paper's numbers came from.
from improve.localize import center_map, feats, load_flat, metrics, up
from improve.residual import eval_split, get_data, inv_freq

OUT = os.path.join(OUTPUT_ROOT, "headswap")
CACHE = os.path.join(OUT, "cache")
MASK_DIR = os.environ.get("HAM_MASK_DIR", "masks_ham10000")
HEADS = ["lowrank", "ssf", "mlp", "cam"]

# Every head answers forward(macro, micro, cls=None) and returns (logits, map). `cls` names
# the class whose map is wanted and only the class-conditional head reads it, but no caller
# in this module passes it. eval_head_live and train_head_sup both call the head without it,
# so that head falls back to its own argmax in training and in evaluation alike. The other
# heads produce one map regardless, which is itself the property this table measures.


# --------------------------------------------------------------------------------------
# the readouts
# --------------------------------------------------------------------------------------

class Scaffold(nn.Module):
    """What every head needs in order to reach the classifier, and what none of them is judged on.

    The classifier takes a D_ATTN vector, so a macro projection, a micro projection and the
    output norm have to exist whatever produces the saliency. These are the same three layers
    the reported configuration uses. Holding them fixed across the heads is what makes the
    parameter counts comparable: the difference between two rows is the mechanism, not the
    plumbing around it.
    """

    def __init__(self, s2_dim, final_dim, n_out):
        super().__init__()
        self.macro_proj = nn.Linear(final_dim, D_ATTN)
        self.micro_vec = nn.Linear(s2_dim, D_ATTN)
        self.norm_out = nn.LayerNorm(D_ATTN)
        self.classifier = nn.Sequential(nn.Linear(D_ATTN, 128), nn.LayerNorm(128), nn.ReLU(),
                                        nn.Dropout(0.3), nn.Linear(128, n_out))

    def fuse(self, macro, pooled):
        return self.norm_out(self.macro_proj(macro) + self.micro_vec(pooled))


class LowRankHead(Scaffold):
    """A rank-r linear score over positions: the factorisation LoRA uses, as a saliency map.

    `down` carries no bias, so the score is a pure linear functional of the channel vector at
    each position, read through an r-dimensional bottleneck. It is not an attention: there is no
    query, so the score of a position does not depend on the rest of the image, and the same
    learned direction is applied to every image. That is the property worth reporting -- if a
    fixed direction localises as well as a macro-conditioned query, the conditioning is not what
    the overlap is measuring.
    """

    def __init__(self, s2_dim, final_dim, n_out, rank=4, map_mode="relu"):
        super().__init__(s2_dim, final_dim, n_out)
        self.down = nn.Conv2d(s2_dim, rank, kernel_size=1, bias=False)
        self.score = nn.Conv2d(rank, 1, kernel_size=1)
        self.map_mode = map_mode

    def mechanism(self):
        return list(self.down.parameters()) + list(self.score.parameters())

    def forward(self, macro, micro, cls=None):
        raw = self.score(self.down(micro))[:, 0]                      # B,H,W
        B, H, W = raw.shape
        w = torch.softmax(raw.reshape(B, H * W), dim=1)
        pooled = torch.einsum("bchw,bhw->bc", micro, w.reshape(B, H, W))
        # Two ways to report the same score. `softmax` reports the weights the head actually
        # pools with, which is the analogue of the published module's attention weights and the
        # object mask supervision acts on; `relu` reports the rectified score, which is the
        # convention this repository applies to Grad-CAM. The pointing score and the
        # area-matched IoU are the same either way, both being functions of the ranking alone;
        # the energy ratio is not, since it weights by the values.
        shown = w.reshape(B, H, W) if self.map_mode == "softmax" else torch.relu(raw)
        return self.classifier(self.fuse(macro, pooled)), shown


class SSFHead(Scaffold):
    """Per-channel scale and shift, with the scales doubling as the map.

    Scale-and-shift tuning is the smallest of the parameter-efficient baselines: two numbers per
    channel and nothing else. The map is free rather than cheap -- summing the scaled channels is
    the same linear functional the pooled vector is built from, so no parameter exists that only
    serves the saliency. If this row matches the published one, the overlap is reproduced by a
    mechanism that was not designed to localise anything.
    """

    def __init__(self, s2_dim, final_dim, n_out, signed=False, map_mode="relu"):
        super().__init__(s2_dim, final_dim, n_out)
        self.gamma = nn.Parameter(torch.ones(s2_dim))
        self.beta = nn.Parameter(torch.zeros(s2_dim))
        self.signed = signed
        self.map_mode = map_mode

    def mechanism(self):
        return [self.gamma, self.beta]

    def forward(self, macro, micro, cls=None):
        h = micro * self.gamma[None, :, None, None] + self.beta[None, :, None, None]
        sal = h.sum(1)                                                # B,H,W
        # The channel sum is signed, unlike attention weights, so it needs a convention. The
        # default is the one this repository already applies to Grad-CAM, whose activations are
        # signed for the same reason: negative positions are evidence against the class, not
        # evidence of a different location. `--signed` keeps them instead and lets
        # improve.overlap's own shift handle the range, which is the alternative worth having
        # on hand because the choice is a convention and not a measurement.
        if self.map_mode == "softmax":
            B, H, W = sal.shape
            shown = torch.softmax(sal.reshape(B, H * W), dim=1).reshape(B, H, W)
        else:
            shown = sal if self.signed else torch.relu(sal)
        return self.classifier(self.fuse(macro, h.mean((2, 3)))), shown


class MLPHead(Scaffold):
    """A position-wise MLP score, which is the smallest head whose map is not a single plane.

    The rank-4 score of LowRankHead collapses: a bottleneck followed by a scalar output is one
    linear functional of the channel vector, whatever the rank, so its 773 parameters buy 193
    degrees of freedom and its map is a single hyperplane in feature space. That is why its
    alignment loss stops falling after three epochs. Putting a nonlinearity in the bottleneck
    removes the collapse without adding any spatial mechanism: the score of a position still
    depends on that position's channels alone, through a two-layer network applied pointwise.
    """

    def __init__(self, s2_dim, final_dim, n_out, hidden=32, map_mode="relu"):
        super().__init__(s2_dim, final_dim, n_out)
        self.norm = nn.GroupNorm(1, s2_dim)
        self.fc1 = nn.Conv2d(s2_dim, hidden, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden, 1, kernel_size=1)
        self.map_mode = map_mode

    def mechanism(self):
        return (list(self.norm.parameters()) + list(self.fc1.parameters())
                + list(self.fc2.parameters()))

    def forward(self, macro, micro, cls=None):
        raw = self.fc2(torch.nn.functional.gelu(self.fc1(self.norm(micro))))[:, 0]
        B, H, W = raw.shape
        w = torch.softmax(raw.reshape(B, H * W), dim=1)
        pooled = torch.einsum("bchw,bhw->bc", micro, w.reshape(B, H, W))
        shown = w.reshape(B, H, W) if self.map_mode == "softmax" else torch.relu(raw)
        return self.classifier(self.fuse(macro, pooled)), shown


class CAMHead(Scaffold):
    """One map per class, read at the class in question: the mechanism of class activation maps.

    Every other head here produces a single map whatever the model predicted, and a map that
    does not depend on the predicted class cannot distinguish the evidence for one class from
    the evidence for another. Whether that is why their faithfulness sits at zero is testable
    only with a head whose map is class-conditional, and the established way to build one is a
    pointwise linear layer with one output per class, which is what this is.
    """

    def __init__(self, s2_dim, final_dim, n_out, map_mode="relu"):
        super().__init__(s2_dim, final_dim, n_out)
        self.norm = nn.GroupNorm(1, s2_dim)
        self.score = nn.Conv2d(s2_dim, n_out, kernel_size=1)
        self.n_out = n_out
        self.map_mode = map_mode

    def mechanism(self):
        return list(self.norm.parameters()) + list(self.score.parameters())

    def forward(self, macro, micro, cls=None):
        maps = self.score(self.norm(micro))                           # B,C,H,W
        B, C, H, W = maps.shape
        # Pooling uses the class-mean attention so that the pooled vector does not depend on a
        # prediction that has not been made yet; only the reported map is class-conditional.
        w = torch.softmax(maps.mean(1).reshape(B, H * W), dim=1)
        pooled = torch.einsum("bchw,bhw->bc", micro, w.reshape(B, H, W))
        logits = self.classifier(self.fuse(macro, pooled))
        pick = logits.argmax(1) if cls is None else cls
        raw = maps[torch.arange(B, device=maps.device), pick.clamp(0, C - 1)]
        if self.map_mode == "softmax":
            shown = torch.softmax(raw.reshape(B, H * W), dim=1).reshape(B, H, W)
        else:
            shown = torch.relu(raw)
        return logits, shown


def build_head(name, net, s2_dim, final_dim, n_out, rank, signed=False, map_mode="relu"):
    if name == "lowrank":
        return LowRankHead(s2_dim, final_dim, n_out, rank, map_mode).to(DEVICE)
    if name == "ssf":
        return SSFHead(s2_dim, final_dim, n_out, signed, map_mode).to(DEVICE)
    if name == "mlp":
        return MLPHead(s2_dim, final_dim, n_out, 32, map_mode).to(DEVICE)
    if name == "cam":
        return CAMHead(s2_dim, final_dim, n_out, map_mode).to(DEVICE)
    raise SystemExit(f"unknown head {name}")


def seed_out(seed, base=None):
    """Where a run of this seed writes, mirroring improve.localize's convention.

    Every seed but the default gets a subdirectory of its own. Without this a second seed
    overwrites the first in place, and because the readout seed changes neither the mechanism
    size, nor the image counts, nor this module's hash, nor the centred Gaussian, there is no
    check anywhere that would notice.
    """
    d = OUT if base is None else base
    if int(seed) != 42:
        d = os.path.join(d, f"seed{int(seed)}")
    os.makedirs(d, exist_ok=True)
    return d


def source_id(n=10):
    """A short hash of this file, recorded in every result it writes.

    Two results that disagree are otherwise hard to tell from one result written twice by
    different versions of this module, and this study has already added measurements to rows
    that were produced by an earlier version. The stamp makes the provenance of each row
    readable from the row itself, as verify_all.py does for the table dump.
    """
    import hashlib
    try:
        with open(os.path.abspath(__file__), "rb") as fh:
            return hashlib.sha1(fh.read()).hexdigest()[:n]
    except OSError:                                                # noqa: BLE001
        return "unknown"


def counts(head):
    total = sum(p.numel() for p in head.parameters())
    mech = sum(p.numel() for p in head.mechanism())
    return total, mech


# --------------------------------------------------------------------------------------
# the cache
# --------------------------------------------------------------------------------------

def probe(net):
    """The two widths the heads need, read the way improve.localize reads them."""
    with torch.no_grad():
        f, s2, _ = feats(net, torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE))
        if s2.dim() != 4:
            raise SystemExit(f"the backbone gave a stage-2 map of {s2.dim()} dimensions; "
                             f"the cache layout assumes N,C,H,W")
        s2_dim = s2.shape[1]
        final_dim = net.forward_head(f, pre_logits=True).shape[1]
    return s2_dim, final_dim, (s2.shape[2], s2.shape[3])


def build_cache(net, df, s2_dim, final_dim, hw, path):
    """One backbone pass, stored as float16 memmaps.

    The backbone is frozen, so its output for an image never changes and recomputing it each
    epoch buys nothing. Everything downstream reads these three arrays, which is why a head's
    training cost stops depending on the backbone at all.
    """
    n = len(df)
    H, W = hw
    micro = np.lib.format.open_memmap(path + ".micro.npy", mode="w+", dtype=np.float16,
                                      shape=(n, s2_dim, H, W))
    macro = np.lib.format.open_memmap(path + ".macro.npy", mode="w+", dtype=np.float16,
                                      shape=(n, final_dim))
    logit = np.lib.format.open_memmap(path + ".logit.npy", mode="w+", dtype=np.float32,
                                      shape=(n, 7))
    loader = make_loader(df, "label", build_val_transform(), shuffle=False)
    t0, k = time.time(), 0
    with torch.no_grad():
        for img, _ in loader:
            img = img.to(DEVICE)
            with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                f, s2, _ = feats(net, img)
                m = s2
                mac = net.forward_head(f, pre_logits=True)
                lg = net.forward_head(f)
            b = img.shape[0]
            micro[k:k + b] = m.float().cpu().numpy().astype(np.float16)
            macro[k:k + b] = mac.float().cpu().numpy().astype(np.float16)
            logit[k:k + b] = lg.float().cpu().numpy()
            k += b
    for arr in (micro, macro, logit):
        arr.flush()
    gb = (micro.nbytes + macro.nbytes + logit.nbytes) / 2 ** 30
    print(f"  cached {k} images in {time.time() - t0:.0f}s, {gb:.2f} GiB -> {path}.*.npy")
    return k


def drop_cache(fold, suffix=""):
    """Remove one fold's cache and report what that freed.

    The cache is an intermediate and nothing reported depends on it: the overlap measurements
    are written to their own npz files, and a dropped cache is rebuilt by one backbone pass.
    Only this fold's files are touched, so a run over several folds can free each as it goes.
    """
    freed = 0
    for split in ("train", "val", "eval"):
        for arr in ("micro", "macro", "logit"):
            # Named exactly rather than globbed: a pattern wide enough to catch these also
            # catches a reduced run's separately named cache, which is not this one's to remove.
            p = os.path.join(CACHE, f"f{fold}_{split}{suffix}.{arr}.npy")
            if os.path.exists(p):
                try:
                    freed += os.path.getsize(p)
                    os.remove(p)
                except OSError:
                    pass
    return freed


def open_cache(path):
    return (np.load(path + ".micro.npy", mmap_mode="r"),
            np.load(path + ".macro.npy", mmap_mode="r"),
            np.load(path + ".logit.npy", mmap_mode="r"))


def cache_fold(net, fold, s2_dim, final_dim, hw, dfs, rebuild, suffix=""):
    """Build whichever split caches are missing, and return their handles.

    A reduced run gets its own cache names, so the smoke test cannot leave a short cache where
    the real run expects a full one. Where a cache is found, its row count is checked against
    the split rather than trusted, because a cache that silently holds the wrong images would
    produce results that look fine.
    """
    os.makedirs(CACHE, exist_ok=True)
    out = {}
    for name, df in dfs.items():
        path = os.path.join(CACHE, f"f{fold}_{name}{suffix}")
        if rebuild or not os.path.exists(path + ".micro.npy"):
            print(f"  building cache for {name} ({len(df)} images)")
            build_cache(net, df, s2_dim, final_dim, hw, path)
        got = open_cache(path)
        if got[0].shape[0] != len(df):
            raise SystemExit(f"cache {path} holds {got[0].shape[0]} rows, split has {len(df)}; "
                             f"re-run with --rebuild-cache")
        out[name] = got
    return out


# --------------------------------------------------------------------------------------
# training and evaluation on the cache
# --------------------------------------------------------------------------------------

def batches(n, bs, shuffle, rng=None):
    idx = np.arange(n)
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, n, bs):
        yield np.sort(idx[i:i + bs])          # sorted, so the memmap is read in order


def to_dev(cache, sel):
    micro, macro, _ = cache
    return (torch.from_numpy(np.ascontiguousarray(macro[sel])).float().to(DEVICE),
            torch.from_numpy(np.ascontiguousarray(micro[sel])).float().to(DEVICE))


def train_head(head, tr_cache, va_cache, y_tr, y_va, epochs, bs, lr, seed, quiet):
    ce = nn.CrossEntropyLoss(weight=inv_freq(np.bincount(y_tr, minlength=7)))
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=5e-5)
    rng = np.random.default_rng(seed)
    yt = torch.from_numpy(y_tr).long()
    best, state = -1.0, None
    for ep in range(epochs):
        head.train()
        tot, nb = 0.0, 0
        for sel in batches(len(y_tr), bs, True, rng):
            macro, micro = to_dev(tr_cache, sel)
            z, _ = head(macro, micro)
            loss = ce(z, yt[sel].to(DEVICE))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss)
            nb += 1
        head.eval()
        P = []
        with torch.no_grad():
            for sel in batches(len(y_va), bs, False):
                macro, micro = to_dev(va_cache, sel)
                z, _ = head(macro, micro)
                P.append(torch.softmax(z.float(), 1).cpu().numpy())
        P = np.concatenate(P)
        acc = float((P.argmax(1) == y_va).mean())
        # The readout's own accuracy is not a result: its logits are discarded and only its map
        # is measured. It is tracked solely to choose the epoch, exactly as improve.localize does.
        if acc > best:
            best = acc
            state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        if not quiet:
            print(f"    epoch {ep + 1:02d}/{epochs}  loss {tot / max(nb, 1):.4f}  val acc {acc:.4f}")
    if state is not None:
        head.load_state_dict(state)
    return best


def eval_head(head, cache, df, name, bs):
    from PIL import Image
    _, _, logit = cache
    cen = center_map()
    head.eval()
    rows, ids, labels, fpred, rpred, areas = [], [], [], [], [], []
    miss = 0
    with torch.no_grad():
        for sel in batches(len(df), bs, False):
            macro, micro = to_dev(cache, sel)
            z, sal = head(macro, micro)
            M = up(sal).cpu().numpy()
            for j, i in enumerate(sel):
                iid = df.iloc[int(i)]["image_id"]
                mp = os.path.join(MASK_DIR, f"{iid}_segmentation.png")
                if not os.path.exists(mp):
                    miss += 1
                    continue
                mask = np.array(Image.open(mp).convert("L")
                                .resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)) > 127
                rows.append(metrics(np.stack([M[j], cen]), mask))
                ids.append(iid)
                labels.append(int(df.iloc[int(i)]["label"]))
                fpred.append(int(np.asarray(logit[int(i)]).argmax()))
                rpred.append(int(z[j].argmax()))
                areas.append(float(mask.mean()))
    if miss:
        print(f"    [warning] {miss} masks not found under {MASK_DIR} and left out")
    return (np.stack(rows), np.array(ids), np.array(labels),
            np.array(fpred), np.array(rpred), np.array(areas))


# --------------------------------------------------------------------------------------
# entry points
# --------------------------------------------------------------------------------------

def count_only(args):
    """Parameter counts with no data, no checkpoint and no GPU.

    The backbone is created untrained purely to read the two widths the heads depend on, so this
    says what each row will cost before any of them is trained.
    """
    import timm
    net = timm.create_model(BACKBONE_NAME, pretrained=False, num_classes=7).to(DEVICE).eval()
    s2_dim, final_dim, hw = probe(net)
    print(f"backbone {BACKBONE_NAME} at {IMG_SIZE}x{IMG_SIZE}: stage-2 map "
          f"{s2_dim}x{hw[0]}x{hw[1]}, macro width {final_dim}")
    per = s2_dim * hw[0] * hw[1] * 2 + final_dim * 2 + 28
    print(f"cache cost: {per / 2 ** 20:.2f} MiB per image in float16, "
          f"{per * 8000 / 2 ** 30:.1f} GiB for 8,000")
    print()
    print(f"    {'head':<10}{'total':>12}{'mechanism':>12}{'vs first':>10}")
    ref = None
    for name in HEADS:
        try:
            head = build_head(name, net, s2_dim, final_dim, 7, args.rank, map_mode=args.map)
        except Exception as exc:                                   # noqa: BLE001
            print(f"    {name:<10}  could not be built: {type(exc).__name__}: {exc}")
            continue
        total, mech = counts(head)
        if ref is None:
            ref = mech
        print(f"    {name:<10}{total:>12,}{mech:>12,}{mech / max(ref, 1):>9.4f}x")
        del head
    print()
    print("    total is every parameter the readout trains; mechanism is the part that produces")
    print("    the saliency, which is what the rows differ by, and the last column is its size")
    print("    against the first row's. The scaffolding the counts share -- two fusion projections")
    print("    and the classifier -- is the same in all of them and is required by the pipeline,")
    print("    not by any of the mechanisms.")


def train_head_live(head, net, s2_dim, tr, va, epochs, lr, quiet):
    """Train through the frozen backbone each epoch, which is what improve.localize does.

    Caching the backbone's output is only sound if nothing about the input varies between
    epochs, and the training transform is random, so a cached run silently trains without
    augmentation. On this task that is not a small difference: the readout drives its loss to
    near zero within three epochs and its attention map degenerates, scoring below a random map
    of the same area. So the expensive path is the correct one, and the cache is kept only for
    the comparison that showed this.

    Every choice here follows improve.localize rather than convenience: the same two transforms,
    the same optimiser and learning rate, autocast with a gradient scaler, and the epoch chosen
    on the validation macro AUC, so a row measured here is comparable with one measured there.
    """
    from sklearn.metrics import roc_auc_score

    y = tr["label"].astype(int).values
    ce = nn.CrossEntropyLoss(weight=inv_freq(np.bincount(y, minlength=7)))
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=5e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
    tl = make_loader(tr, "label", build_train_transform(), shuffle=True)
    vl = make_loader(va, "label", build_val_transform(), shuffle=False)
    best, state = -1.0, None
    for ep in range(epochs):
        head.train()
        tot, nb = 0.0, 0
        for img, lbl in tl:
            img, lbl = img.to(DEVICE), lbl.to(DEVICE).long()
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                f, s2, _ = feats(net, img)
                macro = net.forward_head(f, pre_logits=True)
                micro = s2
            with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                z, _ = head(macro.float(), micro.float())
                loss = ce(z, lbl)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += float(loss)
            nb += 1
        head.eval()
        P, Y = [], []
        with torch.no_grad():
            for img, lbl in vl:
                with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                    f, s2, _ = feats(net, img.to(DEVICE))
                    z, _ = head(net.forward_head(f, pre_logits=True).float(),
                                s2.float())
                P.append(torch.softmax(z.float(), 1).cpu().numpy())
                Y.append(lbl.numpy())
        P, Y = np.concatenate(P), np.concatenate(Y).astype(int)
        try:
            auc = roc_auc_score(Y, P / P.sum(1, keepdims=True), multi_class="ovr",
                                average="macro")
        except Exception:                                          # noqa: BLE001
            auc = float((P.argmax(1) == Y).mean())
        if auc > best:
            best = auc
            state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        if not quiet:
            print(f"    epoch {ep + 1:02d}/{epochs}  loss {tot / max(nb, 1):.4f}  "
                  f"val AUC {auc:.4f}  acc {(P.argmax(1) == Y).mean():.4f}")
    if state is not None:
        head.load_state_dict(state)
    return best


def eval_head_live(head, net, s2_dim, df):
    """Score the head's map against the masks, reading the backbone live.

    No gradient is needed anywhere: unlike improve.localize, which also produces Grad-CAM from
    the same pass, this only needs the head's own map.
    """
    from PIL import Image

    cen = center_map()
    head.eval()
    rows, ids, labels, fpred, rpred, areas = [], [], [], [], [], []
    miss, k = 0, 0
    loader = make_loader(df, "label", build_val_transform(), shuffle=False)
    # No autocast here, because improve.localize's evaluation uses none either. The reason
    # matters: the pointing score is the argmax over 1,600 positions, so half precision can
    # move it to a neighbouring cell, while the area-matched IoU, being a quantile threshold,
    # barely notices. Evaluating in half precision therefore reproduces the published IoU while
    # losing the published pointing score, which is exactly the discrepancy that found this.
    with torch.no_grad():
        for img, lbl in loader:
            f, s2, _ = feats(net, img.to(DEVICE))
            flat = net.forward_head(f)
            z, sal = head(net.forward_head(f, pre_logits=True).float(),
                          s2.float())
            M = up(sal.float()).cpu().numpy()
            fl = flat.float().argmax(1).cpu().numpy()
            for i in range(img.shape[0]):
                iid = df.iloc[k]["image_id"]
                k += 1
                mp = os.path.join(MASK_DIR, f"{iid}_segmentation.png")
                if not os.path.exists(mp):
                    miss += 1
                    continue
                mask = np.array(Image.open(mp).convert("L")
                                .resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)) > 127
                rows.append(metrics(np.stack([M[i], cen]), mask))
                ids.append(iid)
                labels.append(int(lbl[i]))
                fpred.append(int(fl[i]))
                rpred.append(int(z[i].argmax()))
                areas.append(float(mask.mean()))
    if miss:
        print(f"    [warning] {miss} masks not found under {MASK_DIR} and left out")
    return (np.stack(rows), np.array(ids), np.array(labels),
            np.array(fpred), np.array(rpred), np.array(areas))


def train_head_sup(head, net, s2_dim, tr, va, epochs, lr, lam, grid, quiet, select="att"):
    """Train the head with the lesion masks in the alignment loss.

    Supervising a position-wise score over the backbone's features is the ordinary way
    attention supervision is done, so the result is not tied to one bespoke module.

    The recipe is the MaskSet of improve.localize_sup, which applies one geometric
    augmentation to the image and the mask together, and a cross-entropy between the map and
    the mask after both are normalised to distributions. The epoch is chosen on the validation
    alignment loss by default, which is the rule the reported supervised rows use; passing
    select="auc" chooses it on the classification AUC instead. The grid is the backbone's own
    resolution rather than a pooled 40x40.
    """
    from sklearn.metrics import roc_auc_score
    from torch.utils.data import DataLoader

    from SkinCancer.configs.training_config import BATCH_SIZE
    from improve.localize_sup import MaskSet

    y = tr["label"].astype(int).values
    ce = nn.CrossEntropyLoss(weight=inv_freq(np.bincount(y, minlength=7)))
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=5e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
    tl = DataLoader(MaskSet(tr, True, grid=grid), batch_size=BATCH_SIZE, shuffle=True,
                    num_workers=4, pin_memory=True)
    vl = DataLoader(MaskSet(va, False, grid=grid), batch_size=BATCH_SIZE, shuffle=False,
                    num_workers=4, pin_memory=True)
    best, state = -1.0, None
    for ep in range(epochs):
        head.train()
        tc = ta = 0.0
        nb = 0
        for img, lbl, m in tl:
            img, lbl, m = img.to(DEVICE), lbl.to(DEVICE).long(), m.to(DEVICE)
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                f, s2, _ = feats(net, img)
                macro = net.forward_head(f, pre_logits=True)
                micro = s2
            with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                z, sal = head(macro.float(), micro.float())
                att = sal.reshape(sal.shape[0], -1).float()
                att = att / att.sum(1, keepdim=True).clamp_min(1e-12)
                tgt = m / m.sum(1, keepdim=True).clamp_min(1e-12)
                l_att = -(tgt * torch.log(att.clamp_min(1e-12))).sum(1).mean()
                l_cls = ce(z, lbl)
                loss = l_cls + lam * l_att
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tc += float(l_cls)
            ta += float(l_att)
            nb += 1
        head.eval()
        P, Y, va_att = [], [], []
        with torch.no_grad():
            for img, lbl, m in vl:
                m = m.to(DEVICE)
                with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                    f, s2, _ = feats(net, img.to(DEVICE))
                    z, sal = head(net.forward_head(f, pre_logits=True).float(),
                                  s2.float(), lbl.to(DEVICE).long())
                a = sal.reshape(sal.shape[0], -1).float()
                a = a / a.sum(1, keepdim=True).clamp_min(1e-12)
                t = m / m.sum(1, keepdim=True).clamp_min(1e-12)
                va_att.append(float(-(t * torch.log(a.clamp_min(1e-12))).sum(1).mean()))
                P.append(torch.softmax(z.float(), 1).cpu().numpy())
                Y.append(lbl.numpy())
        P, Y = np.concatenate(P), np.concatenate(Y).astype(int)
        try:
            auc = roc_auc_score(Y, P / P.sum(1, keepdims=True), multi_class="ovr",
                                average="macro")
        except Exception:                                          # noqa: BLE001
            auc = float((P.argmax(1) == Y).mean())
        att_val = float(np.mean(va_att))
        # The quantity this run reports is the map, so the epoch is chosen on the validation
        # alignment loss and not on the classification AUC. Selecting on the AUC is what
        # improve.localize_sup does, and for a module whose alignment loss settles within three
        # epochs it makes no difference; for a mechanism still converging at the end of the
        # budget it picks an early epoch whose map is nearly uninformative, which is a property
        # of the selection rule rather than of the mechanism.
        score = -att_val if select == "att" else auc
        if score > best:
            best = score
            state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        if not quiet:
            print(f"    epoch {ep + 1:02d}/{epochs}  cls {tc / max(nb, 1):.4f}  "
                  f"att {ta / max(nb, 1):.4f}  val att {att_val:.4f}  val AUC {auc:.4f}"
                  + ("  <-" if score == best else ""))
    if state is not None:
        head.load_state_dict(state)
    return best


def curve_within(net, img, base, maps, mask, cls, steps):
    """Insertion and deletion restricted to the positions inside the lesion.

    The standard test ranks every position in the image, so on a dataset whose object is large
    and centred it rewards a map for covering the object and says little about what the map
    knows beyond that. The centred Gaussian makes this concrete: it carries no information
    about the model at all and still reaches the highest insertion-minus-deletion of any map
    measured here, because deleting the centre of a centred lesion is enough to destroy the
    prediction.

    Restricting the ordering to the lesion removes that. Positions outside the mask are left
    untouched in both directions, and the fraction of the lesion that has been blurred or
    restored is what advances with each step, so the question becomes which parts of the lesion
    the map considers decisive rather than whether it found the lesion. A map that decreases
    monotonically from the image centre orders the inside of a lesion by distance from that
    centre, which is unrelated to where the evidence is, so this measure is near zero for it by
    construction rather than by measurement -- which is the property the standard test lacks.

    Images whose mask is empty or covers everything have no inside to rank and are left to the
    caller to drop.
    """
    B = img.shape[0]
    mf = mask.view(B, 1, IMG_SIZE, IMG_SIZE).float()
    inside = mask.reshape(B, -1) > 0.5
    flat = maps.reshape(B, -1).float().clone()
    flat[~inside] = -float("inf")                     # rank among in-lesion positions only
    order = flat.argsort(dim=1, descending=True)
    rank = torch.empty_like(order)
    ar = torch.arange(flat.shape[1], device=img.device)
    rank.scatter_(1, order, ar.expand(B, -1))
    n_in = inside.sum(1)
    out = {"del": [], "ins": []}
    for i in range(steps + 1):
        k = (n_in.float() * i / steps).long()[:, None]
        sel = ((rank < k) & inside).view(B, 1, IMG_SIZE, IMG_SIZE).float()
        for mode in ("del", "ins"):
            # Deletion blurs the selected part of the lesion and leaves the rest of the image
            # alone; insertion blurs the whole lesion and restores the selected part.
            x = (img * (1 - sel) + base * sel) if mode == "del" else \
                (img * (1 - mf) + base * (mf - sel) + img * sel)
            with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                feat = net.forward_intermediates(x, intermediates_only=False)[0]
                p = torch.softmax(net.forward_head(feat).float(), 1)
            out[mode].append(p.gather(1, cls[:, None]).squeeze(1).cpu())
    return {m: torch.stack(v, 1).numpy() for m, v in out.items()}


def faith_of(net, head, s2_dim, df, n_img, steps):
    """Insertion minus deletion, over the whole image and inside the lesion, with the
    centred Gaussian beside both.

    Overlap says where a map points; it does not say whether the model used what it points at.
    The whole-image curves come from improve.faithful: its `curve` takes the map as an argument
    and is indifferent to what produced it, so this measures the replacements with the published
    implementation rather than a second one, down to the blur kernel and the eleven steps. The
    within-lesion curves are computed here, by curve_within, and are the part of this study that
    the published protocol does not contain.

    The centred Gaussian is measured in the same pass for two reasons. On the whole-image
    measure its value is published (0.173), so reproducing it shows the path is right. On the
    within-lesion measure it is the reference that must come out near zero, since it knows
    nothing about the model; a run in which it does not is a run whose within-lesion numbers
    mean nothing.
    """
    from PIL import Image

    from improve.faithful import blur, curve

    cen = torch.tensor(center_map(), dtype=torch.float32, device=DEVICE)
    sub = df.head(n_img).reset_index(drop=True)
    keys = ("del", "ins", "wdel", "wins")
    acc = {k: {j: [] for j in keys} for k in ("head", "center")}
    head.eval()
    seen, used = 0, 0
    with torch.no_grad():
        for img, _ in make_loader(sub, "label", build_val_transform(), shuffle=False):
            x = img.to(DEVICE)
            B = x.shape[0]
            mk, ok = [], []
            for j in range(B):
                iid = sub.iloc[seen + j]["image_id"]
                mp = os.path.join(MASK_DIR, f"{iid}_segmentation.png")
                a = (np.array(Image.open(mp).convert("L")
                              .resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)) > 127
                     if os.path.exists(mp) else np.zeros((IMG_SIZE, IMG_SIZE), bool))
                mk.append(a)
                ok.append(bool(0.0 < a.mean() < 1.0))
            seen += B
            mask = torch.tensor(np.stack(mk), dtype=torch.float32, device=DEVICE)
            keep = np.array(ok)
            f, s2, _ = feats(net, x)
            cls = net.forward_head(f).argmax(1)
            _, sal = head(net.forward_head(f, pre_logits=True).float(),
                          s2.float(), cls)
            base = blur(x)
            M = {"head": up(sal.float()).detach(), "center": cen.expand(B, -1, -1)}
            for nm in acc:
                for mode in ("del", "ins"):
                    acc[nm][mode].append(curve(net, x, base, M[nm], cls, steps, mode))
                if keep.any():
                    w = curve_within(net, x, base, M[nm], mask, cls, steps)
                    acc[nm]["wdel"].append(w["del"][keep])
                    acc[nm]["wins"].append(w["ins"][keep])
            used += int(keep.sum())
    out = {}
    for nm in acc:
        v = {j: np.concatenate(acc[nm][j]).mean(1).mean() if acc[nm][j] else float("nan")
             for j in keys}
        out[nm] = (float(v["del"]), float(v["ins"]), float(v["ins"] - v["del"]),
                   float(v["wdel"]), float(v["wins"]), float(v["wins"] - v["wdel"]))
    return out, len(sub), used


def check_splits(tr, va, ev, limit):
    """Refuse to run on a split that lost rows because images could not be found.

    The split readers drop any row whose image path did not resolve, so a data root that is
    wrong by one directory level yields a smaller split rather than an error. Nothing
    downstream notices: the cache is built, the heads train, and the overlap comes out of
    however many images happened to resolve. An empty split at least fails, eventually and
    obscurely; a half-resolved one produces a plausible number, which is worse. So the row
    counts are compared against the split files themselves before a backbone is loaded.
    """
    import pandas as pd

    from SkinCancer.configs.data_config import SPLIT_ROOT

    if min(len(tr), len(va), len(ev)) == 0:
        raise SystemExit(
            f"a split came out empty (train {len(tr)}, val {len(va)}, eval {len(ev)}). The "
            f"images did not resolve: check HAM_DATA_ROOT, which must be the directory holding "
            f"HAM10000_metadata.csv and the two image directories. env.sh sets it to its own "
            f"default, so an override has to be exported after env.sh is sourced, not before.")
    if limit:
        return
    want = 0
    for s in ("train", "val"):
        p = os.path.join(SPLIT_ROOT, f"global_{s}.csv")
        if not os.path.exists(p):
            print(f"  [note] {p} is absent, so the split size could not be verified")
            return
        want += len(pd.read_csv(p))
    got = len(tr) + len(va) + (0 if ev is va or len(ev) == len(va) else len(ev))
    if got < want:
        raise SystemExit(
            f"the cross-validation split holds {got} images but the split files list {want}: "
            f"{want - got} images did not resolve and were dropped silently. Fix HAM_DATA_ROOT "
            f"before running, because the measurements would otherwise be taken over a subset "
            f"without saying so.")
    print(f"  split verified: {got} images, matching the split files")


def run_fold(args, head_names):
    set_seed(args.seed)
    pm = build_image_path_map()
    tr, va, _ = get_data(args.fold, pm)
    ev = eval_split(args.fold, pm)
    if args.limit:
        tr = tr.groupby("label").head(max(2, args.limit // 7)).reset_index(drop=True)
        va = va.head(args.limit).reset_index(drop=True)
        ev = ev.head(args.limit).reset_index(drop=True)
    check_splits(tr, va, ev, args.limit)
    net = load_flat(args.fold)
    s2_dim, final_dim, hw = probe(net)
    print(f"[head_swap] fold {args.fold} seed {args.seed}  train={len(tr)} val={len(va)} "
          f"eval={len(ev)}  stage-2 map {s2_dim}x{hw[0]}x{hw[1]}  macro {final_dim}")
    print(f"[head_swap] produced by head_swap.py {source_id()}")


    # Without HAM_INNER_VAL the epoch is chosen on the same held-out fold that is then measured,
    # which is what improve.localize does and therefore the configuration the published overlap
    # comes from. When the two splits are the same images there is no reason to encode them
    # twice, and with the variable set they genuinely differ and both caches are built.
    cache = None
    if not args.live and not args.measure_only:
        splits = {"train": tr, "val": va}
        same = len(va) == len(ev) and np.array_equal(va["image_id"].values,
                                                    ev["image_id"].values)
        if same:
            print("  val and eval are the same held-out fold; one cache serves both")
        else:
            splits["eval"] = ev
        cache = cache_fold(net, args.fold, s2_dim, final_dim, hw, splits, args.rebuild_cache,
                           suffix=f"_lim{args.limit}" if args.limit else "")
        if same:
            cache["eval"] = cache["val"]
    y_tr = tr["label"].astype(int).values
    y_va = va["label"].astype(int).values

    os.makedirs(OUT, exist_ok=True)
    summary = []
    for name in head_names:
        t0 = time.time()
        head = build_head(name, net, s2_dim, final_dim, 7, args.rank, args.signed, args.map)
        total, mech = counts(head)
        stem = name + (f"_sup{args.supervise:g}" if args.supervise else "")
        print(f"  [{name}] total {total:,} parameters, saliency mechanism {mech:,}"
              + (f", mask supervision lambda={args.supervise:g}" if args.supervise else ""))
        if args.measure_only:
            # Measure a head that was already trained, so that a measurement added after the
            # fact does not change the numbers already reported. Retraining would: the readout
            # is not deterministic, and its overlap moves by more than the quantity being added.
            # This is what the saved checkpoints are for.
            ck = os.path.join(OUT, f"head_{stem}_f{args.fold}.pth")
            if not os.path.exists(ck):
                raise SystemExit(f"--measure-only needs a trained head at {ck}")
            head.load_state_dict(torch.load(ck, map_location=DEVICE, weights_only=True),
                                 strict=True)
            print(f"  [{name}] loaded {os.path.basename(ck)}; not training")
            acc = float("nan")
            R, ids, labels, fpred, rpred, areas = eval_head_live(head, net, s2_dim, ev)
        elif args.supervise:
            if hw[0] != hw[1]:
                raise SystemExit(f"the mask grid assumes a square map, got {hw}")
            acc = train_head_sup(head, net, s2_dim, tr, va, args.epochs, args.lr,
                                 args.supervise, hw[0], args.quiet, args.select)
            R, ids, labels, fpred, rpred, areas = eval_head_live(head, net, s2_dim, ev)
        elif args.live:
            acc = train_head_live(head, net, s2_dim, tr, va, args.epochs, args.lr, args.quiet)
            R, ids, labels, fpred, rpred, areas = eval_head_live(head, net, s2_dim, ev)
        else:
            acc = train_head(head, cache["train"], cache["val"], y_tr, y_va,
                             args.epochs, args.batch, args.lr, args.seed, args.quiet)
            R, ids, labels, fpred, rpred, areas = eval_head(head, cache["eval"], ev, name,
                                                            args.batch)
        # The trained head is saved before anything else that could fail, because training is
        # the only expensive part: a later step raising after it has run would otherwise throw
        # away the minutes it cost. That is exactly how this rerun became necessary. A
        # measure-only run has nothing new to save and must not overwrite its own input.
        if not args.measure_only:
            torch.save(head.state_dict(), os.path.join(OUT, f"head_{stem}_f{args.fold}.pth"))
        fa = None
        if args.faith_n and (args.live or args.supervise or args.measure_only):
            fa, n_fa, n_w = faith_of(net, head, s2_dim, ev, args.faith_n, args.faith_steps)
            print(f"  [{name}] Ins-Del {fa['head'][2]:+.4f} over {n_fa} images "
                  f"(centred Gaussian {fa['center'][2]:+.4f})")
            print(f"  [{name}] within-lesion Ins-Del {fa['head'][5]:+.4f} over {n_w} images "
                  f"(centred Gaussian {fa['center'][5]:+.4f})")
        # A reduced run is a smoke test, not a result, and must not be left where --summary
        # would read it as one.
        tag = (f"{stem}"
               + (f"_map{args.map}" if args.measure_only
                  and getattr(args, "map_explicit", False) else "")
               + f"_f{args.fold}"
               + ("" if args.live or args.supervise or args.measure_only else "_cached")
               + (f"_lim{args.limit}" if args.limit else ""))
        path = os.path.join(OUT, f"hs_{tag}.npz")
        np.savez(path, metrics=R, names=np.array([name, "center"]), image_id=ids, label=labels,
                 flat_pred=fpred, readout_pred=rpred, lesion_area=areas,
                 params_total=total, params_mechanism=mech, readout_val_acc=acc,
                 source_sha1=source_id(), select=args.select if args.supervise else "",
                 **({} if fa is None else
                    {"faith_head": np.array(fa["head"]), "faith_center": np.array(fa["center"]),
                     "faith_n": args.faith_n, "faith_n_within": n_w}))
        mine = [float(np.nanmean(R[:, 0, c])) for c in range(3)]
        ref = [float(np.nanmean(R[:, 1, c])) for c in range(3)]
        summary.append((name, total, mech, mine, ref, time.time() - t0, len(ids), fa))
        print(f"  [{name}] n={len(ids)}  pointing {mine[0]:.3f}  energy {mine[1]:.3f}  "
              f"IoU {mine[2]:.3f}   ({time.time() - t0:.0f}s)  -> {path}")
        del head
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    # The cache is a pure intermediate: every result is already in the npz files, and rebuilding
    # it costs one backbone pass. At 0.85 MiB an image a fold's cache runs to several GiB, so
    # keeping all five would cost tens of GiB to save a pass that takes minutes. It is dropped
    # only once every requested head has finished, so a failure part way through leaves the
    # cache in place and the retry is cheap.
    if args.drop_cache and cache is not None and len(summary) == len(head_names):
        cache.clear()                      # release the memmaps before unlinking them
        freed = drop_cache(args.fold, suffix=f"_lim{args.limit}" if args.limit else "")
        print(f"  cache for fold {args.fold} dropped, {freed / 2 ** 30:.1f} GiB freed; "
              f"the hs_*.npz results do not need it")

    if not summary:
        return
    print()
    print(f"    fold {args.fold}, {summary[0][6]} images, the centred Gaussian measured in the "
          f"same run as the reference")
    print(f"    {'head':<10}{'mechanism':>12}{'Pointing':>11}{'Energy':>9}{'IoU':>9}"
          f"{'Ins-Del':>10}{'within':>10}{'sec':>7}")
    for name, _total, mech, mine, _ref, sec, _n, fa in summary:
        fd = (f"{fa['head'][2]:>+10.4f}{fa['head'][5]:>+10.4f}" if fa
              else f"{'---':>10}{'---':>10}")
        print(f"    {name:<10}{mech:>12,}{mine[0]:>11.4f}{mine[1]:>9.4f}{mine[2]:>9.4f}"
              f"{fd}{sec:>7.0f}")
    gauss = summary[0][4]
    gf = summary[0][7]
    gd = (f"{gf['center'][2]:>+10.4f}{gf['center'][5]:>+10.4f}" if gf
          else f"{'---':>10}{'---':>10}")
    print(f"    {'center':<10}{0:>12,}{gauss[0]:>11.4f}{gauss[1]:>9.4f}{gauss[2]:>9.4f}{gd}")
    print()
    print("    `within` is insertion minus deletion with the ordering restricted to the lesion,")
    print("    so a map that merely covers the lesion scores nothing there and the centred")
    print("    Gaussian is near zero by construction. It separates finding the lesion from")
    print("    knowing which part of it the classifier used.")
    # Every head scores the same fixed map on the same images, so the reference must come out
    # identical whichever row produced it; a disagreement would mean the image sets differ.
    for name, _t, _m, _mine, ref, _s, _n, _f in summary[1:]:
        if max(abs(a - b) for a, b in zip(ref, gauss)) > 1e-9:
            print(f"    [warning] the {name} run scored a different centred Gaussian "
                  f"({ref}); its image set is not the same")
    print()
    print("    These are one fold of the cross-validation split. Nothing here touches the test")
    print("    split, which is the protocol the paper states for designs that were explored")
    print("    rather than pre-specified.")


def summarize(args):
    """The table across folds, averaged the way the paper averages.

    A fold's images are averaged first and the fold means are then averaged, which is the
    convention every overlap number in the paper follows; pooling the images instead moves a
    value in the fourth decimal. The spread reported is across folds, so it says how much the
    ordering between the heads depends on which fold is held out -- which is the only thing that
    makes a difference between two heads worth reading.
    """
    import glob
    import re

    # The rows are discovered rather than listed, so a supervised variant appears without this
    # function having to know the weight it was trained with. A reduced run is a smoke test and
    # a cached one was trained without the random transform, so neither belongs in a table
    # beside the others and both are excluded by name.
    stems = []
    for f in glob.glob(os.path.join(OUT, "hs_*_f*.npz")):
        b = os.path.basename(f)
        if any(x in b for x in ("_lim", "_cached")):
            continue
        mt = re.fullmatch(r"hs_(.+)_f\d+\.npz", b)
        if mt and mt.group(1) not in stems:
            stems.append(mt.group(1))
    stems.sort(key=lambda n: (HEADS.index(n.split("_")[0]) if n.split("_")[0] in HEADS else 9,
                              n))
    if not stems:
        print("    nothing run yet")
        return
    w = max(10, max(len(n) for n in stems) + 1)
    print(f"    {'head':<{w}}{'folds':>7}{'mechanism':>12}{'Pointing':>19}{'Energy':>17}"
          f"{'IoU':>17}{'Ins-Del':>16}{'within':>16}")
    ref_rows, gauss_faith, gauss_within = [], [], []
    for name in stems:
        # Only the live runs: a reduced run is a smoke test, and a cached one was trained
        # without the random transform, so neither belongs in a table beside the other.
        # A full match, not the glob `hs_{name}_f*.npz`: that also matches a longer stem whose
        # continuation begins with `f`, which `hs_s3_mlp_frac...` does, so the label-efficiency
        # archives were averaged into the unsupervised row. Admitting nothing between the stem
        # and the fold number also excludes `_lim` and `_cached` by construction.
        want = re.compile(r"^hs_" + re.escape(name) + r"_f\d+\.npz$")
        files = sorted(p for p in glob.glob(os.path.join(OUT, "hs_*.npz"))
                       if want.fullmatch(os.path.basename(p)))
        if not files:
            continue
        per_fold, mech, ref, fd, fc, wd, wc = [], None, [], [], [], [], []
        srcs = set()
        for p in files:
            with np.load(p, allow_pickle=True) as z:
                R = z["metrics"]
                per_fold.append([np.nanmean(R[:, 0, c]) for c in range(3)])
                ref.append([np.nanmean(R[:, 1, c]) for c in range(3)])
                mech = int(z["params_mechanism"]) if "params_mechanism" in z.files else None
                srcs.add(str(z["source_sha1"]) if "source_sha1" in z.files else "unstamped")
                if "faith_head" in z.files:
                    fd.append(float(z["faith_head"][2]))
                    fc.append(float(z["faith_center"][2]))
                    if len(z["faith_head"]) > 5:
                        wd.append(float(z["faith_head"][5]))
                        wc.append(float(z["faith_center"][5]))
        A = np.array(per_fold)
        m, s = A.mean(0), A.std(0, ddof=1) if len(A) > 1 else np.zeros(3)
        ref_rows.append(np.array(ref).mean(0))
        cells = "".join(f"{m[c]:>13.4f} ±{s[c]:<5.4f}" for c in range(3))
        ins = (f"{np.mean(fd):>+9.4f} ±{np.std(fd, ddof=1) if len(fd) > 1 else 0:<5.4f}"
               if fd else f"{'---':>9} {'':<6}")
        wit = (f"{np.mean(wd):>+9.4f} \u00b1{np.std(wd, ddof=1) if len(wd) > 1 else 0:<5.4f}"
               if wd else f"{'---':>9} {'':<6}")
        print(f"    {name:<{w}}{len(files):>7}"
              f"{mech if mech is not None else 0:>12,}{cells}{ins}{wit}")
        if fc:
            gauss_faith.append(float(np.mean(fc)))
        if wc:
            gauss_within.append(float(np.mean(wc)))
        if len(srcs) > 1:
            print(f"    {'':<{w}}  [note] this row was written by more than one version of "
                  f"head_swap.py: {', '.join(sorted(srcs))}")
    if ref_rows:
        g = np.mean(ref_rows, axis=0)
        cells = "".join(f"{g[c]:>13.4f} {'':<6}" for c in range(3))
        gf = (f"{np.mean(gauss_faith):>+9.4f} {'':<6}" if gauss_faith
              else f"{'---':>9} {'':<6}")
        gw = (f"{np.mean(gauss_within):>+9.4f} {'':<6}" if gauss_within
              else f"{'---':>9} {'':<6}")
        print(f"    {'center':<{w}}{'':>7}{0:>12,}{cells}{gf}{gw}")
    print()
    print("    Each fold's images are averaged first and the fold means then averaged, as every")
    print("    overlap number in the paper is. The spread is across folds. These are")
    print("    cross-validation folds; no row here was measured on the test split.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", action="store_true", help="parameter counts only; no GPU, no data")
    ap.add_argument("--summary", action="store_true", help="the table across whatever folds ran")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--head", default="all", choices=HEADS + ["all"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rank", type=int, default=4, help="rank of the lowrank head's score")
    ap.add_argument("--epochs", type=int, default=10, help="as improve.localize uses")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4, help="as improve.localize uses")
    ap.add_argument("--limit", type=int, default=0, help="smoke test on a few images per class")
    ap.add_argument("--signed", action="store_true",
                    help="keep the ssf map signed instead of rectifying it")
    ap.add_argument("--supervise", type=float, default=0.0, metavar="LAM",
                    help="weight on the mask-alignment loss, as improve.localize_sup's --lam; "
                         "0 trains without masks")
    ap.add_argument("--map", default=None, choices=["relu", "softmax"],
                    help="what the replacement heads report as their map; defaults to softmax "
                         "when supervising, since that is the distribution the loss acts on, "
                         "and to relu otherwise")
    ap.add_argument("--measure-only", action="store_true",
                    help="load the head saved by an earlier run and only measure it; adds a "
                         "measurement to a finished row without moving the numbers in it")
    ap.add_argument("--select", default="att", choices=["att", "auc"],
                    help="what the epoch is chosen on in a supervised run: the validation "
                         "alignment loss (default, and the rule the reported supervised rows "
                         "use, because the map is what is measured) or the classification AUC")
    ap.add_argument("--faith-n", type=int, default=400,
                    help="images per fold for insertion/deletion, as improve.faithful uses; "
                         "0 skips it")
    ap.add_argument("--faith-steps", type=int, default=10)
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--drop-cache", action="store_true",
                    help="delete this fold's cache once every head has finished; the npz "
                         "results do not need it, but keeping it makes a re-train free")
    ap.add_argument("--cached", dest="live", action="store_false", default=True,
                    help="train on cached backbone output: faster, but without the random "
                         "training transform, which degenerates the attention map and is why "
                         "the live path is the default")
    ap.add_argument("--quiet", action="store_true", help="suppress the per-epoch lines")
    args = ap.parse_args()
    # Remember whether the mode was asked for, not just what it ended up as. A measure-only run
    # that re-reads a trained head under a different map mode must not write over the archive
    # the training run produced: that file is the measurement the tables were built from, and
    # overwriting it would be silent. The tag carries the mode only when the mode was chosen.
    args.map_explicit = args.map is not None
    if args.map is None:
        args.map = "softmax" if args.supervise else "relu"

    if args.count:
        count_only(args)
        return
    if args.summary:
        summarize(args)
        return
    run_fold(args, HEADS if args.head == "all" else [args.head])


if __name__ == "__main__":
    main()
