"""Count what each configuration costs, for the parts that do not depend on the hardware.

    python check_model_cost.py
    python check_model_cost.py --time # also measure latency and memory
    python check_model_cost.py --macs # also count multiply-accumulates
"""

import argparse
import os
import sys


def millions(model):
    """Saved tensors in millions: the parameters and the buffers."""
    par = sum(p.numel() for p in model.parameters())
    buf = sum(b.numel() for b in model.buffers())
    return par / 1e6, buf / 1e6


def build_configurations(device):
    """Each configuration as (name, factory). The factory is called inside a try block."""
    import torch

    from improve.residual import ResidualModel

    def single():
        m = ResidualModel("flat", pretrained=False).to(device)
        return [m], (lambda x: m(x))

    def with_replacement(supervised):
        """The head the paper proposes, as improve.deep_head builds it."""
        from improve.deep_head import StageView
        from improve.head_swap import build_head, probe
        from improve.localize import feats

        flat = ResidualModel("flat", pretrained=False).to(device)
        net = StageView(flat.net, 3)
        s_dim, final_dim, _hw = probe(net)
        head = build_head("mlp", net, s_dim, final_dim, 7, 4, False,
                          "softmax" if supervised else "relu").to(device)

        def step(x):
            """One inference: the backbone, then the head on the features it produced."""
            f, s2, _ = feats(net, x)
            macro = net.forward_head(f, pre_logits=True).float()
            return head(macro, s2.float())

        return [flat, head], step

    # the third field is how many of the parts consume the image, so that the
    return [("Single-stage", single, 1),
            ("+ explanation head (unsup.)", lambda: with_replacement(False), 0),
            ("+ explanation head (mask sup.)", lambda: with_replacement(True), 0)]


def macs_of(parts, device, size):
    """Multiply-accumulates of every part that consumes the image, added together."""
    import torch
    x = torch.randn(1, 3, size, size, device=device)
    out = []
    for part in parts:
        try:
            from thop import profile
            m, _ = profile(part, inputs=(x,), verbose=False)
            out.append(m / 1e9)
            continue
        except ImportError:
            pass
        try:
            from fvcore.nn import FlopCountAnalysis
            out.append(float(FlopCountAnalysis(part, x).total()) / 1e9)
        except ImportError:
            return None
    return out


def timing(parts, step, device, size, batch, repeats=30):
    """Median forward time per image, after a warm-up, and the peak activation memory."""
    import torch
    x = torch.randn(batch, 3, size, size, device=device)
    for p in parts:
        p.eval()
    with torch.no_grad():
        for _ in range(5):
            step(x)
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(repeats):
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            step(x)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end) / batch)
        peak = torch.cuda.max_memory_allocated() - base
    times.sort()
    return times[len(times) // 2], peak / 2 ** 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time", action="store_true", help="measure latency and memory too")
    ap.add_argument("--macs", action="store_true", help="count multiply-accumulates too")
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 16])
    args = ap.parse_args()

    try:
        import torch
    except Exception as exc:                                   # noqa: BLE001
        sys.exit(f"torch is not importable here: {type(exc).__name__}: {exc}")
    from SkinCancer.configs.model_config import IMG_SIZE

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {device}, input {IMG_SIZE}x{IMG_SIZE}, parameters in millions, "
          f"weights as parameters x 4 bytes")
    if args.time and device != "cuda":
        print("  [note] --time needs a CUDA device; latency and memory are skipped")
        args.time = False
    print()
    head = f"    {'configuration':<32}{'Saved':>9}{'Params':>9}{'Weights':>10}"
    if args.macs:
        head += f"{'GMACs':>9}"
    if args.time:
        head += "".join(f"{f'ms b{b}':>10}" for b in args.batch) + f"{'Act MiB':>10}"
    print(head)

    for name, factory, n_image_parts in build_configurations(device):
        try:
            parts, step = factory()
        except Exception as exc:                               # noqa: BLE001
            print(f"    {name:<32}  could not be built: {type(exc).__name__}: {exc}")
            continue
        counts = [millions(m) for m in parts]
        par = sum(c[0] for c in counts)
        buf = sum(c[1] for c in counts)
        p = par + buf
        row = f"    {name:<32}{p:>9.2f}{par:>9.2f}{p * 1e6 * 4 / 2 ** 20:>10.0f}"
        if args.macs:
            each = macs_of(parts[:n_image_parts], device, IMG_SIZE) if n_image_parts else None
            if each is None:
                row += f"{'---':>9}"
            else:
                row += f"{sum(each):>9.2f}"
                if len(each) > 1:
                    row += "   (" + " + ".join(f"{g:.2f}" for g in each) + ")"
        if args.time:
            for b in args.batch:
                try:
                    ms, peak = timing(parts, step, device, IMG_SIZE, b)
                    row += f"{ms:>10.2f}"
                except Exception as exc:                       # noqa: BLE001
                    row += f"{'err':>10}"
                    peak = float("nan")
            row += f"{peak:>10.0f}"
        print(row)
        del parts
        if device == "cuda":
            torch.cuda.empty_cache()

    print()
    print("    Saved is the parameters plus the buffers, which is what a checkpoint holds and")
    print("    what the paper's table counts; Params leaves the buffers out. The weight column")
    print("    is Saved times four bytes, against which the paper reports what the allocator")
    print("    reserved, a little larger.")

    sys.exit(0)


if __name__ == "__main__":
    main()
