#!/usr/bin/env python3
"""Draw the quantitative figures from the manuscript's own tables.

    python make_figs.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figdata  # noqa: E402

# Figures are written beside the manuscript. Point HAM_PAPER_SRC at the directory holding
# main.tex and supplementary.tex, or run this from that directory.
OUT = os.environ.get("HAM_PAPER_SRC") or os.path.dirname(os.path.abspath(__file__))

BAND = "#f1f1f1"
GREY = "#cccccc"
INK = "#1a1a1a"
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

# the order the rows are drawn in, top to bottom, and the label each gets
ORDER = [
    ("Random map", "Random map"),
    ("Centered Gaussian", "Centered Gaussian"),
    ("Grad-CAM (stage-2)", "Grad-CAM (stage-2)"),
    ("Grad-CAM (stage-3)", "Grad-CAM (stage-3)"),
    ("Integrated Gradients", "Integrated Gradients"),
    ("Explanation head", "Explanation head"),
    ("no global path", "    no global-feature path"),
    ("+ mask supervision", "    + mask supervision"),
]
PANELS = [("iou", "(a)  Overlap", "area-matched IoU"),
          ("insdel", "(b)  Whole image", r"INS $-$ DEL"),
          ("within", "(c)  Inside the lesion", r"INS $-$ DEL")]


def bands(ax, n):
    for i in range(n):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color=BAND, zorder=0, lw=0)


def quant():
    rows = dict(figdata.table3())
    missing = [k for k, _ in ORDER if k not in rows]
    if missing:
        sys.exit(f"make_figs: Table 1 has no rows {missing}")
    n = len(ORDER)
    ys = np.arange(n)[::-1]

    fig, axes = plt.subplots(1, 3, figsize=(5.2, 1.95), dpi=450)
    for ax, (key, title, xlab) in zip(axes, PANELS):
        bands(ax, n)
        mu = np.array([rows[k][key][0] for k, _ in ORDER])
        sd = np.array([rows[k][key][1] for k, _ in ORDER])
        ref = rows["Centered Gaussian"][key][0]
        ax.axvline(ref, color="#888888", ls="--", lw=0.7, zorder=1)
        if key != "iou":
            ax.axvline(0, color=INK, lw=0.5, zorder=1)
        ax.errorbar(mu, ys, xerr=sd, fmt="o", ms=2.6, color=INK,
                    ecolor=GREY, elinewidth=0.9, capsize=0, zorder=3)
        ax.set_yticks(ys)
        ax.set_yticklabels([lab for _, lab in ORDER], fontsize=5.0)
        ax.set_ylim(-0.6, n - 0.4)
        if key == "iou":
            ax.set_xlim(0, 0.95)
        ax.set_title(title, fontsize=5.6, fontweight="bold", loc="left", pad=3)
        ax.set_xlabel(xlab, fontsize=5.2)
        ax.tick_params(axis="x", labelsize=4.4, pad=1.5, length=2)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color("#555555")
    for ax in axes[1:]:
        ax.set_yticklabels([])
    # the dashed reference is explained in the caption, not written over the data
    fig.tight_layout(w_pad=0.5, pad=0.4)
    p = os.path.join(OUT, "fig_quant.png")
    fig.savefig(p)
    plt.close(fig)
    print(f"  wrote {os.path.basename(p)}")
    for k, _ in ORDER:
        print(f"      {k:<22} IoU {rows[k]['iou'][0]:.3f}  whole {rows[k]['insdel'][0]:+.3f}"
              f"  within {rows[k]['within'][0]:+.3f}")


def shuffle():
    rows = figdata.shuffle_control()
    n = len(rows)
    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.3), dpi=450)

    for ax, (key, title, ylab) in zip(axes, [
            ("iou", "(a)  Overlap is unchanged", "area-matched IoU"),
            ("within", "(b)  Within-lesion scores approach zero",
             r"INS $-$ DEL, lesion interior")]):
        for i, (lab, iou, w, ws) in enumerate(rows):
            if key == "iou":
                y0 = y1 = iou[0]
                e0 = e1 = iou[1]
            else:
                (y0, e0), (y1, e1) = w, ws
            ax.plot([0, 1], [y0, y1], "-", color=GREY, lw=0.9, zorder=2)
            ax.errorbar([0, 1], [y0, y1], yerr=[e0, e1], fmt="o", ms=2.4, color=INK,
                        ecolor="#9e9e9e", elinewidth=0.8, capsize=0, zorder=3)
            ax.annotate(lab, xy=(-0.06, y0), xycoords=("data", "data"),
                        ha="right", va="center", fontsize=4.8, color=INK)
        if key == "within":
            ax.axhline(0, color=INK, lw=0.5, zorder=1)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["as measured", "permuted inside\nthe lesion"], fontsize=4.8)
        ax.set_xlim(-1.15, 1.12)
        ax.set_title(title, fontsize=5.6, fontweight="bold", loc="left", pad=3)
        ax.set_ylabel(ylab, fontsize=5.2)
        ax.tick_params(axis="y", labelsize=4.4, pad=1.5, length=2)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("bottom", "left"):
            ax.spines[sp].set_color("#555555")
    fig.tight_layout(w_pad=1.0, pad=0.4)
    p = os.path.join(OUT, "fig_shuffle.png")
    fig.savefig(p)
    plt.close(fig)
    print(f"  wrote {os.path.basename(p)}")
    for lab, iou, w, ws in rows:
        print(f"      {lab:<22} IoU {iou[0]:.3f} (unchanged)   "
              f"within {w[0]:+.3f} -> {ws[0]:+.3f}")


if __name__ == "__main__":
    quant()
    shuffle()
