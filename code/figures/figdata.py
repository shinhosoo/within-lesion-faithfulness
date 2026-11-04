#!/usr/bin/env python3
"""Read the manuscript's tables so the figures are drawn from them, not retyped beside them."""
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# Point HAM_PAPER_SRC at the directory holding main.tex and supplementary.tex.
SRC = os.environ.get("HAM_PAPER_SRC") or HERE
MAIN = os.path.join(SRC, "main.tex")
SUPP = os.path.join(SRC, "supplementary.tex")


def _tabular(tex, label):
    i = tex.find("\\label{%s}" % label)
    if i < 0:
        sys.exit(f"figdata: no \\label{{{label}}}")
    b = tex.find("\\begin{tabular}", i)
    e = tex.find("\\end{tabular}", b)
    if not (0 < b < e):
        sys.exit(f"figdata: no tabular for {label}")
    return tex[b:e]


def _rows(body):
    out = []
    for line in body.splitlines():
        if "&" not in line or line.lstrip().startswith("%"):
            continue
        if "\\multicolumn" in line or "\\cmidrule" in line:
            continue
        cells = [c.strip() for c in re.sub(r"\\\\\s*$", "", line.strip()).split("&")]
        lab = cells[0]
        lab = re.sub(r"\\tnote\{[a-z]\}", "", lab)
        lab = re.sub(r"\\(textbf|emph|textit|boldmath|quad)\b", " ", lab)
        lab = re.sub(r"\$\s*\+\s*\$", "+", lab)
        lab = re.sub(r"[{}$\\]", " ", lab)
        lab = re.sub(r"\s+", " ", lab).strip()
        out.append((lab, cells[1:]))
    return out


def _val(cell):
    """(mean, sd or None) from a table cell, or None when it holds no number."""
    c = re.sub(r"\\(textbf|boldmath)\b|[{}]", " ", cell)
    m = re.search(r"([-+]?\d*\.\d+)\s*(?:\\pm|\$\\pm\$)\s*([-+]?\d*\.\d+)", c)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = re.search(r"([-+]?\d*\.\d+)", c)
    return (float(m.group(1)), None) if m else None


def table3():
    """Main Table 1 -> [(label, {measure: (mean, sd)})] in the table's own row order."""
    body = _tabular(io.open(MAIN, encoding="utf-8").read(), "tab:explanation_quality")
    cols = ["pointing", "energy", "iou", "insdel", "within"]
    out = []
    for lab, cells in _rows(body):
        if lab.lower().startswith("saliency map"):
            continue
        vals = [_val(c) for c in cells[:len(cols)]]
        if len(vals) != len(cols) or any(v is None for v in vals):
            continue
        out.append((lab, dict(zip(cols, vals))))
    if len(out) != 8:
        sys.exit(f"figdata: Table 1 gave {len(out)} data rows, expected 8")
    return out


def shuffle_control():
    """The shuffle control -> [(label, iou, within, within_shuffled)], each (mean, sd)."""
    body = _tabular(io.open(SUPP, encoding="utf-8").read(), "tab:shuffle_control")
    out = []
    for lab, cells in _rows(body):
        if lab.lower().startswith("map"):
            continue
        v = [_val(c) for c in cells[:3]]
        if len(v) != 3 or any(x is None for x in v):
            continue
        out.append((lab, v[0], v[1], v[2]))
    if len(out) != 5:
        sys.exit(f"figdata: the shuffle control gave {len(out)} data rows, expected 5")
    return out


if __name__ == "__main__":
    print("=== main Table 1 ===")
    for lab, d in table3():
        print(f"  {lab:<24} " + "  ".join(
            f"{k}={d[k][0]:+.3f}" + (f"±{d[k][1]:.3f}" if d[k][1] is not None else "       ")
            for k in ("iou", "insdel", "within")))
    print("\n=== the shuffle control ===")
    for lab, iou, w, ws in shuffle_control():
        print(f"  {lab:<24} IoU {iou[0]:.3f}±{iou[1]:.3f}   "
              f"within {w[0]:+.3f}±{w[1]:.3f} -> {ws[0]:+.3f}±{ws[1]:.3f}")
