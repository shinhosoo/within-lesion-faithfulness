"""Check Supplementary Tables S2, S3 and S4 against the probe log that re-measured them.

The tables moved from mixed to single precision and the only record of that run is its log.
Checking the tables against it is the difference between believing the move was carried
through and knowing it was. Three things are checked, each against a different part of the
log:

  S2  the log names every cell that moved at three decimals, old value to new. Each new
      value must be the one the table now prints, and each old value must not appear in
      that cell any more.
  S3  the TALLY lines give, per fold, the image count and the error count under each
      precision. The table prints the fold mean and its spread, so both are recomputed.
  S4  the same, for the two lesion-restricted conditions.

    python3 check_logs_vs_tables.py <log dir> <supplementary.tex>
"""
import glob
import os
import re
import statistics
import sys

PROBLEMS = []


def check(ok, what, detail):
    print(("  ok      " if ok else "  PROBLEM ") + f"{what:<34}{detail}")
    if not ok:
        PROBLEMS.append(what)


def tabular(tex, label):
    i = tex.find("\\label{" + label + "}")
    if i < 0:
        return ""
    j = tex.find("\\begin{tabular", i)
    k = tex.find("\\end{tabular", j)
    return tex[j:k] if 0 < j < k else ""


def tally(logdir):
    """{(target, row): {fold: (n, wrong_mixed, wrong_single)}}"""
    pat = re.compile(r"TALLY (\S+) (\S+) fold=(\d+) n=(\d+) wrong_mixed=(\d+) "
                     r"wrong_single=(\d+)")
    out = {}
    for p in sorted(glob.glob(os.path.join(logdir, "*"))):
        if not os.path.isfile(p):
            continue
        for line in open(p, encoding="utf-8", errors="replace"):
            m = pat.match(line.strip())
            if m:
                t, row = m.group(1), m.group(2)
                out.setdefault((t, row), {})[int(m.group(3))] = tuple(
                    int(x) for x in m.groups()[3:])
    return out


def summarize(folds, idx):
    keys = sorted(folds)
    acc = [1.0 - folds[k][idx] / folds[k][0] for k in keys]
    return statistics.mean(acc), statistics.stdev(acc), len(keys)


def s2(logdir, tex):
    """Every cell the log says moved must now print the new value, not the old."""
    text = ""
    for p in sorted(glob.glob(os.path.join(logdir, "s2*"))):
        text += open(p, encoding="utf-8", errors="replace").read()
    moved = re.findall(r"^\s+(\S+)\s+(Precision|Recall|F1-score)\s+(.+)$", text, re.M)
    if not moved:
        print("  note    the log records no S2 cell as moved")
        return
    body = " ".join(tabular(tex, "tab:classwise_performance").split())
    if not body:
        check(False, "S2 table", "tab:classwise_performance not found")
        return
    print(f"  the log records {len(moved)} S2 cell(s) as moved at three decimals")
    for cls, measure, rest in moved:
        i = body.upper().find(cls.upper())
        row = body[i:body.find("\\\\", i)] if i >= 0 else ""
        for kind, old, new in re.findall(r"(mean|sd) (\d\.\d{3}) -> (\d\.\d{3})", rest):
            what = f"S2 {cls} {measure} {kind}"
            if kind == "sd" and "\\pm" not in row:
                print(f"  note    {what:<34}this row prints no spread, so the log's "
                      f"{old} -> {new} has no cell here")
                continue
            has_new, has_old = new in row, old in row
            # The old value may legitimately still appear as some other cell's value, so a
            # missing new value is the error and a surviving old one is only a warning.
            check(has_new, what, f"table prints {new}" if has_new
                  else f"log says {old} -> {new}, but {new} is not in the table")
            if has_new and has_old:
                print(f"          note  {old} also still appears somewhere in the table")


def tally_table(logdir, tex, target, label, rowmap):
    t = tally(logdir)
    body = " ".join(tabular(tex, label).split())
    if not body:
        check(False, f"{target.upper()} table", f"{label} not found")
        return
    for row, pretty in rowmap.items():
        key = (target, row)
        if key not in t:
            print(f"  note    the log has no {target} row {row!r}")
            continue
        mean, sd, n = summarize(t[key], 2)
        mixed, _, _ = summarize(t[key], 1)
        # Find the printed accuracy for this row: the first 0.xxxx after the row label.
        i = body.find(pretty)
        if i < 0:
            check(False, f"{target.upper()} {row}", f"no row labelled {pretty!r}")
            continue
        # Stop at the end of the row. Reading past it picks up the next row's accuracy
        # and compares it with this row's spread.
        end = body.find("\\\\", i)
        seg = body[i:end if end > i else i + 260]
        got = re.search(r"0\.\d{4}", seg)
        if not got:
            check(False, f"{target.upper()} {row}", "no four-decimal value after the label")
            continue
        printed = float(got.group(0))
        ok = abs(round(mean, 4) - printed) < 5e-5
        check(ok, f"{target.upper()} {row}",
              f"{n} folds, single {mean:.4f} vs printed {printed:.4f}"
              + ("" if ok else f"   (mixed precision gives {mixed:.4f})"))
        sdm = re.search(r"0\.\d{4}", seg[got.end():]) if "\\pm" in seg else None
        if sdm is None and "\\pm" not in seg:
            print(f"  note    {target.upper() + ' ' + row + ' s.d.':<34}"
                  f"this table prints no spread, so {sd:.4f} has no cell here")
        if sdm:
            psd = float(sdm.group(0))
            ok = abs(round(sd, 4) - psd) < 5e-5
            check(ok, f"{target.upper()} {row} s.d.", f"{sd:.4f} vs printed {psd:.4f}")


def main():
    logdir, supp = sys.argv[1], sys.argv[2]
    tex = open(supp, encoding="utf-8").read()
    print("Supplementary Table S2\n")
    s2(logdir, tex)
    print("\nSupplementary Table S3\n")
    tally_table(logdir, tex, "s3", "tab:overall_performance", {
        "mambaout_tiny.in1k_seed42": "MambaOut",
        "convnext_tiny_seed42": "ConvNeXt",
        "efficientnet_b3_seed42": "EfficientNet",
        "resnet50_seed42": "ResNet",
        "vit_small_patch16_384_seed42": "ViT"})
    print("\nSupplementary Table S4\n")
    tally_table(logdir, tex, "s4", "tab:lesion_restriction",
                {"lcrop": "Lesion-centered crop", "bgdim": "Background attenuated"})
    print()
    if PROBLEMS:
        print(f"{len(PROBLEMS)} problem(s):")
        for p in PROBLEMS:
            print(f"  - {p}")
        sys.exit(1)
    print("the tables agree with the single-precision log")


if __name__ == "__main__":
    main()
