"""Check every cell of Supplementary Table S18 against the shipped JSON.

The table was typeset from a tabular body the measurement script emitted, and the JSON
was written by the same script on a later run. They agree only if the run was the same
one, so this compares all of it rather than a sample: the value, its fold-to-fold
standard deviation, the centred Gaussian measured in the same pass, the paired
difference, both ends of its interval, and the count of folds in which it was positive.

    python3 check_s18_json.py <perturb_heads.json> <supplementary.tex>
"""
import json
import re
import sys

BLOCK = {"Within-lesion": "within_lesion", "Whole-image": "whole_image"}
HEAD = {"Explanation head": "unsup", "\\quad no global path": "nomacro",
        "\\quad $+$ mask supervision": "sup"}
BASE = {"Blur $\\sigma=20$": "blur20", "Blur $\\sigma=5$": "blur5",
        "Per-image mean": "mean", "Noise": "noise"}
PROBLEMS = []


def rows(supp):
    tex = open(supp, encoding="utf-8").read()
    i = tex.find("\\label{tab:perturb_heads}")
    j = tex.find("\\begin{tabular", i)
    k = tex.find("\\end{tabular", j)
    block = head = None
    out = []
    for raw in tex[j:k].split("\\\\"):
        t = " ".join(raw.split())
        for key, val in BLOCK.items():
            if "multicolumn" in t and key in t:
                block = val
        if "multicolumn" in t or t.count("&") != 6:
            continue
        c = [x.strip() for x in t.split("&")]
        c[0] = re.sub(r"^\\(midrule|addlinespace|toprule)\s*", "", c[0]).strip()
        if c[0] in HEAD:
            head = HEAD[c[0]]
        if head is None or block is None or c[1] not in BASE:
            continue
        out.append((block, head, BASE[c[1]], c[2:]))
    return out


def nums(cell):
    return [float(x) for x in re.findall(r"[-+]?\d+\.\d+", cell.replace("$", ""))]


def cmp(got, want, dec, what):
    ok = want is not None and abs(round(got, dec) - want) < 10 ** -(dec + 1) * 5.0001
    tag = "  ok      " if ok else "  PROBLEM "
    print(f"{tag}{what:<46}{got:+.6f} -> {got:+.{dec}f}  printed {want:+.{dec}f}"
          if want is not None else f"  PROBLEM {what:<46}nothing printed")
    if not ok:
        PROBLEMS.append(what)


def main():
    doc = json.load(open(sys.argv[1], encoding="utf-8"))
    table = rows(sys.argv[2])
    print(f"{len(table)} rows parsed from the table, "
          f"{sum(len(v) for v in doc['measures'].values())} entries in the JSON\n")
    if len(table) != 24:
        PROBLEMS.append(f"expected 24 rows, parsed {len(table)}")

    for block, head, base, cells in table:
        e = doc["measures"][block][f"{head}/{base}"]
        tag = f"{block.split('_')[0]:<6}{head:<8}{base:<7}"
        v = nums(cells[0])
        cmp(e["mean"], v[0] if v else None, 4, tag + "value")
        cmp(e["sd"], v[1] if len(v) > 1 else None, 3, tag + "s.d.")
        g = nums(cells[1])
        cmp(e["centered_gaussian"], g[0] if g else None, 4, tag + "Gaussian")

        d = e.get("difference")
        printed_diff = nums(cells[2])
        if head == doc["paired_against"]:
            ok = not printed_diff and "---" in cells[2] and d is None
            print(("  ok      " if ok else "  PROBLEM ")
                  + f"{tag + 'reference head, no difference':<46}"
                  + ("as printed" if ok else f"printed {cells[2]!r}, JSON {d!r}"))
            if not ok:
                PROBLEMS.append(tag + "reference row")
            continue
        if d is None:
            PROBLEMS.append(tag + "difference missing from the JSON")
            print(f"  PROBLEM {tag + 'difference':<46}not in the JSON")
            continue
        cmp(d["mean"], printed_diff[0] if printed_diff else None, 4, tag + "difference")
        ci = nums(cells[3])
        cmp(d["ci95"][0], ci[0] if len(ci) > 1 else None, 4, tag + "interval, lower")
        cmp(d["ci95"][1], ci[1] if len(ci) > 1 else None, 4, tag + "interval, upper")
        m = re.match(r"(\d+)/(\d+)", cells[4])
        ok = m and int(m.group(1)) == d["folds_positive"] and int(m.group(2)) == d["folds"]
        print(("  ok      " if ok else "  PROBLEM ")
              + f"{tag + 'folds positive':<46}"
              + f"{d['folds_positive']}/{d['folds']}  printed {cells[4]}")
        if not ok:
            PROBLEMS.append(tag + "folds positive")

    print()
    if PROBLEMS:
        print(f"{len(PROBLEMS)} problem(s):")
        for p in PROBLEMS:
            print(f"  - {p}")
        sys.exit(1)
    print("every cell of Supplementary Table S18 agrees with the shipped JSON")


if __name__ == "__main__":
    main()
