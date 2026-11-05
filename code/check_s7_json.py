"""Check every cell of Supplementary Table S7 against the values file that produced it.

The table reports the centered-Gaussian reference at seven widths. Its claim is as much
about what does not move as about what does: the pointing score and the area-matched IoU
are said to be identical at every width, and only the energy ratio to vary. So the check is
not only that each printed number is in the file, but that the invariance the caption
asserts holds in the file too, and that the two image counts are the ones measured.

    python3 check_s7_json.py <sigma_sweep.json> <supplementary.tex>
"""
import json
import re
import sys

PROBLEMS = []


def check(ok, what, detail):
    print(("  ok      " if ok else "  PROBLEM ") + f"{what:<30}{detail}")
    if not ok:
        PROBLEMS.append(what)


def rows(supp):
    """The seven width rows as printed: [(frac, px, pointing, energy, iou)]."""
    tex = open(supp, encoding="utf-8").read()
    i = tex.find("\\label{tab:sigma_sweep}")
    j = tex.find("\\begin{tabular", i)
    k = tex.find("\\end{tabular", j)
    out = []
    for raw in tex[j:k].split("\\\\"):
        t = " ".join(raw.split())
        if "multicolumn" in t or t.count("&") != 4:
            continue
        # The published row is bold and carries a note, so a cell is not always a bare
        # number. Take the first number in each cell; a cell with none is not a data row.
        c = [re.sub(r"\\[a-zA-Z]+|[{}$]", "", x).strip() for x in t.split("&")]
        got = [re.search(r"-?\d+\.?\d*", x) for x in c]
        if any(g is None for g in got):
            continue
        out.append(tuple(float(g.group(0)) for g in got))
    return out, tex


def main():
    doc = json.load(open(sys.argv[1], encoding="utf-8"))
    printed, tex = rows(sys.argv[2])
    print(f"  {len(printed)} width rows parsed, {len(doc['widths'])} in the values file\n")
    check(len(printed) == len(doc["widths"]), "row count",
          f"{len(printed)} printed, {len(doc['widths'])} measured")

    for frac, px, point, energy, iou in printed:
        key = f"{frac:g}"
        if key not in doc["widths"]:
            check(False, f"width {frac:g}", "not in the values file")
            continue
        w = doc["widths"][key]
        check(abs(round(w["sigma_px"], 1) - px) < 5e-2, f"width {frac:g} px",
              f"{w['sigma_px']:.4f} rounds to {w['sigma_px']:.1f}, printed {px}")
        for name, want, dec in (("pointing", point, 4), ("energy", energy, 4),
                                ("iou", iou, 6)):
            got = w[name]
            ok = abs(round(got, dec) - want) < 10 ** -dec * 0.501
            check(ok, f"width {frac:g} {name}",
                  f"{got:.8f} rounds to {got:.{dec}f}, printed {want}")

    print()
    # The caption's two invariance claims, which are the reason the table exists.
    for name in ("pointing", "iou"):
        vals = {round(w[name], 12) for w in doc["widths"].values()}
        check(len(vals) == 1, f"{name} invariant",
              f"one value at every width, {next(iter(vals)):.6f}" if len(vals) == 1
              else f"{len(vals)} distinct values, the caption says one")
    e = [doc["widths"][k]["energy"] for k in sorted(doc["widths"], key=float)]
    check(len(set(e)) == len(e), "energy varies", f"{len(set(e))} distinct values")
    check(all(a > b for a, b in zip(e, e[1:])), "energy monotone",
          "falls at every step" if all(a > b for a, b in zip(e, e[1:])) else "not monotone")

    print()
    cap = " ".join(tex[max(tex.rfind("\\caption{", 0, tex.find(
        "\\label{tab:sigma_sweep}")), 0):tex.find("\\label{tab:sigma_sweep}")].split())
    for pat, got, what in ((r"over all ([\d{},]+) cross-validation masks", doc["images"],
                            "caption, masks"),
                           (r"over the ([\d{},]+) of them", doc["images_with_usable_area"],
                            "caption, usable area")):
        m = re.search(pat, cap)
        if m is None:
            print(f"  note    {what:<30}the caption does not give it")
            continue
        want = int(re.sub(r"[^0-9]", "", m.group(1)))
        check(want == got, what, f"caption {want}, measured {got}")

    print()
    if PROBLEMS:
        print(f"{len(PROBLEMS)} problem(s):")
        for p in PROBLEMS:
            print(f"  - {p}")
        sys.exit(1)
    print("every cell of Supplementary Table S7 agrees with the values file, and the "
          "caption's invariance claims hold in it")


if __name__ == "__main__":
    main()
