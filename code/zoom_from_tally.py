"""Re-derive Supplementary Table S15 from the per-fold counts in the fp32 probe log.

The table says what it is: accuracy as the out-of-fold total over the five folds, 7,991
images, and Delta as the difference from the unrestricted input of the same classifier.
The log's TALLY lines carry the fold's image count and its error count under both
precisions for every condition, so the table is a sum and a division away and needs
neither a GPU nor the dataset.

This matters because the per-fold archives under archives/outputs/zoom/ are the earlier
mixed-precision run. They still reproduce themselves, and the mixed_precision block of
the JSON written here is that run's table, which is what collect_zoom.py prints from
them. The table in the manuscript is the single-precision one, and the log is its
per-fold record.

    python3 zoom_from_tally.py <log>... [--supp <supplementary.tex>] [--json <out>]

With --supp, every value the table prints is compared with the derived one, together
with the two counts in Supplementary Table S17's caption, which are read from the same
classifier. With --json, the comparison is skipped unless --supp is given as well.
"""
import json
import re
import statistics
import sys

ORDER = ["base", "mask_m0", "mask_m20", "mask_m40", "mask_m60", "mask_m80", "mask_m120"]
LABEL = {"base": "Unrestricted (reference)", "mask_m0": "Lesion box, margin 0%",
         "mask_m20": "Lesion box, margin 20%", "mask_m40": "Lesion box, margin 40%",
         "mask_m60": "Lesion box, margin 60%", "mask_m80": "Lesion box, margin 80%",
         "mask_m120": "Lesion box, margin 120%"}
TALLY = re.compile(r"^TALLY\s+s15\s+(\S+)\s+fold=(\d+)\s+n=(\d+)\s+"
                   r"wrong_mixed=(\d+)\s+wrong_single=(\d+)")
PROBLEMS = []


def read(paths):
    """{condition: {fold: (n, wrong_mixed, wrong_single)}}, refusing a repeated fold."""
    rows = {}
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                m = TALLY.match(line.strip())
                if not m:
                    continue
                cond, fold = m.group(1), int(m.group(2))
                rec = tuple(int(x) for x in m.groups()[2:])
                have = rows.setdefault(cond, {}).get(fold)
                if have is not None and have != rec:
                    sys.exit(f"{cond} fold {fold} appears twice with different counts, "
                             f"{have} and {rec}")
                rows[cond][fold] = rec
    return rows


def derive(rows, precision):
    """Pooled accuracy, fold-to-fold s.d. and error counts, per condition."""
    idx = {"mixed": 1, "single": 2}[precision]
    out = {}
    for cond in ORDER:
        folds = rows[cond]
        keys = sorted(folds)
        n = sum(folds[k][0] for k in keys)
        wrong = sum(folds[k][idx] for k in keys)
        per = [folds[k][idx] for k in keys]
        acc = [1.0 - folds[k][idx] / folds[k][0] for k in keys]
        out[cond] = {"n": n, "wrong": wrong, "wrong_per_fold": per,
                     "accuracy": 1.0 - wrong / n,
                     "fold_sd": statistics.stdev(acc) if len(acc) > 1 else 0.0}
    base = out["base"]["accuracy"]
    for cond in ORDER:
        # From the unrounded accuracies, which is what collect_zoom.py does. Taking the
        # difference of the rounded ones moves margin 0% from -0.0501 to -0.0500.
        out[cond]["delta_pp"] = None if cond == "base" else \
            (out[cond]["accuracy"] - base) * 100
    return out


def as_json(rows):
    d = {p: derive(rows, p) for p in ("single", "mixed")}
    def block(t):
        return {
            "accuracy": {c: round(t[c]["accuracy"], 6) for c in ORDER},
            "delta_pp": {c: round(t[c]["delta_pp"], 4) for c in ORDER if c != "base"},
            "fold_sd": {c: round(t[c]["fold_sd"], 6) for c in ORDER},
            "wrong": {c: t[c]["wrong"] for c in ORDER},
            "wrong_per_fold": {c: t[c]["wrong_per_fold"] for c in ORDER},
            "largest_change_pp": round(max(t[c]["delta_pp"] for c in ORDER
                                           if c != "base"), 4),
        }
    out = {"folds": len(rows["base"]), "images": d["single"]["base"]["n"],
           "precision": "single",
           "derived_from": "the TALLY lines of archives/logs/fp32_probe/s15.gpu*.log",
           "misclassified": d["single"]["base"]["wrong"]}
    out.update(block(d["single"]))
    out["mixed_precision"] = block(d["mixed"])
    out["mixed_precision"]["files"] = [f"zoom2_f{i}_s42.npz" for i in range(5)]
    out["mixed_precision"]["note"] = ("the earlier run, which archives/outputs/zoom/ "
                                      "holds and collect_zoom.py prints")
    return out


def printed(supp):
    tex = open(supp, encoding="utf-8").read()
    i = tex.find("\\label{tab:zoom_inference}")
    j = tex.find("\\begin{tabular", i)
    k = tex.find("\\end{tabular", j)
    out = {}
    for raw in tex[j:k].split("\\\\"):
        t = " ".join(raw.split())
        if t.count("&") != 2 or "Input at inference" in t:
            continue
        c = [re.sub(r"\\[a-zA-Z]+|[{}$]", "", x).strip() for x in t.split("&")]
        name = c[0].replace("\\", "").strip()
        key = next((k2 for k2, v in LABEL.items()
                    if name in (v, v.replace("%", "\\%"))), None)
        if key is None:
            continue
        acc = re.search(r"0\.\d+", c[1])
        d = re.search(r"[-+]\d+\.\d+", c[2])
        if acc:
            out[key] = (float(acc.group(0)), float(d.group(0)) if d else None)
    return out, tex


def check(ok, what, detail):
    print(("  ok      " if ok else "  PROBLEM ") + f"{what:<32}{detail}")
    if not ok:
        PROBLEMS.append(what)


def compare(rows, supp):
    t = derive(rows, "single")
    pub, tex = printed(supp)
    print(f"against the supplement, {len(pub)} rows parsed")
    for cond in ORDER:
        if cond not in pub:
            check(False, LABEL[cond], "no printed row")
            continue
        acc_p, d_p = pub[cond]
        acc = t[cond]["accuracy"]
        check(abs(round(acc, 4) - acc_p) < 5e-5, LABEL[cond],
              f"accuracy {acc:.6f} -> {acc:.4f}, printed {acc_p:.4f}")
        if cond == "base":
            continue
        dd = t[cond]["delta_pp"]
        check(d_p is not None and abs(round(dd, 2) - d_p) < 5e-3, LABEL[cond] + ", delta",
              f"{dd:+.5f} -> {dd:+.2f}, printed {d_p:+.2f}")

    # Supplementary Table S17 splits this classifier's maps by whether its prediction was
    # correct, so the counts in its caption are this condition's totals. Taking the
    # misclassified count from the mixed-precision run instead would be a silent error of
    # four images.
    i = tex.find("\\label{tab:overlap_by_correctness}")
    cap = " ".join(tex[max(tex.rfind("\\caption{", 0, i), 0):i].split()) if i > 0 else ""
    for pat, want, what in (
            (r"of which ([\d{},]+) are misclassified", t["base"]["wrong"],
             "S17, misclassified"),
            (r"over all ([\d{},]+) images", t["base"]["n"], "S17, images")):
        m = re.search(pat, cap)
        if m is None:
            print(f"  note    {what:<32}the caption does not give it")
            continue
        got = int(re.sub(r"[^0-9]", "", m.group(1)))
        other = derive(rows, "mixed")["base"]["wrong"]
        check(got == want, what, f"caption {got}, the log's single-precision total {want}"
              + ("" if got == want else f" (mixed precision gives {other})"))


def main():
    argv = sys.argv[1:]
    supp = out = None
    logs = []
    i = 0
    while i < len(argv):
        if argv[i] == "--supp":
            supp = argv[i + 1]; i += 2
        elif argv[i] == "--json":
            out = argv[i + 1]; i += 2
        else:
            logs.append(argv[i]); i += 1
    if not logs:
        sys.exit(__doc__)

    rows = read(logs)
    missing = [c for c in ORDER if len(rows.get(c, {})) != 5]
    if missing:
        sys.exit(f"the log does not hold five folds for {missing}")
    print(f"{len(rows)} conditions x 5 folds read from {len(logs)} log file(s)\n")

    for precision, note in (("single", "the published table"),
                            ("mixed", "the earlier run, as the archives hold it")):
        t = derive(rows, precision)
        print(f"{precision} precision -- {note}")
        print(f"  {'condition':<26}{'n':>6}{'wrong':>7}{'accuracy':>11}"
              f"{'delta pp':>10}{'fold s.d.':>11}")
        for cond in ORDER:
            r = t[cond]
            dd = "" if cond == "base" else f"{r['delta_pp']:+.5f}"
            print(f"  {LABEL[cond]:<26}{r['n']:>6}{r['wrong']:>7}"
                  f"{r['accuracy']:>11.6f}{dd:>10}{r['fold_sd']:>11.6f}")
        print()

    if supp:
        compare(rows, supp)
    if out:
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(as_json(rows), fh, indent=1)
            fh.write("\n")
        print(f"\nwrote {out}")

    if PROBLEMS:
        print(f"\n{len(PROBLEMS)} problem(s): " + ", ".join(PROBLEMS))
        sys.exit(1)
    if supp:
        print("\nevery value Supplementary Table S15 prints re-derives from the log, and "
              "Supplementary Table S17's counts agree with it")


if __name__ == "__main__":
    main()
