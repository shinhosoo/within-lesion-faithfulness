"""Prepare the fp32 probe logs for the repository, and say exactly what changed.

Two things in the raw logs do not belong in a public repository, and neither is a
measurement. Progress bars, which tqdm writes by returning the carriage and overwriting
the line, so a file holds every intermediate frame even though a terminal only ever
showed the last one. And the absolute paths of the machine the runs were made on, which
name directories and a person.

Both are removed mechanically, so the result is not an edited log:

  - Each line is split on the carriage return and only the last field is kept, which is
    what the terminal displayed. A line tqdm never touched has no carriage return and is
    passed through whole, trailing text on a progress line included.
  - Each root in the map is replaced by the name given for it, longest first so that a
    shorter root cannot consume the head of a longer one.

Nothing else is touched: no line is dropped and no number is altered. The script then
checks its own work and writes nothing at all if any root, or any directory name from
one, survived anywhere in the output.

The map is a file of `path<TAB>name` lines, and it is not in this repository, because
those paths are the thing being removed and shipping them here would publish what the
scrub exists to take out. The names the shipped logs use are $PROJECT_ROOT,
$HAM_DATA_ROOT, $HAM_MASK_DIR, $HAM_DATA_PARENT and $HOME.

    python3 scrub_logs.py <in dir> <out dir> <map file>
"""
import os
import sys


def read_map(path):
    """[(root, name)] longest first, with the guard strings the output must not hold."""
    pairs = []
    for raw in open(path, encoding="utf-8"):
        line = raw.rstrip("\n")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if "\t" not in line:
            sys.exit(f"{path}: a map line needs a tab between the path and the name:\n"
                     f"  {line!r}")
        root, name = line.split("\t", 1)
        pairs.append((root.strip(), name.strip()))
    if not pairs:
        sys.exit(f"{path}: no substitutions given")
    pairs.sort(key=lambda p: -len(p[0]))
    # A root can survive in pieces: a directory named after a person is as revealing as
    # the whole path. Guard on every component of every root that is long enough to
    # identify anything, and on the carriage return the first pass removes.
    guards = {"\r"}
    for root, _ in pairs:
        guards.add(root)
        for part in root.replace("\\", "/").split("/"):
            if len(part) > 3:
                guards.add(part)
    for _, name in pairs:
        guards.discard(name)
    return pairs, sorted(guards, key=len, reverse=True)


def scrub(text, pairs):
    text = text.replace("\r\n", "\n")
    text = "\n".join(line.split("\r")[-1] for line in text.split("\n"))
    for root, name in pairs:
        text = text.replace(root, name)
    return text


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    src, dst, mapping = sys.argv[1], sys.argv[2], sys.argv[3]
    pairs, guards = read_map(mapping)
    names = sorted(n for n in os.listdir(src) if os.path.isfile(os.path.join(src, n)))

    # Scrub everything and check it before a single file is written, so a map that misses
    # a root cannot leave half the logs published and half not.
    done = {}
    for n in names:
        # newline="" or Python's universal newlines turns every carriage return into a
        # line feed on the way in, which splits the progress bars into lines instead of
        # letting them be collapsed. That read silently defeats the whole script.
        with open(os.path.join(src, n), encoding="utf-8", errors="replace",
                  newline="") as fh:
            raw = fh.read()
        new = scrub(raw, pairs)
        bad = [g for g in guards if g in new]
        if bad:
            sys.exit(f"{n}: {len(bad)} guard string(s) survived the scrub, "
                     f"nothing written")
        a = sum(1 for ln in raw.split("\n") if ln.startswith("TALLY"))
        b = sum(1 for ln in new.split("\n") if ln.startswith("TALLY"))
        if a != b:
            sys.exit(f"{n}: {a} TALLY lines in, {b} out")
        done[n] = (raw, new, b)

    os.makedirs(dst, exist_ok=True)
    print(f"{'file':<34}{'bytes in':>10}{'bytes out':>11}{'frames':>8}{'TALLY':>7}")
    tin = tout = 0
    for n in names:
        raw, new, tally = done[n]
        with open(os.path.join(dst, n), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(new)
        tin += len(raw.encode()); tout += len(new.encode())
        print(f"  {n:<32}{len(raw.encode()):>10}{len(new.encode()):>11}"
              f"{raw.count(chr(13)):>8}{tally:>7}")
    print(f"\n  {'total':<32}{tin:>10}{tout:>11}")
    print(f"{len(names)} files, {tin - tout} bytes of progress bars and path text "
          f"removed, every TALLY line kept, {len(guards)} guard string(s) checked")


if __name__ == "__main__":
    main()
