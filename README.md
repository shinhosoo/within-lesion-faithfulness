# Lesion localization does not establish within-lesion faithfulness in dermoscopic classifiers

Code, lesion-grouped splits and measurement archives for a study that separates **where a
saliency map falls** from **what it says about the classifier**. An attention map learned from
the classification loss alone is routinely presented as evidence that a dermoscopic model
attends to the lesion. This code tests that on HAM10000 under lesion-grouped five-fold
cross-validation, scoring each map against three references that require no learning: a centered
Gaussian, an area-matched random map, and the lesion mask used as its own map. It adds an
insertion and deletion test restricted to the lesion interior, which measures how a map ranks
positions inside the lesion without crediting it for locating the lesion.

> The HAM10000 images and the Tschandl lesion masks are **CC BY-NC** and are not redistributed
> here. The 874 model checkpoints are omitted for size. Every overlap number the paper prints
> remains checkable from the archives alone, with no images and no GPU, which is the procedure
> under *Verification from the archives* below.

## Repository contents

```
code/                 the code behind every reported number
splits/               the lesion-grouped splits and the cached cross-validation fold table
archives/             every per-fold measurement archive, mirroring the output tree
results/              collected tables and two sets of verification dumps
MANIFEST.md           which archive produces which published row, and its provenance
REPRODUCIBILITY.md    configuration, script entry points and figure commands
```

`code/` contains only what a reported number depends on. Earlier drafts compared a two-stage
routing model, swept augmentation and zoom settings, and carried checkers written against table
numbers that have since changed. Neither those experiments nor their checkers are included.

## Scope of reproduction

| Target | Provided here | Also required |
| --- | --- | --- |
| Overlap values and their fold spreads | the per-fold archives | Python, NumPy, SciPy |
| Supplementary Tables S13 and S14 | the recorded fold values, plus archives for two rows of S14's upper block | images and the frozen classifier, for the remaining rows |
| Supplementary Table S18 | the sixty per-fold archives, their reading as JSON, and their hashes | Python; PyTorch only to re-read the archives with the measurement script |
| Supplementary Tables S2, S3, S4 and S15 | the per-fold counts of the single-precision re-measurement | Python |
| Figures 2 and 3 | the plotting scripts | the matching `main.tex` and `supplementary.tex`, and Matplotlib |
| Figure 4 | the qualitative plotting script | images, masks, the frozen classifier, two head checkpoints |
| Training and evaluation from scratch | the source and the fixed folds | images, masks, the model dependencies, a GPU |

## Verification from the archives

The commands below run from the repository root. They require no images, no masks, no
checkpoints, no GPU and no `env.sh`, since `verify_heads.py` reads only the archives.

```bash
python -m pip install numpy scipy
mkdir -p dumps
python code/verify_heads.py dump dumps/main.json     --root archives/outputs
python code/verify_heads.py dump dumps/convnext.json --root archives/outputs_convnext_tiny
python code/verify_heads.py dump dumps/nested.json   --root archives/outputs_nested
```

The three dumps correspond to `results/package_main.json`, `results/package_convnext_tiny.json`
and `results/package_nested.json`, the output of the same commands against this repository.
Absolute output paths differ by machine and need not match. The `table` mode prints the same
values with the diagnostics:

```bash
python code/verify_heads.py table --root archives/outputs
```

SciPy is required only for the paired tests. Without it the script writes every other value and
reports that the tests were skipped.

Two caveats apply when matching a row against that output. The pooled scale-and-shift and
class-activation rows are under `archives/outputs/headswap_s42_pooled`, so no single root holds
them all. The script also still prints some older row labels: `Linear score (rank 1)` is the
rank-4 head, whose 773 mechanism parameters are 192×4+5. [MANIFEST.md](MANIFEST.md) gives the
archive, the reading and the source stamp for every published row, and is the reference a
published value should be matched against.

The manuscript tables print means and spreads over the five folds. The fold values themselves,
which the paired tests, fold counts and correlations in the text rest on, are in
`results/fold_level.json`, rebuilt by `python code/collect_fold_level.py results/fold_level.json`.

Each archive carries its fold's per-image overlap measures, the lesion area, the predicted and
true labels, and the fold-level faithfulness summary. The five folds hold 7,991 images, of which
7,989 have a lesion area that area matching can use, the two exceptions being masks covering the
whole frame. Deletion and insertion curves are summarised to six numbers per fold rather than
stored step by step.

## Setup for training and model-based evaluation

Required only for the last two targets in the scope table above.

```bash
cd code && cp env.sh.example env.sh && $EDITOR env.sh && . ./env.sh
```

`env.sh` must be sourced from inside `code/`, since its example split path is relative to the
working directory. **The defaults in this repository are not the published configuration**, which
is why the example sets `HAM_AUGMENT=paper` and `HAM_WEIGHT_DECAY=5e-5` explicitly.
[REPRODUCIBILITY.md](REPRODUCIBILITY.md) lists the dependencies, every setting that differs, and
the entry point for each experiment. The supplied `splits/cv_folds.csv` defines the published
folds and is preferable to a regenerated one.

## Not included

**Images and lesion masks.** The HAM10000 images and the Tschandl segmentation masks are CC BY-NC
and are not redistributed here. Their original releases are the source, under those terms.
`configs/data_config.py` resolves the two image folders case-insensitively, since the distribution
spells `part_1` upper case and `part_2` lower.

**HAM10000 metadata beyond the split definition.** The split files carry `lesion_id`, `image_id`,
`diagnosis` and `fold` only. The `dx_type`, `age`, `sex` and `localization` columns belong to the
same CC BY-NC release as the images, no code here reads them, and `make_splits.py` drops them, so
regenerating the splits from a local copy of the release reproduces these files exactly.

**Model checkpoints.** 874 `.pth` files, derivable by re-running the training.

## License

The code in `code/` and the documentation in this repository are released under the MIT License
([LICENSE](LICENSE)).

**Data derived from HAM10000.** The four files under `splits/` carry `image_id`, `lesion_id`,
`diagnosis` and `fold`. The first three are taken from `HAM10000_metadata.csv` in the HAM10000
release, created by Philipp Tschandl, Cliff Rosendahl and Harald Kittler and published at
<https://doi.org/10.7910/DVN/DBW86T> under the Creative Commons Attribution-NonCommercial 4.0
International Public License, which that record gives as its Terms of Use. The licence text is at
<https://creativecommons.org/licenses/by-nc/4.0/>. `image_id` appears again in 393 of the 423
archives under `archives/`, and the integer form of `diagnosis` in 388 of them as `label`. Those
files are adapted material under that licence. They are not covered by the MIT License above and
may be redistributed for non-commercial purposes only, with this attribution kept.

The adaptation is this. The `dx` column was renamed to `diagnosis` and lower-cased, and in the
archives it is carried as an integer label. The `dx_type`, `age`, `sex` and `localization` columns
were dropped. A lesion-grouped fold index was added as `fold`. No metadata value was otherwise
altered.

The lesion masks are not redistributed here. The release asks that work using them cite Tschandl
et al., *Nature Medicine* 2020, and the manuscript does.

## Further documentation

- [MANIFEST.md](MANIFEST.md) — a published row to its archive, and the measurement history
  bearing on its interpretation
- [REPRODUCIBILITY.md](REPRODUCIBILITY.md) — configuration, script entry points, results files,
  figure commands
