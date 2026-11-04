# Reproducibility

Configuration, script entry points and figure commands. The archive behind each published row is
in [MANIFEST.md](MANIFEST.md), and the archive-only checks are in
[README.md](README.md#verification-from-the-archives).

## Environment

`verify_heads.py` uses NumPy, imports SciPy for the paired tests, and imports neither the model
nor the image loader, so archive inspection requires no `env.sh` and no dataset.

The model-based scripts require PyTorch and torchvision built for the target hardware, plus
timm, NumPy, pandas, Pillow, scikit-learn and tqdm. Matplotlib draws the figures.

This repository does not pin a training environment. The runs behind the published tables were
made on one RTX 4090 under the versions below, recorded from the environment that produced them.
Nothing here depends on these exact versions, and they are given so that a difference can be
located rather than guessed at.

```
torch==2.5.1+cu121        torchvision==0.20.1+cu121   timm==1.0.24
numpy==1.26.4             scipy==1.15.3               scikit-learn==1.7.2
pandas==2.3.3             pillow==12.0.0              matplotlib==3.10.8
tqdm==4.67.1
```

The four paths in `code/env.sh` are `HAM_DATA_ROOT`, `HAM_MASK_DIR`, `HAM_SPLIT_ROOT` and
`HAM_OUTPUT_ROOT`, naming the images, the masks, the splits and the run outputs. New output
belongs under a root of its own rather than inside `archives/`.

Without `env.sh` a model-based command fails inside `SkinCancer/data/data_config.py`, either on a
missing `splits/global_train.csv` or with `[Data] Found 0 images`. The built-in defaults point
inside the repository rather than at a local copy of the release, so that failure is a missing
`env.sh`.

## Configuration of the published runs

| Setting | Published | Repository default |
| --- | --- | --- |
| `HAM_AUGMENT` | `paper` | `code` |
| vertical flip probability | 0.2 | 0.5 |
| rotation range | ±30° | ±180° |
| brightness and contrast jitter | 0.1 each | 0.2 each |
| `HAM_WEIGHT_DECAY` | `5e-5` | `0.05` |
| `HAM_INNER_VAL` | unset, except the nested runs at `0.15` | unset |

The augmentation defaults live in `SkinCancer/data/transforms.py` and the weight decay in
`SkinCancer/configs/training_config.py`. The training logs record `wd=5e-5 aug=paper`.
`code/env.sh.example` sets both explicitly.

## Folds and evaluation sets

`splits/cv_folds.csv` is the definition of the published folds, and `improve/residual.py` reads
it when present. It is a lesion-grouped five-fold split of the pooled 7,991 training and
validation images, with fold sizes 1,599 / 1,598 / 1,598 / 1,598 / 1,598. Regeneration under a
different scikit-learn version yields a different lesion-disjoint partition with the same fold
sizes, so matching sizes are not evidence of matching folds.

Overlap is measured over the whole held-out fold; faithfulness over a subset, typically 400
images per fold. Each archive records its own counts, which require checking before
values are compared across tables.

The main protocol selects the head's epoch on the same held-out fold it then measures. The nested
protocol reserves an inner validation subset for that choice, at the cost of 15% of the head's
training data, which is why the two cannot be pooled.

## Script entry points

Paths are relative to `code/`.

| Component | Script |
| --- | --- |
| the single-stage classifier that is later frozen | `improve/residual.py` |
| the same classifier at the other seeds | `improve/flat_seed.py`, via `ablation/onestage.py` |
| the explanation heads and the map readings | `improve/head_swap.py` |
| the stage-specific and image-conditional readouts | `improve/deep_head.py`, `improve/cond_head.py` |
| removal of the classifier's global-feature path | `improve/ablate_macro.py` |
| Grad-CAM, integrated gradients, Gaussian, random map | `improve/refs_overlap.py` |
| the within-lesion measure and its shuffle control | `improve/within_refs*.py` |
| sensitivity of the measure to the perturbation baseline | `improve/perturb_variants.py` |
| the same sensitivity for the three explanation heads | `improve/perturb_heads.py` |
| that sensitivity pooled over folds | `collect_perturb.py` |
| accuracy when the lesion is magnified at inference | `collect_zoom.py` |
| the published reading of that table, from the probe log | `zoom_from_tally.py` |
| each reported accuracy computed under both precisions | `fp32_probe.py` |
| the check of Supplementary Table S18 against its values file | `check_s18_json.py` |
| the label-efficiency runs | `improve/sup_frac*.py` |
| the second backbone | `improve/bb_run.py` |
| heads trained on occlusion and gradient targets | `improve/distil.py`, `improve/grad_distil.py` |
| a head read through its own gradient | `improve/head_grad.py` |
| threshold-free pixel agreement | `improve/pixel_ap*.py` |
| background and crop restrictions, context-margin sweep | `improve/aug2.py`, `improve/zoom2.py` |
| archive summaries and value dumps | `verify_heads.py`, `collect_*.py`, `zoom2_table.py` |
| the per-fold values behind the reported tests | `collect_fold_level.py` |
| the inference cost table | `check_model_cost.py` |
| split generation and validation | `make_splits.py`, `check_split.py` |

### Sensitivity to the perturbation baseline

Insertion and deletion replace part of the image with a baseline, and every number reported here
uses one: the Gaussian blur with `k = 51` and `sigma = 20` in `improve/faithful.py`. That is a
parameter of the measure rather than a property of the maps.
`improve/perturb_variants.py` builds the reference maps once and then measures those same maps
under several baselines, so any movement in the scores comes from the perturbation alone, and
reports the rank correlation of the map ordering between baselines.

```bash
python -m improve.perturb_variants --fold 0 --baselines blur20,mean --n 100
```

`HAM_MASK_DIR` must hold the Tschandl segmentations, since the within-lesion curves rank
positions inside the lesion; the script stops before the loop if it does not. The defaults are
about 845k forward passes for one fold, so `--n`, `--maps` and `--baselines` are the knobs for a
smaller run.

The run behind the Discussion and Supplementary Table S16 is all five folds at `--n 400` with
`--baselines blur20,blur5,mean,noise`, one `perturb_variants_f*.npz` per fold. Its `blur20`
column is the published protocol and reproduces every faithfulness figure of Supplementary
Table S10 for all eight reference maps, which is the check that the path is right before the
other columns are read. Pooled over the folds the within-lesion ordering held against all three
alternatives, at mean Spearman correlations of 0.890, 0.881 and 0.900 and in five folds of five
for each. The whole-image ordering held against the narrower blur (0.890, five of five) and
against neither of the other two (0.510 and 0.505, no fold). The run covers the reference maps
and not the trained heads.

```bash
for F in 0 1 2 3 4; do
  python -m improve.perturb_variants --fold "$F" --baselines blur20,blur5,mean,noise --n 400
done
python code/collect_perturb.py --json results/perturb_variants.json
```

`improve/perturb_heads.py` carries those four baselines to the three explanation heads, which is
Supplementary Table S18. It loads each published checkpoint and measures it without retraining, so
its `blur20` column has to reproduce that head's faithfulness figures in Supplementary Table S10
before the other columns mean anything, and the script stops if it does not. The centred Gaussian
is measured in the same pass and reproduces the Centered Gaussian row of Supplementary Table S16.
Differences from the head trained on the classification loss alone are paired within folds and
summarized by an unadjusted two-sided 95% paired t interval over the five folds. Nothing is written
beside the published runs: results go to `--out`, which the script refuses to place inside a run
directory, and `torch.save` is refused outright.

The sixty per-fold archives are shipped under `archives/outputs/s16_heads/`, named
`<head>_<baseline>_f<fold>.npz`, so `--collect`, `--latex` and `--json` need no GPU, no dataset
and no checkpoints. They do need PyTorch, because the script imports it at the top for the
`torch.save` refusal and the generator, long before it knows which mode it is in.

`results/perturb_heads.json` is the reading of those archives that needs nothing but Python. It
carries every value of the table with the five fold values behind it, and the SHA-256 of each
archive, which is how the archives read are checked against the archives measured.

`check_s18_json.py` compares that file with the table as the supplement prints it, every value,
standard deviation, Gaussian column, paired difference, interval end and fold count, and imports
nothing outside the standard library. It takes the supplement by path, as the figure scripts do.

```bash
python code/check_s18_json.py results/perturb_heads.json "$HAM_PAPER_SRC/supplementary.tex"
```

```bash
bash code/run_s16_heads.sh                               # five folds, split across two GPUs
python code/improve/perturb_heads.py --collect ~/s16_heads
python code/improve/perturb_heads.py --latex   ~/s16_heads   # the table body
python code/improve/perturb_heads.py --json    ~/s16_heads results/perturb_heads.json
```

`collect_zoom.py` is the companion for Supplementary Table S15, which reports what the same
frozen classifier scores when the lesion's bounding box is re-input at inference with a margin
around it. The runs are shipped under `archives/outputs/zoom/`, so it needs no GPU and no
dataset, and it reproduces that measurement's largest change of 0.31 percentage points from the
unrestricted input.

Those archives are the earlier mixed-precision measurement. The table in the manuscript is the
single-precision one, whose largest change is 0.38 percentage points, and its per-fold record is
the probe log rather than an npz, because the probe refuses `np.savez` and `np.savez_compressed`
so that it could not write beside the published runs, and `code/fp32_probe.py` is that probe. It
left the per-fold image and error counts
on the `TALLY` lines of `archives/logs/fp32_probe/s15.gpu0.log` and `s15.gpu1.log`, and since the
table is an out-of-fold total, `zoom_from_tally.py` derives every value of it from those counts.
The same counts give the two totals in Supplementary Table S17's caption, 7,991 images of which
1,116 are misclassified, because that table splits the same classifier's maps by whether its
prediction was correct. `results/zoom_inference.json` holds both readings, the single-precision
one at the top level and the earlier one under `mixed_precision`.

The archived files carry the mask box only. A second box drawn from an attention map was measured
at the time and is neither shipped nor read, because the mechanism behind it is not one this
manuscript describes.

```bash
python code/zoom_from_tally.py archives/logs/fp32_probe/s15.gpu0.log \
    archives/logs/fp32_probe/s15.gpu1.log --json results/zoom_inference.json
python code/collect_zoom.py --json /tmp/zoom_mixed.json    # the earlier mixed-precision run
```

Adding `--supp` and the path to `supplementary.tex` checks the derived table against the printed
one, value by value, and checks the two counts in Supplementary Table S17's caption as well. The
second command is the one that reads the archives, and it is written to a scratch path here
because its values are the earlier reading and would otherwise replace the published ones.

`collect_perturb.py` reads the per-fold npz files together and reports the rank correlation
fold by fold with its mean, its spread and the number of folds in which the two orderings agree,
which is the per-fold accounting the rest of this study uses. It re-measures nothing, names any
fold whose file is absent and refuses to pool folds that were run with different maps,
variants or `--n`.

### The global-feature ablation

The auxiliary classifier that trains the explanation head reads the backbone's global feature as
well as the attention-pooled one, `Scaffold.fuse` being
`norm_out(macro_proj(macro) + micro_vec(pooled))`, so a map that pools nothing useful still
leaves the classification loss satisfiable. `improve/ablate_macro.py` replaces `fuse` on the
instance, zeroes and freezes `macro_proj`, and retrains the head under the same recipe, editing
nothing in `head_swap.py`. The zeroed projection's parameters stay in the total, so the
mechanism count is unchanged.

```bash
for f in 0 1 2 3 4; do python -m improve.ablate_macro --fold "$f"; done
python -m improve.ablate_macro --compare
```

`--compare` requires the baseline head archives under the same output root. Paired over the five
folds, removing the path raises the area-matched IoU from 0.474 ± 0.034 to 0.745 ± 0.007, in all
five folds, while the within-lesion value stays at −0.011. The archives are
`archives/outputs/headswap/hs_s3_mlp_nomacro_f*.npz`. The same run under the nested protocol
(`HAM_INNER_VAL=0.15`, a separate output root, its own classifier checkpoints) is in
`archives/outputs_nested/headswap/` and gives IoU 0.746 ± 0.015 and within-lesion
−0.013 ± 0.020.

## Pooled and rectified readings

Every head scores each position and pools with a spatial softmax. The three stem forms are
different measurements and are not interchangeable:

- `_mapsoftmax` — the pooled reading, the softmax distribution the pooling and the alignment loss
  act on. The main tables report this one.
- `_maprelu` — a later re-measurement of the same saved head with negative scores set to zero.
  Supplementary Table S10's rectified block is read from these.
- a plain stem — the training run's own convention, which for an unsupervised head is the
  rectified raw score, not necessarily the pooled reading.

`verify_heads.py` reads one root at a time, and the pooled `ssf_mapsoftmax` and `cam_mapsoftmax`
rows are under `archives/outputs/headswap_s42_pooled`, not `archives/outputs/headswap`. A
similarly named plain stem is not a substitute for either.

## Results files

| File under `results/` | Contents |
| --- | --- |
| `main.json`, `convnext.json`, `nested.json` | `verify_heads.py` dumps from the experiment machine |
| `package_main.json`, `package_convnext_tiny.json`, `package_nested.json` | the same three dumps, from this repository's archives |
| `main3.json` | the dump carrying the pixel-level values |
| `prose_values.json`, `rows.json`, `fig3_values.json` | in-text values, table rows, figure aggregates |
| `acc.json`, `seed_block.json`, `seed_folds.json` | accuracy, the seed block, the fold assignment |
| `map_modes.json`, `map_convnext.json`, `map_nested.json` | pooled against rectified, one per root |
| `interventions.json` | the pooled intervention rows |
| `onestage_7class_macro_95ci.csv` | the classifier's bootstrap interval |
| `side_analyses.json` | every value of Supplementary Tables S13 and S14, per fold |
| `fold_level.json` | the per-fold values behind the tests, fold counts and correlations in the text |
| `zoom_inference.json`, `perturb_variants.json` | Supplementary Tables S15 and S16, S15 in both precisions |
| `perturb_heads.json` | Supplementary Table S18, per fold, with each archive's hash |
| `source_hashes.json` | hashes of the files in `code/` as shipped, and what changed |

Some JSON keys and script names keep table or figure numbers from earlier drafts. Rows match by
stem and measure, with [MANIFEST.md](MANIFEST.md) giving the current table references.

## Figures

| figure | image | script | reproducible here |
| --- | --- | --- | --- |
| 1 | `fig_pipeline.png` | drawn by hand | not generated |
| 2 | `fig_quant.png` | `code/figures/make_figs.py` | byte for byte |
| 3 | `fig_shuffle.png` | `code/figures/make_figs.py` | byte for byte |
| 4 | `fig_qual.png` | `code/improve/fig_qual_head.py` | requires images, masks and two heads |

Figures 2 and 3 reproduce byte for byte, because `make_figs.py` and `figdata.py` parse their
values out of the manuscript at draw time. `HAM_PAPER_SRC` must point at the directory holding
`main.tex` and `supplementary.tex`, and the PNGs are written beside them.

```bash
HAM_PAPER_SRC=/path/to/manuscript python code/figures/make_figs.py
```

`code/make_fig3.py`, which despite its name derives Figure 2's aggregates, reads the archives
instead and writes `results/fig3_values.json`. Its `--out`, `--fig` and `--json` options choose
the archive root and the outputs. That is the check on the table values the figure reads.

Figure 4 requires the images, the masks and the two head checkpoints
`headswap/head_s3_mlp_f0.pth` and `headswap/head_s3_mlp_sup0.2_f0.pth` under the configured
output root. It draws the unsupervised head, the mask-supervised head and a centered Gaussian on
the same four images in the pooled reading.

```bash
python -m improve.fig_qual_head --fold 0 --out figs/fig_qual_head
```

Retraining the omitted checkpoints changes which examples are selected, so the panel will not be
identical to the published one.
