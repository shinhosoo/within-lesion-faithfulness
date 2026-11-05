# Manifest

The archive behind each published row, and the measurement history that bears on its
interpretation. The archive-only checks are in
[README.md](README.md#verification-from-the-archives) and the configuration and entry points in
[REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Table conventions

- Archive paths are relative to `archives/`. `outputs/headswap` means
  `archives/outputs/headswap`.
- A stem `s3_mlp_mapsoftmax` names the files `hs_s3_mlp_mapsoftmax_f0.npz` through `_f4.npz`.
- `folds` is the number of fold files in that set.
- `select` is the epoch-selection rule recorded in the archive: `att` is the validation alignment
  loss, `-` is the classification AUC. 365 of the 388 head archives record the field; 23 older
  ones omit it, and for those the rule is the one given here.
- `source` is the first ten hex digits of the SHA-1 of `improve/head_swap.py` at the time of the
  run. [Source stamps](#source-stamps) gives the limits of that identifier.
- The live output root is `outputs/headswap`. Rows whose archives are located elsewhere are
  marked, since one `--root` will not reproduce them.
- Row names follow the manuscript. `verify_heads.py` still prints `Linear score (rank 1)` for the
  rank-4 head; the configured bottleneck is four channels and the archives record 773 mechanism
  parameters, which is 192×4+5.

Pooled (`_mapsoftmax`) and rectified (`_maprelu`) rows are different measurements of the same
checkpoints, and
[Pooled and rectified readings](REPRODUCIBILITY.md#pooled-and-rectified-readings) gives the
distinction.

## Table 1, main text

| row | stem | archives | folds | select | source |
| --- | --- | --- | --- | --- | --- |
| Explanation head | `s3_mlp_mapsoftmax` | `outputs/headswap` | 5 | - | `708b2a70b0` |
| + mask supervision | `s3_mlp_sup0.2` | `outputs/headswap` | 5 | att | `76f6cd468f` |

## Table 2, main text, label efficiency

| row | stem | archives | folds | select | source |
| --- | --- | --- | --- | --- | --- |
| 10% | `s3_mlp_frac0.1_sup0.2` | `outputs/headswap` | 5 | att | `76f6cd468f` |
| 25% | `s3_mlp_frac0.25_sup0.2` | `outputs/headswap` | 5 | att | `76f6cd468f` |
| 50% | `s3_mlp_frac0.5_sup0.2` | `outputs/headswap` | 5 | att | `76f6cd468f` |
| 100% | `s3_mlp_frac1_sup0.2` | `outputs/headswap` | 5 | att | `76f6cd468f` |

## Table 3, main text, the nested protocol

Root `outputs_nested`.

| row | stem | archives | folds | select | source |
| --- | --- | --- | --- | --- | --- |
| Explanation head, unsupervised | `s3_mlp_mapsoftmax` | `outputs_nested/headswap` | 5 | - | `5668415cc6` |
| no global-feature path | `s3_mlp_nomacro` | `outputs_nested/headswap` | 5 | - | `5668415cc6` |
| with mask supervision | `s3_mlp_sup0.2` | `outputs_nested/headswap` | 5 | att | `76f6cd468f` |

## Table 4, main text, second backbone

Root `outputs_convnext_tiny`.

| row | stem | archives | folds | select | source |
| --- | --- | --- | --- | --- | --- |
| Explanation head | `s3_mlp_mapsoftmax` | `outputs_convnext_tiny/headswap` | 5 | - | `5668415cc6` |
| + mask supervision | `s3_mlp_sup0.2` | `outputs_convnext_tiny/headswap` | 5 | att | `76f6cd468f` |
| Grad-CAM and centered Gaussian | `refs_ov`, `refs2` | `outputs_convnext_tiny/headswap` | 5 | - | `76f6cd468f` |

## Supplementary Table S10, pooled reading

| row | stem | archives | folds | select | source |
| --- | --- | --- | --- | --- | --- |
| Scale and shift, per channel | `ssf_mapsoftmax` | `outputs/headswap_s42_pooled` **(not in `outputs/headswap`)** | 5 | - | `5668415cc6` |
| Linear score (rank 4) | `lowrank_mapsoftmax` | `outputs/headswap` | 5 | - | `5668415cc6` |
| Class activation map | `cam_mapsoftmax` | `outputs/headswap_s42_pooled` **(not in `outputs/headswap`)** | 5 | - | `5668415cc6` |
| Position-wise MLP | `mlp_mapsoftmax` | `outputs/headswap` | 5 | - | `5668415cc6` |
| Image-conditional | `cond_mapsoftmax` | `outputs/headswap` | 5 | - | `5668415cc6` |
| Linear score (rank 4), mask sup. | `lowrank_sup0.2` | `outputs/headswap` | 5 | att | `76f6cd468f` |
| Class activation map, mask sup. | `cam_sup0.2` | `outputs/headswap` | 5 | att | `76f6cd468f` |
| Position-wise MLP, mask sup. | `mlp_sup0.2` | `outputs/headswap` | 5 | att | `76f6cd468f` |
| Image-conditional, mask sup. | `cond_sup0.2` | `outputs/headswap` | 5 | att | `76f6cd468f` |
| MLP at the third stage, no global-feature path | `s3_mlp_nomacro` | `outputs/headswap` | 5 | - | `5668415cc6` |
| Conditional at the third stage | `s3_cond_mapsoftmax` | `outputs/headswap` | 5 | - | `5668415cc6` |
| with mask supervision | `s3_cond_sup0.2` | `outputs/headswap` | 5 | att | `76f6cd468f` |
| MLP, gradient target | `gradgc3_mlp_sup0.2` | `outputs/headswap` | 5 | - | `76f6cd468f` |

## Supplementary Table S10, rectified reading

The same checkpoints as the pooled block, read with negative scores set to zero.

| row | stem | archives | folds | select | source |
| --- | --- | --- | --- | --- | --- |
| Linear score (rank 4), second stage | `lowrank_maprelu` | `outputs/headswap` | 5 | - | `5668415cc6` |
| MLP, second stage | `mlp_maprelu` | `outputs/headswap` | 5 | - | `5668415cc6` |
| Conditional, second stage | `cond_maprelu` | `outputs/headswap` | 5 | - | `5668415cc6` |
| MLP, third stage | `s3_mlp_maprelu` | `outputs/headswap` | 5 | - | `5668415cc6` |
| Conditional, third stage | `s3_cond_maprelu` | `outputs/headswap` | 5 | - | `5668415cc6` |

## Reference maps

The eight reference maps — Grad-CAM at each of the four stages, integrated gradients, the lesion
mask, a centered Gaussian and a random map — are not head runs and are not in the `hs_*`
archives. The same names are used under every root.

| archives | contents | evaluation set |
| --- | --- | --- |
| `refs_ov_f*.npz` | overlap for all eight references | the whole held-out fold |
| `refs2_f*.npz` | faithfulness summaries | the subset, typically 400 images per fold |
| `refs2_shuf_f*.npz` | the label-shuffled control | the subset |

A reference row requires both the overlap and the faithfulness file.
`verify_heads.py` reports `folds with reference archives: none` when they are absent.

## Supplementary Table S18, the three heads under four perturbation baselines

The published head checkpoints measured again under each baseline without retraining. These are
not head runs and carry no source stamp, because `improve/perturb_heads.py` reads a checkpoint
and never writes one.

| archives | contents | evaluation set |
| --- | --- | --- |
| `outputs/s16_heads/<head>_<baseline>_f*.npz` | insertion and deletion summaries for one head under one baseline | the subset, 400 images per fold |

`<head>` is `unsup`, `nomacro` or `sup`, in the order Table 1 of the main text prints the three
rows, and `<baseline>` is `blur20`, `blur5`, `mean` or `noise`. Sixty files, five folds for each
of the twelve combinations. `results/perturb_heads.json` holds every value of the table at full
precision with the fold values behind it, and the SHA-256 of each archive, so the files read can
be checked against the files measured.

## Rows with no archive under the live root

- `ssf_mapsoftmax` and `cam_mapsoftmax` — under `outputs/headswap_s42_pooled`, as marked above.
- Supplementary Tables S13 and S14 — below.

## Supplementary Tables S13 and S14

The runs behind these two tables rebuild their targets during training rather than caching them,
so no stem or source stamp applies to most of their rows. Every value in both tables is in
`results/side_analyses.json` at full precision, keyed by table, block and row, with the run it
came from.

| table | row or block | source here | what can be checked |
| --- | --- | --- | --- |
| S13 | fold-level fit of the gradient-gated target | `results/side_analyses.json`, key `S13` | the fold values and their aggregates; regeneration of the target requires images and the classifier |
| S14 upper | gradient-gated target | `outputs/headswap/hs_gradgc3_mlp_sup0.2_f*.npz` | the summaries, from five fold archives |
| S14 upper | uniform-mask target at λ = 0.2 | `outputs/headswap/hs_mlp_sup0.2_f*.npz` | the summaries, from five fold archives |
| S14 upper | occlusion-sensitivity targets | `results/side_analyses.json`, key `S14.upper_block` | the recorded mean and SD; the `outputs/distil` archives are not shipped |
| S14 lower | attention and head-gradient maps, and their correlation | `results/side_analyses.json`, key `S14.lower_block` | the five-fold values; the gradient maps require a model-based re-run |

S14's lower block uses 400 images per fold and a different map reading from S10, so its overlap
values are not interchangeable with S10's whole-fold overlap.

---

# Provenance

## The single-precision probe logs

`archives/logs/fp32_probe/` holds the logs of the re-measurement that moved the affected tables
from mixed to single precision. They are the per-fold record for runs that wrote no archive: the
probe refuses `np.savez` and `np.savez_compressed`, so that it could not disturb the published
outputs, and its counts exist only in these logs.

| log | what it records | what re-derives from it |
| --- | --- | --- |
| `s15.gpu0.log`, `s15.gpu1.log` | `TALLY` lines: per fold, the image count and the error count under each precision, for the unrestricted input and the six margins | every value of Supplementary Table S15, and the two counts in Supplementary Table S17's caption |
| `s2.gpu0.log` | the overall accuracy under each precision, the number of predictions that differ, and each of the twelve cells of twenty-four that move at three decimals, old value to new | which entries of Supplementary Table S2 changed, and by how much |
| `s3.gpu*.log`, `s3_<backbone>.gpu*.log` | `TALLY` lines for the three readout seeds of the main backbone and for each of the four others | the error counts behind those accuracies |
| `s4.gpu*.log` | `TALLY` lines for the lesion-crop and dimmed-background conditions | the error counts behind those accuracies |

`code/zoom_from_tally.py` does the Supplementary Table S15 derivation, and with `--supp` checks it
against the manuscript. `code/check_logs_vs_tables.py` checks Supplementary Tables S2, S3 and S4
against the same logs, each against the part of them that records it. The logs carry no image identifiers, so they support the counts and the
accuracies and not any per-image grouping.

The shipped copies were passed through `code/scrub_logs.py`, which collapses each progress bar to
the line a terminal displayed and replaces the absolute roots of the experiment machine with the
names the logs now use. It drops no line and alters no number, and it writes nothing at all if any
root, or any directory name taken from one, survived anywhere in the output.

The root-to-name map it reads is not in this repository, because those paths are the thing the
scrub exists to remove and shipping them here would publish what was taken out.

`code/fp32_probe.py` is the probe that wrote these logs. It runs each target twice, once under
`torch.amp.autocast` as the experiments ran it and once with autocast replaced, and reimplements
nothing: every target calls the functions its own experiment called, so a difference can only come
from the precision. Before any project module is imported it replaces `np.savez`,
`np.savez_compressed`, `np.lib.format.open_memmap`, `torch.save` and `pandas.DataFrame.to_csv`
with functions that raise, and it refuses to create a directory or open a file for writing
anywhere under the project, the splits or the data. That is why the re-measurement left logs and
no archives.

## Source stamps

The stamp is the first ten hex digits of the SHA-1 of `improve/head_swap.py`. `source_id()` is
defined there and imported by `deep_head.py` and `cond_head.py`, so every module stamps
`head_swap.py`'s hash and the stamp does not see changes in the importing modules. Two archives
sharing a stamp may have been written by different versions of the module that produced them.

`improve/head_swap.py` as shipped no longer hashes to the `5668415cc6` its archives carry,
because the file was edited after they were written. An earlier version carried a further
mechanism that no table reports, together with the scripts that trained and scored it, and it has
been removed with its archives. Supplementary Table S10's five mechanisms are the scale-and-shift, rank-4, class-activation-map, position-wise MLP and
image-conditional heads, each a `Scaffold` subclass that builds its own map.
`results/source_hashes.json` records what was removed and the new hash. Nothing else in the file
changed.

## The rectified block was re-measured and one row changed

Those five rows were first measured by a version of `head_swap.py` hashing to `76f6cd468f`, which
is not in this repository. Loading each saved head under the shipped code and re-measuring it,
which is what `--measure-only --map relu` does, reproduces four of the five to three decimals
including the spreads. The rank-4 row did not. Measured from `head_lowrank_f*.pth` it gives

```
earlier     0.514 / 0.524 / 0.298 / +0.050 / -0.065
re-measured 0.294 / 0.288 / 0.261 / +0.028 / +0.005
```

for pointing, energy, IoU, whole-image insertion minus deletion, and the within-lesion value, and
agrees exactly with `outputs/headswap/seedrun_unsorted` and `outputs/headswap_B`. The earlier
figure came from `outputs/headswap/pre_retrain`: the head was retrained after that measurement,
the pooled row was re-measured and the rectified one was not, so two rows of one head in one
table described two different checkpoints. The paper prints the re-measured row and every
rectified row is read from `hs_*_maprelu_f*`.

## The two sets of dumps

`results/main.json`, `convnext.json` and `nested.json` were produced on the machine the
experiments ran on. `results/package_*.json` are the same three commands run against this
repository's archives. For the two replication roots every measured value agrees. The main root
differs in three places, all in rows the paper does not print:

- the later dump gained rows the earlier one lacked: every `_mapsoftmax` and `_maprelu` stem, and
  the global-feature ablation;
- `hs_lowrank_f*.npz` is no longer at the live root. The head was retrained and the earlier
  measurement is under `outputs/headswap/pre_retrain`. The published rank-4 row comes from
  `lowrank_mapsoftmax` and `lowrank_maprelu`, both present;
- the plain `ssf` row differs in the third decimal, being a different run. The published row is
  `ssf_mapsoftmax` under `outputs/headswap_s42_pooled` and reproduces to every printed digit.

The draft-era dumps keyed by table numbers the submitted supplement does not use are not shipped.
`package_*.json` replaces them and is reproducible with the commands in the README.

## A fixed aggregation bug

An earlier `head_swap.py --summary` collected a stem's folds with the glob `hs_{stem}_f*.npz`,
which also matches a longer stem whose continuation begins with `f`. The label-efficiency
archives, `hs_s3_mlp_frac...`, were averaged into the unsupervised `s3_mlp` row by that path. The
shipped version matches the stem exactly, and `verify_heads.py` always did. The published numbers
come from `verify_heads.py` and are unaffected.

## The S14 lower-block re-run

`results/side_analyses.json` records a five-fold re-run of the lower block with
`improve/head_grad.py`, which replaces an earlier log and is what the table prints. The
class-activation-map, position-wise MLP, supervised MLP and Gaussian rows agree with the earlier
run to three decimals; the scale-and-shift and rank-4 rows changed.

That script looks for the attention reference as `hs_lowrank_f*.npz`, which the shipped root does
not hold, so it prints no agreement tag for the rank-4 row. Compared directly against
`hs_lowrank_maprelu_f*.npz`, which is what the root holds, it agrees on all five folds to within
5e-5, at insertion minus deletion of +0.0200, +0.0324, +0.0258, +0.0519 and +0.0115.

## The SkinCancer package

`code/SkinCancer/` is the data, configuration and model-layer library the rest of the code is
built on. `configs/` holds the data, model and training settings, `data/` the dataset, loaders,
split readers and transforms, and `eval/` the bootstrap and metric helpers. It contains no model
definition. The classifier is a timm backbone with its own seven-class head, built by
`improve/residual.py`, and each explanation head builds its own layers in
`improve/head_swap.py`.

An earlier version of this work compared a two-stage routing model against the single-stage
classifier. The paper reports none of it and nothing in the paper derives from it, so that
pipeline is not included. The `flat` checkpoints the explanation code loads hold the backbone and
its classifier only, `ResidualModel("flat")` adding nothing on top. `code/ablation/onestage.py`
trains the single-stage classifier and is what `improve/flat_seed.py` imports.

## Related and historical archive directories

These accompany the primary rows. A different seed, backbone, protocol or snapshot is not a
substitute for the row it resembles.

| context | stem | directory | folds | role | source |
| --- | --- | --- | --- | --- | --- |
| Table 1 | `s3_mlp_mapsoftmax` | `outputs/headswap/seed43`, `seed44` | 5 each | other readout seeds | `708b2a70b0` |
| Table 1 | `s3_mlp_mapsoftmax` | `outputs_convnext_tiny/headswap` | 5 | the second backbone | `5668415cc6` |
| Table 1 | `s3_mlp_mapsoftmax` | `outputs_nested/headswap` | 5 | the nested protocol | `5668415cc6` |
| Table 1 | `s3_mlp_sup0.2` | `outputs/headswap/seed43`, `seed44` | 5 each | other readout seeds | `708b2a70b0` |
| Table 1 | `s3_mlp_sup0.2` | `outputs_convnext_tiny/headswap` | 5 | the second backbone | `76f6cd468f` |
| Table 1 | `s3_mlp_sup0.2` | `outputs_nested/headswap` | 5 | the nested protocol | `76f6cd468f` |
| S10 pooled | `ssf_mapsoftmax` | `outputs/headswap_s43`, `_s44` | 5 each | other readout seeds | `5668415cc6` |
| S10 pooled | `lowrank_mapsoftmax` | `outputs/headswap_s43`, `_s44` | 5 each | other readout seeds | `5668415cc6` |
| S10 pooled | `lowrank_mapsoftmax` | `outputs/headswap/pre_retrain2` | 5 | snapshot before retraining | `5668415cc6` |
| S10 pooled | `cam_mapsoftmax` | `outputs/headswap_s43`, `_s44` | 5 each | other readout seeds | `5668415cc6` |
| S10 pooled | `mlp_mapsoftmax` | `outputs/headswap_s43`, `_s44` | 5 each | other readout seeds | `5668415cc6` |
| S10 pooled | `cond_mapsoftmax` | `outputs/headswap/seed43`, `seed44` | 5 each | other readout seeds | `5668415cc6` |
| S10 pooled | `lowrank_sup0.2` | `outputs/headswap/superseded` | 3 | incomplete earlier run | `-` |
| S10 pooled | `s3_mlp_nomacro` | `outputs_nested/headswap` | 5 | the nested protocol | `5668415cc6` |
| Table 3 | `s3_mlp_mapsoftmax`, `s3_mlp_nomacro`, `s3_mlp_sup0.2` | `outputs/headswap` | 5 each | the main protocol | as Table 1 |
