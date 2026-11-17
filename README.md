# Balanced Transduction with Multi-Scale Representations

Evaluation code for the paper *Balanced Transduction with Multi-Scale
Representations: Training-Free Few-Shot Classification under Support Label Noise
and Out-of-Distribution Queries*.

The method needs no training, no noise-rate prior and no out-of-distribution
(OOD) rate prior. It concatenates three depth-wise features of one frozen
backbone, assigns support and query rows under a per-class quota, re-estimates
class centres transductively, and decides from a calibrated test whether the
quota should be held or released.

## What reproduces what

| Artefact in the paper | Script | Output |
|---|---|---|
| Table 3, Table 4, Figure 2, Figure 3, Figure 4 | `suite_main.py` | `suite_output/<backbone>_main.json` |
| paired intervals, p-values in Section 4.2 to 4.4 | `suite_stats.py` | `suite_output/stats_suite.json` |
| Table 5 (backbones), Figure 5 | `stats_backbone.py` | `suite_output/*_stats.json` |
| Table 6 (label propagation, LaplacianShot) | `tp_baseline.py`, `lap_shot.py` | `suite_output/*.json` |
| Table 7, Table 8 (query composition, imbalance) | `imbalance_experiment.py` | `suite_output/imbalance_*.json` |
| Table 9, Section 4.8 (when the quota fails) | `imbalance_experiment.py` with `--alpha` | same file |
| Figure 6 (oracle vs random removal) | `oracle_probe.py`, `loo_probe.py` | `probe_output/*.json` |
| Figure 7 (sensitivity) | `sensitivity_experiment.py` | `suite_output/sensitivity_*.json` |
| Figures 1 to 7 | `make_figures_v2.py`, `make_ms_figures.py` | `figures/figE*.pdf` and `.png` |

The remaining scripts are the probes behind Sections 4.5 and 5.2: they are the
negative results that motivated the design (`loo_probe.py` for the detection
precision, `diagnostic.py` for the oracle bound, `ood_transductive.py` for the
assumption that OOD queries break transduction, `ms_check.py` for the
84x84 re-extraction that rules out resolution as an explanation).

## Environment

Python 3.11 or newer; a CUDA device is recommended only for feature
extraction. Install with

```
pip install -r requirements.txt
```

The experiments reported here were produced with Python 3.13, numpy 2.5,
scipy 1.18, torch 2.9 (cu128) and matplotlib 3.11 on a single 24 GB GPU. Feature
extraction dominates the runtime; the assignment and re-estimation steps are
CPU-light.

## Data

All three domains are public. Download them yourself; nothing is bundled.

| Domain | Source | Expected location |
|---|---|---|
| EuroSAT_RGB | https://doi.org/10.5281/zenodo.7711810 | `data/EuroSAT_RGB/` (class folders) |
| DTD | https://www.robots.ox.ac.uk/~vgg/data/dtd/ | `data/dtd/images/` (class folders) |
| miniImageNet | the 20-class test split of Ravi and Larochelle's split, distributed as parquet by `GATE-engine/mini_imagenet` on Hugging Face | see below |

`mini_prepare.py` turns the parquet shards into an `ImageFolder` tree:

```
python scripts/mini_prepare.py --src <parquet dir> --dst data/miniimagenet/test
```

`--resize 84` is the switch used for the resolution control in Section 4.6. The
native images are 84x84 and are upsampled to 224x224 for the backbone, which is
what the paper reports.

## Feature cache

The scripts expect one `.npz` per (backbone, domain). The names predate a
refactor and are kept so that the published numbers regenerate exactly:

```
multiscale_output/eurosat_multiscale.npz     ResNet-18, EuroSAT_RGB
multiscale_output/dtd_multiscale.npz         ResNet-18, DTD
backbone_output/miniimagenet_resnet18.npz    ResNet-18, miniImageNet
backbone_output/{eurosat,dtd,miniimagenet}_resnet50.npz
backbone_output/{eurosat,dtd,miniimagenet}_vit_b_16.npz
```

Regenerate them with

```
python scripts/mb_features.py --backbone resnet18 --domain EuroSAT_RGB
python scripts/mb_features.py --backbone resnet18 --domain DTD
python scripts/mb_features.py --backbone resnet50 --domain both
python scripts/mb_features.py --backbone vit_b_16 --domain both
python scripts/mini_prepare.py --src <parquet dir> --dst data/miniimagenet/test
python scripts/mb_features.py --backbone resnet18 --domain MINI
python scripts/mb_features.py --backbone resnet50 --domain MINI
python scripts/mb_features.py --backbone vit_b_16 --domain MINI
```

The full stack is about 1 GB. It is a deterministic function of public images
and public pretrained weights, which is why it is regenerated rather than
shipped.

## Running the ladder

```
python scripts/suite_main.py --backbone resnet18 --episodes 100 --tag main
python scripts/suite_stats.py
python scripts/imbalance_experiment.py --backbone resnet18 --alpha 0.3,1.0,inf
python scripts/lap_shot.py
python scripts/tp_baseline.py
python scripts/sinkhorn_baseline.py
python scripts/make_figures_v2.py
```

`--only <domain>` restricts a run to a single domain, and `--mini-file` points
at an alternative miniImageNet cache; both are how the per-domain rows and the
84x84 control were produced. `--help` on any script lists the switches.

Note that `make_figures_v2.py` reads `suite_output/resnet18_main.json` when the
module is imported, so run the pipeline above first; it will fail immediately on
a fresh clone, by design rather than silently drawing an empty figure.

Ten splits are used throughout, each with 100 episodes; every reported cell is
the mean over those splits, and paired comparisons use the per-split differences
(two-sided t interval, df = 9).

## Checks

Two scripts re-derive the numbers in the manuscript from the raw JSON, so a
typo cannot survive a rebuild:

```
python scripts/verify_v5_tables.py <manuscript.md>
python scripts/verify_claims_v5.py <manuscript.md>
```

They print one line per table cell and per numerical claim; any mismatch is a
non-zero exit.

## What is not here

The manuscript sources are not part of this repository. The 1 GB feature cache
is not shipped (regenerate it). The GPU hours behind the three backbones, three
domains and ten splits are not reproducible on a laptop; expect a few hours of
feature extraction and a few minutes per cell.

## Licence

MIT. See `LICENSE`.

## Citation

Xu, C. (2026). *Balanced Transduction with Multi-Scale Representations:
Training-Free Few-Shot Classification under Support Label Noise and
Out-of-Distribution Queries*. Manuscript under review.
