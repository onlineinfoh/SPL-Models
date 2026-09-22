# SPL-Models

Two-stage segmentation and classification of subpleural pulmonary lesions on grayscale ultrasound.

## Current scope

The active segmentation model is the original **nnU-Net Dataset000_lung, 2d,
fold `all`, checkpoint_best.pth**. The original classification checkpoints remain
in `binary_classification/runs/`. No retraining or inference was performed for
this code and documentation update.

Classification training uses 300×300 input. Inference and Grad-CAM now default
to **300×300**, correcting the previous 224×224 inference setting. The deposited
classification predictions were generated at 224×224; they have not been
relabelled as 300×300 results. Corrected inference will write separately to
`binary_classification/predictions_tight_300/`.

This is a restoration of the original model path with documented corrections,
not an exact checkout of commit `6475d07`. Later experiments and shared analysis
helpers are retained. No Git history, checkpoint, result or historical log was
removed by this update.

## Status of implementations

| Implementation | Status |
|---|---|
| Original nnU-Net `Dataset000_lung`, `fold_all` | Active segmentation checkpoint |
| `binary_classification/train.py` | Original training/calculation logic; future external-metric text logging suppressed |
| `binary_classification/infer_probs_tight.py` | Original checkpoint/threshold logic, corrected to 300 px; new output directory |
| `binary_classification/heatmap.py` | 300 px default, paired with new prediction directory |
| `analysis/code/` shared statistics | Historical outputs mapped below; no new calculations run |
| ~~Revision `Dataset001_lungval`, fold 0~~ | Retained alternative experiment, not the active segmentation model |
| ~~`protocol/run_protocol.py` as the reported pipeline~~ | Retained re-analysis; its EfficientNet-B0 results are a separate result set |
| ~~224 px inference as the current setting~~ | Historical code preserved in `docs/history/`; saved predictions remain intact |

## Reproducibility documents

- [Result-to-code/configuration map](docs/REPRODUCE.md)
- [Exact changes and preserved versions](docs/CHANGELOG.md)
- [Manuscript/Supplement checklist for colleagues](docs/MANUSCRIPT_HANDOFF.md)
- [Machine-readable file and checkpoint hashes](docs/provenance_manifest.json)
- [Historical revision code status](protocol/README.md)

## nnU-Net metric wording

~~Internal-validation Dice during `fold_all` training~~ → **training-set Dice**.
For `fold_all`, nnU-Net assigns its training cases to its validation loader too.
Its pseudo-Dice/validation output therefore does not measure held-out internal
validation. The raw logs keep their original framework labels.

The separate Center 1 tuning-cohort evaluation (257 cases) is a different
measurement and retains its tuning/internal-evaluation label. External Test 1
and External Test 2 contain 108 and 94 cases. Their original saved nnU-Net mean
Dice values remain **0.91347653** and **0.91491973**, respectively; neither masks,
weights nor metric files were modified.

## Classification configuration

Thirteen architectures: Inception-v3, VGG-19, ResNet-18/50/101, EfficientNet-B0–B5,
DenseNet121 and DenseNet201. The deposited original run uses seed **67**; the
separate multi-seed sweep uses **67, 1234, 2025**. The reported comparison is the **original seed-67 automatic-mask analysis**: DenseNet121
ranked first in internal accuracy and first on External Test 2 in accuracy and AUC,
but third on External Test 1 in both metrics. These are the saved **224 px**
inference results, not results of the pending 300 px correction. Full evidence:
[selection audit](docs/selection_audit/README.md).

Input: image and mask channels, lesion bounding box plus 10% halo, per-channel
standardisation. Training: up to 40 epochs, batch 16, AdamW (LR 1e-4, weight decay
5e-4), cosine schedule with minimum LR 1e-6, label smoothing 0.1, patience 8.
The original checkpoint rule is **internal-validation accuracy**, not AUC.
Inference selects each mask variant's threshold by maximum tuning accuracy.

Only future external-metric text output from training is suppressed. External
evaluation still runs and external plot curves remain; the computation and
checkpoint rule are unchanged. This is a logging change, not a hold-out change.

## Existing results versus corrected inference

`analysis/results/` and `binary_classification/predictions_tight/` contain the
original result set. `protocol/results/` contains the later re-analysis. Do not
combine them into a single table without identifying the different models.
The corrected 300 px probabilities, thresholds, classification tables and
heatmaps have **not** been generated. Historical DenseNet121 thresholds 0.5483
(manual masks) and 0.5000 (automatic masks) do not establish the new thresholds.

Centers 2 and 3 are described as **post-selection multicentre performance
evaluation** unless contemporaneous lock evidence or a genuinely untouched
cohort is available. An internal checkpoint rule alone does not establish the
history of architecture selection.

Patient data and weights are local resources; see [data/README.md](data/README.md).
The deposited analysis environment is [analysis/environment_lock.txt](analysis/environment_lock.txt).
The original nnU-Net debug record separately records torch 2.9.1+cu128; the later
environment must not be presented as the original run's environment.
