# Changes and removals

Record of what this revision removed from the repository, what it changed, and why.
Nothing is deleted silently.
Every item struck through below was present in commits `ccdf241` through `322770f` and is **not** in the current tree.

## Why anything was removed

Reviewer Comment 3 asked for a single identifiable implementation, with every reported result mapped to the code that produced it.

Between the original submission and this revision, a parallel re-analysis was built in `protocol/`: a pre-committed analysis plan, a runtime hold-out gate, and a re-run of the architecture sweep under that gate.
That work produced a **second, different** set of results, using a different architecture, a different nnU-Net fold and a different selection rule from the ones the manuscript reports.
Keeping both in one repository is the exact condition Comment 3 objects to: two implementations, no way to tell which produced the published numbers.

The repository has therefore been returned to the implementation that generated the submitted results, at commit `6475d07`.
The re-analysis is not deleted from history. It remains reachable at `322770f` and can be recovered with `git checkout 322770f` if the editor asks to see it.

## Removed: parallel locked re-analysis

This machinery implemented a different pipeline from the one reported. It produced no manuscript result.

- ~~`protocol/analysis_plan_v1.yaml`~~ - pre-committed plan. Declared internal-validation **AUC** as the selection criterion; the reported run used internal **accuracy** (`train.py:690`).
- ~~`protocol/gate.py`, `protocol/test_gate.py`, `protocol/gate_audit.jsonl`~~ - runtime hold-out guard on the external cohorts.
- ~~`protocol/run_protocol.py`, `protocol/__init__.py`~~ - phase runner for the gated re-analysis.
- ~~`protocol/locked_pipeline.json`, `protocol/locked_pipeline.commit`~~ - lock record. Selected **EfficientNet-B0**, not the reported architecture.
- ~~`protocol/DEVIATIONS.md`, `protocol/README.md`~~ - deviation log for that run.
- ~~`protocol/results/` (81 files)~~ - results of the gated re-analysis.
- ~~`protocol/runs/` (39 files)~~ - per-run training logs, 13 architectures x 3 seeds (67, 1234, 2025).
- ~~`protocol/finalize_rebuild.sh`, `protocol/watchdog.sh`, `protocol/reexport.log`~~ - orchestration.
- ~~`analysis/code/evaluate_frozen_pipeline.py`, `export_locked_predictions.py`, `calibrate_locked_model.py`~~ - downstream steps of the gated run.
- ~~`analysis/logs/train_locked_sweep.log`~~ - sweep console log.

## Removed: segmentation rebuild

A second nnU-Net was trained with an explicit train/validation split to give fold `all` a held-out signal.
The manuscript reports the fold `all` model, so the rebuild is not a reported result.
The fold `all` run at `Dataset000_lung` is untouched and remains the active segmentation model.

- ~~`seg-model-training/nnunet/prepare_dataset_with_validation.py`~~ - built `Dataset001_lungval` (600 train + 257 val).
- ~~`seg-model-training/nnunet/preprocess_with_original_plans.sh`~~ - reused the published plans.
- ~~`seg-model-training/nnunet/splits_final.json`~~ - explicit fold 0 split.
- ~~`seg-model-training/nnunet/predict_all_cohorts.sh`~~ - prediction driver for that model.
- ~~`nnUNet_results/Dataset001_lungval/`~~ - the fold 0 training run.
- ~~`seg-model-training/unet_train_with_validation.py`~~ - U-Net retrain with a real validation split.
- ~~`seg-model-training/unet_locked/train_log.txt`, `train_log_lr_collapsed.txt`, `train_log_lr_floor_undertrained.txt`~~ - logs from three retrain attempts, none reported.
- ~~`seg-model-training/deeplab_locked/predict.log`~~ - DeepLab re-inference log.

## Removed: revision-era documentation and audits

Superseded by [`REPRODUCE.md`](REPRODUCE.md), which is the single map Comment 3 asked for.

- ~~`docs/selection_audit/` (4 files)~~ - 13-architecture ranking audit. Its substance is retained: the same ranking is in `analysis/results/task5_architecture_ranking.csv`, which was part of the original deposit and is therefore better provenance.
- ~~`docs/MANUSCRIPT_HANDOFF.md`~~ - author checklist, superseded by the response letter.
- ~~`docs/CHANGELOG.md`~~ - changelog of the scope-correction work, superseded by this file.
- ~~`docs/history/` (12 files)~~ - snapshots of files edited during the revision.
- ~~`docs/provenance_manifest.json`~~ - hashes of revision-era artifacts.
- ~~`analysis/code/audit_existing_selection.py`, `write_provenance_manifest.py`~~ - generators of the two files above.
- ~~`analysis/code/task0b_cohort_overlap.py`, `analysis/results/task0b_cohort_overlap.{json,md}`~~ - cross-cohort overlap check. Result retained here: overlap with the external cohorts was **zero**; 11 duplicate images were found within the training cohort only.
- ~~`DISCLOSURE.md`, `docs/RESPONSE_LETTER_DRAFT.md`~~ - drafts, not part of the deposit.

## Removed: scratch test scripts

Ad-hoc scripts that produced no reported result. `test/` is now excluded by `.gitignore`.

- ~~`test/calculate_dice.py`~~
- ~~`test/run_binary_classification_auc.py`~~
- ~~`test/run_nnunet_predict_tmp.sh`~~
- ~~`test/binary_classification_predictions.csv`, `binary_classification_model_summary.csv`~~

## Changed: executable code

Every executable difference from `6475d07` is listed here, so that
`git diff 6475d07 HEAD -- '*.py' '*.sh'` can be checked against this table.
None of them affects a reported number.

| File | Change | Reason |
|---|---|---|
| `binary_classification/infer_probs_tight.py:28` | ~~`DATA_CROP = ROOT / "new_data"`~~ to `DATA_CROP = ROOT / "data"` | `new_data/` is not in the repository and never was, so the script could not run as deposited. The cohort images are under `data/`, whose layout matches the script's `SPLITS` table exactly: 600 / 257 / 108 / 94 cases. |
| `binary_classification/build_labels.py:28`, `check_data.py:36` | the same `new_data/` to `data/` correction | The same defect in two further scripts, missed when the first was fixed. `build_labels.py` writes the four label CSVs the classification stage reads, so it could not run either. |
| `analysis/code/seg_metrics_engine.py` | added an existence check on the nnU-Net mask directories, and a guard on the summary write | Without the cohort images the script previously wrote an all-`NaN` `seg_metrics_summary.json` over the deposited one and exited 0. It now reports that nothing was found and leaves the deposited results in place. |
| `analysis/code/run_all.sh:11` | `PY` default from a local virtualenv path to `python3` | The documented commands could not run outside the authors' machine. An explicit `PY=/path/to/python` still overrides it. |
| `analysis/code/run_all.sh:17-25` | added `subgroup_and_precision.py` and `summarize_training_logs.py` to the loop | Both are in the result-to-code map but no documented command ran them. Both read only deposited inputs and reproduce their artifacts byte-identically. |
| `seg-model-training/model_pipeline.sh` | corrected the commented baseline hyperparameters: U-Net epochs 450 to 350 and batch size 4 to 1; DeepLabv3+ total iterations 100000 to 30000 and batch size 8 to 4 | The values in this script disagreed with `README_SPL.md` in both trees. The deposited training logs settle it: `Pytorch-UNet/checkpoints/train_log.txt` ends at epoch 350 at LR 5e-4, and the retained segment of `DeepLabV3Plus-Pytorch/checkpoints/train_log.txt` runs 200 to 30000 on a 200-iteration validation cadence, which is 150 iterations per epoch over 600 slices and therefore batch size 4. These lines are commented out and were never executed by the driver. |
| `binary_classification/run_tight_pipeline.sh:7-9` | added the reason the training call is commented out | The script printed `[1/3] Training` and then ran nothing. |

Docstring-only changes, no executable effect: the interpreter in the usage
examples of five `analysis/code/` scripts, from a local virtualenv path to
`python3`, and the corrected data root in the `build_labels.py` and
`check_data.py` headers.

Added, not modified: `analysis/code/make_table2b.py`, which regenerates the
deposited `table2b_boundary_metrics.csv` and `.md` byte-identically from
`seg_metrics_per_case.csv` and `table2_paired_tests.csv`. Table 2b previously
had no producer in the repository.

Checkpoint selection (`train.py:690`, internal-validation accuracy), training
resolution (`train.py:46`, 300 px), inference resolution
(`infer_probs_tight.py:40`, 224 px) and the threshold rule
(`_best_threshold_from_rows`) are all exactly as submitted.

The per-epoch external AUC logging in `train.py:652-653` is **retained unmodified**.
It was left in place deliberately: the reviewer identified it, and removing it now would destroy the evidence rather than address the concern.
Its relationship to the checkpoint rule is documented in [`REPRODUCE.md`](REPRODUCE.md#training).

## Changed: documentation only

A pass reconciling the repository documentation against the manuscript and supplement found
several facts that were true of the code but stated nowhere, and several gaps that were not
declared. No executable file, result file, figure or environment pin was touched:
`git diff --stat` for this pass is confined to `README.md` and `docs/`.

| File | Added | Why |
|---|---|---|
| `README.md` | Baseline intensity scaling stated explicitly as per-image max with no percentile clipping, and the nnU-Net `CTNormalization` contrast | The supplement describes 1st/99th percentile clipping for the U-Net and DeepLabv3+ baselines. No such clipping exists in either path. Recording what the code does prevents the claim being read back out of this repository. |
| `README.md` | U-Net and DeepLabv3+ optimiser and scheduler, read from the two training scripts, plus LR, epochs/iterations, batch and crop | The supplement states that both baselines used Adam with a cosine annealing schedule. Neither does. U-Net uses `RMSprop` with `ReduceLROnPlateau` (`Pytorch-UNet/train.py:115`, `:117`); DeepLabv3+ uses `SGD` momentum 0.9 with `PolyLR` power 0.9 (`DeepLabV3Plus-Pytorch/main.py:321`, `:328`). Both are upstream defaults and were not modified. Only the two learning rates in the supplement are correct. |
| `README.md` | Dropout 0.4 before the classification head; first-convolution adaptation stated as average-then-copy | Both are in `train.py` and neither was documented. |
| `README.md` | Grad-CAM runs at 224, not at the 300 px training size | `heatmap.py:28`. The supplement states 300 x 300. |
| `README.md` | Implementation note 5 extended: the 39 sweep runs are 13 architectures x 3 seeds, no segmentation model among them | The supplement attributes part of the 39 to segmentation architectures. |
| `README.md` | Implementation note 6: the three selection steps do not share a mask variant | Checkpoint selection reads manual masks; ranking and threshold read automatic masks. All three read the Center 1 tuning cohort only. Worth stating precisely, because the distinction is the one a reviewer checks. |
| `README.md` | Implementation note 7: reported results are seed 67, not a per-architecture best seed | The supplement says results correspond to the best seed per architecture. Seed 67 is the best seed for 5 of 13. |
| `README.md` | "Not in this repository" table | The reader study, Table 1 statistics, Figure 3 and the interobserver Dice had no producer here and this was not declared anywhere. |
| `docs/REPRODUCE.md` | Grad-CAM entry and its resolution; note that `task4_dca.py` plots three cohorts, not four | Neither was in the result-to-code map. |
| `docs/REPRODUCE.md` | Known gaps 9 to 13 | Reader study, Table 1 statistics, Figure 3, Supplementary Table 4 External Test 2 column, interobserver Dice. |

Nothing in this pass changes a reported number. The items it records are inputs to the
manuscript corrections, not to the pipeline.

## Changed: `.gitignore`

Added, to prevent 4 GB of local working artifacts from entering the deposit. The files remain on the authors' machine and are not part of the reported pipeline.

```
protocol/
analysis/masks_locked_v2/
seg-model-training/nnunet/predicted_masks_v2/
seg-model-training/nnunet/nnUNet_results/Dataset001_lungval/
```

Removed one duplicate `_private_response/` entry, which appeared twice.

## Not removed

- All training logs from the submitted run, including the per-epoch external AUC values the reviewer identified: `analysis/logs/training_logs/`, `binary_classification/runs/<arch>/train_log.txt`.
- All deposited per-case probabilities: `binary_classification/predictions_tight/`.
- The nnU-Net fold `all` configuration, plans and training log.
- Every `analysis/results/*.csv` backing a reported table.
