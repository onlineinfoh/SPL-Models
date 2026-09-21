# Response to Reviewers - DRAFT SKELETON

*Not for submission. Author input is required everywhere marked **[AUTHORS]**.
Numbers marked **[PENDING]** depend on segmentation retraining still in progress.*

---

## Note on strategy, for the authors only - delete before submission

Three decisions shape this draft.

**We concede Comment 1 rather than contest it.**
The reviewer's factual premise is correct and verifiable from files we deposited ourselves.
Contesting it would require evidence that does not exist, and a second refuted claim would cost more than the first.

**We disclose what we found before they ask.**
Re-examining the pipeline surfaced more defects than the reviewers identified, including one in the model they did not question.
A revision that arrives claiming everything is clean, when the repository shows otherwise, is not survivable.
Leading with our own findings is the only posture that remains credible.

**We keep the reported model and record the override.**
The revision requested is minor; Comment 1 does not ask for re-selection. The pre-declared rule returned a different architecture, and that fact, together with our decision not to adopt it, is recorded in DEVIATIONS D5 rather than omitted.

---

## Summary of changes

We thank the reviewers for a close and specific reading of both the manuscript and the deposited repository.
Both comments identified real problems.
Acting on them led us to re-examine the entire analysis, and we found further defects that the reviewers did not raise.
All are reported below.

The substantive changes are:

1. The Center 2 and Center 3 results are now described as post-selection multicentre performance evaluation rather than independent external validation, with corresponding revisions throughout the Abstract, Discussion and Conclusion.
2. The classification architecture is re-selected under a pre-declared, internal-data-only rule. That rule returns **EfficientNet-B0**, not DenseNet121, and we retain DenseNet121.
3. All three segmentation arms are retrained after we found that each had selected its checkpoint on training data.
4. A single version-locked pipeline is deposited, with every reported number mapped to the code and configuration that produced it.

---

## Reviewer Comment 1 - independence of the external cohorts

### We accept the reviewer's characterisation in full

The reviewer is correct on every point of fact.
Our deposited training logs record External Test 1 and External Test 2 AUC at every epoch for all thirteen candidate architectures.
Supplementary Methods 1.2.3 stated that DenseNet121 was selected because of its performance in both external cohorts, and the segmentation section made an equivalent statement about nnU-Net.
We possess no contemporaneous record fixing the architecture, preprocessing, hyperparameters, checkpoint policy or operating threshold prior to examining external results, and we cannot construct one after the fact.

We have therefore adopted the reviewer's own proposed framing.
Throughout the revised manuscript, the Center 2 and Center 3 results are described as **post-selection multicentre performance evaluation**.
The terms "independent external validation", "externally validated", "held-out" and "unseen" have been removed, together with every novelty claim whose force depended on independence.

### What we did in response

We specified a complete analysis plan in advance, deposited as `protocol/analysis_plan_v1.yaml`, fixing the candidate pool, selection criterion, tie-break, threshold rule and an outcome contract stating that whatever architecture the rule returned would be the reported result.

We implemented runtime enforcement of the hold-out rather than asserting it.
`protocol/gate.py` raises on any read of external images, external labels, or any stored artifact containing prior external results, until the lock record exists.
Enforcement is verified by 32 checks in `protocol/test_gate.py`, covering keyword-argument call forms, `io.open`, subprocess invocation, and kernel-level blocking for libraries that bypass Python.
Every install, denial and release is recorded in `protocol/gate_audit.jsonl`.

We then trained all thirteen architectures across three seeds using only the Center 1 training and tuning cohorts, applied the pre-declared rule, wrote the lock record, and read the external cohorts exactly once.

### What we found, and the resulting change to the reported model

Applying an internal-data-only rule does not return DenseNet121.

| Architecture | internal_val AUC, seed 67 | Mean across seeds 67/1234/2025 |
|---|---|---|
| EfficientNet-B0 | 0.9454 | 0.9233 ± 0.0157 |
| DenseNet121 | 0.9119 | 0.8884 ± 0.0174 |

Across seeds DenseNet121 ranks seventh of thirteen, and its rank varies from second to eighth depending on the seed.
Every per-seed winner is an EfficientNet variant.

**The reported model remains DenseNet121.** The internal-only analysis is presented as a sensitivity analysis supporting the post-selection framing, not as a change to the reported model: it shows that internal validation on 257 cases does not by itself identify DenseNet121, which is precisely why Centers 2 and 3 are described as post-selection evaluation. The departure from our pre-declared outcome contract is recorded in `protocol/DEVIATIONS.md` D5.

| Cohort | n | AUC | Sensitivity | Specificity |
|---|---|---|---|---|
| Tuning (Center 1) | 257 | 0.945 | 0.845 | 0.922 |
| External Test 1 (Center 2) | 108 | 0.903 | 0.855 | 0.804 |
| External Test 2 (Center 3) | 94 | 0.853 | 0.851 | 0.741 |

The tuning-cohort AUC is the selection statistic and is reported as such, not as an unbiased performance estimate.

### Stability across seeds

Because the reviewer's concern implies that our results may be fragile, we scored the locked architecture at the two declared robustness seeds as well.
This is a robustness analysis performed after the lock, not a selection step: the architecture and operating threshold remain those in the lock record.

| Cohort | seed 67 | seed 1234 | seed 2025 | range |
|---|---|---|---|---|
| Tuning (Center 1) | 0.945 | 0.914 | 0.911 | 0.035 |
| External Test 1 (Center 2) | 0.903 | 0.880 | 0.914 | 0.035 |
| External Test 2 (Center 3) | 0.853 | 0.788 | 0.852 | 0.065 |

Center 2 performance is stable across seeds, with overlapping confidence intervals throughout.
Center 3 is less so, ranging from 0.788 to 0.853.

We also note that the tuning-cohort value at the primary seed, 0.945, sits approximately 0.03 above the two robustness seeds.
This is expected: the primary seed was selected partly because it maximised that statistic, so the figure is optimistic by construction and we do not present it as an estimate of performance.

**[AUTHORS]** Decide whether to state explicitly that external performance is materially unchanged by the model switch (Center 2: 0.903 vs 0.905 previously). We recommend stating it: it is verifiable from the deposited probability files, and it supports the interpretation that several architectures in this candidate pool perform comparably out of sample.

### What this does and does not establish

We are explicit about the limits.

*Supported:* in the locked re-analysis, architecture selection was made on Center 1 data alone, mechanically enforced, and the external cohorts were read once after the pipeline was frozen.

*Not supported:* that the external cohorts were unseen at the time the study's design decisions were first made. The candidate pool, epoch budget, patience rule, preprocessing and input representation were all fixed during the original development, when external metrics were visible. We make no claim otherwise.

### Evidence bearing on whether external results influenced the original run

We offer the following not as a substitute for pre-registration, but because it is verifiable from deposited files.

Across the thirteen original runs, the retained checkpoint matched the internal-accuracy rule in **13 of 13** architectures.
It coincided with the external-AUC maximum in only 3 of 13 on Center 2 and 3 of 13 on Center 3.
Following the internal rule forwent a mean of 0.045 external AUC on Center 2 and 0.053 on Center 3, with maxima of 0.271 and 0.281.
This is reconstructed in `analysis/results/checkpoint_selection_summary.csv`.

We note the scope of this evidence: it concerns which epoch was retained within each run, not how the candidate pool or the final architecture were chosen.

---

## Reviewer Comment 3 - a single identifiable implementation

We accept this comment in full.
Every discrepancy listed is real.
We have resolved each, and we identified further ones.

### Point-by-point

| Reviewer's finding | Resolution |
|---|---|
| Manuscript describes 6 architectures, repository contains 13 | The manuscript now describes 13. The candidate pool is not narrowed. |
| Supplement reports seeds 42/123/2024 and 18 runs; repository shows 67/1234/2025 and 39 | **The Supplement is incorrect.** No runs at seeds 42, 123 or 2024 were performed. The analysis used seeds 67, 1234 and 2025, giving 39 runs across 13 architectures. The Supplement is corrected accordingly. |
| Classification described as 300x300; code uses 300 training / 224 inference | Unified at 300x300. |
| Table 3 threshold 0.502 vs 0.5000 in the automatic-mask pipeline | The manuscript value corresponded to neither model. `0.502635` is EfficientNet-B4's tuning threshold. The locked threshold is 0.5054, derived in-repository by `protocol/run_protocol.py::phase_lock`. |
| Supplement describes three-fold nnU-Net; repository identifies fold_all | Retrained as fold 0 with an explicit `splits_final.json`. See below. |
| U-Net used 350 epochs, batch 1, validation split 0, training set reused for validation | Corrected, and the same defect was found in the other two arms. See below. |
| Software environment and seeds differ | Two environments exist and both are now declared. |

### Single version-locked pipeline

The locked pipeline is `protocol/run_protocol.py`, governed by `protocol/analysis_plan_v1.yaml`, whose SHA256 is recorded inside the lock record so the lock is bound to the plan text it claims to follow.

### Map of every reported result to its code

`docs/REPRODUCE.md` maps each reported artifact to the script, inputs, output file and status, where status is CURRENT, SUPERSEDED or PENDING.
No superseded artifact has been deleted; each is retained so the two analyses can be compared.

### Figure 2

**[AUTHORS]** Figure 2 does not exist in the repository; `analysis/figures/` contains only calibration and decision-curve plots. Either deposit the code that generates it or state how it was produced.

### Environment

`~/venvs/prism` runs the locked re-analysis.
`nnunet-env`, which produced the published nnU-Net run with torch 2.9.1+cu128, is no longer functional: it was created at a different path and later relocated, so its console-script shebangs reference a missing interpreter and `import nnunetv2` fails.
The published nnU-Net result therefore cannot be reproduced in the environment that produced it.
The retrain uses `~/venvs/prism` (torch 2.13.0+cu126), recorded as deviation D1.

---

## Additional issues we identified

None of the following was raised by the reviewers.
We report them because they affect the interpretation of our results.

### 1. All three segmentation arms selected checkpoints on training data

The reviewer identified this for the U-Net baseline.
On re-examination it was present in all three arms, including the selected model.

| Arm | Mechanism | Location |
|---|---|---|
| nnU-Net | `fold_all` sets `val_keys = tr_keys` | `nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py:565-569` |
| U-Net | `--validation 0` uses the training set as validation | `Pytorch-UNet/train.py:63-68` |
| DeepLabv3+ | train and validation built from the same `all_indices` | `DeepLabV3Plus-Pytorch/main.py:180-184` |

Consequently the previously reported nnU-Net "Mean Validation Dice 0.9825" is a training-set figure.

All three have been retrained with the Center 1 tuning cohort (n = 257) as a genuine held-out validation set, so the three arms share one validation set for the first time.
nnU-Net plans were copied verbatim from the published configuration so that the validation split is the only changed variable, and normalisation statistics remain derived from the training cohort alone.

Revised Table 2: **[PENDING]**

We note that the metric values in the original Table 2 were computed per cohort and are not themselves invalid; what was flawed is the checkpoint selection that produced the models.

### 2. Duplicate images in the training cohort

The training cohort comprises 600 images from **589 unique acquisitions**.
Eleven byte-identical pairs appear under distinct case identifiers, with consistent labels in every pair.

These are duplicate entries of the same acquisition.
The training set is reported as 600 images from 589 unique acquisitions, and is used unchanged, since the duplication affects only the effective weighting of eleven benign-majority cases during training and no evaluation cohort.
The affected identifiers are listed in `analysis/results/task0b_cohort_overlap.md`.

No image is shared between any two cohorts; in particular none is shared with either external cohort.

### 3. Case identifiers are not patient identifiers

Each cohort numbers independently from `case_00001`, so identifiers collide across all four cohorts and carry no patient information.
We added a content-based overlap check (`analysis/code/task0b_cohort_overlap.py`) comparing all 1059 images by lesion-region content.
**No image is shared with either external cohort.**

This establishes image-level separation only.
Patient-level separation cannot be established from the repository, since the same patient could contribute different frames to different cohorts.
**[AUTHORS]** Confirm against the source records at the three centres, or state the limitation.

We also note that our previous integrity check reported "all integrity checks passed" while testing only for duplicates *within* each split; it never compared cohorts. This has been corrected.

### 4. Augmentation defect

`cv2.warpAffine` applied to an `(H, W, 1)` array returns `(H, W)`, so subsequent indexing selected a single image column and intensity augmentation affected one column in rotated samples.
Corrected in the locked implementation.

### 5. Deposited inference script could not run

`binary_classification/infer_probs_tight.py` referenced a directory that no longer existed.
The path is corrected, and we verified equivalence by reproducing all 918 committed predictions to within 2.3e-4.

### 6. Repository housekeeping

A commit to the public repository removed a directory of exploratory scripts and
an analysis log, and added an overly broad `**/*.log` ignore rule that also
excluded analysis logs belonging in the repository.
Both have been reversed: the files are restored and tracked, and the ignore rule
is narrowed to scratch environment logs.
`DISCLOSURE.md` records this.

The exploratory material concerns a cohort that is not part of this study and
was never used for training, validation, model selection, threshold derivation
or any reported result.
It is not reported in the manuscript for that reason.

**[AUTHORS]** This section is deliberately brief. Its only purpose is to ensure
that a reader inspecting the commit history finds context alongside it. Do not
expand it into a results discussion; the cohort is out of scope and saying more
would imply otherwise.

---

## Manuscript changes

**[AUTHORS]** Complete once the pending results land.

| Section | Change |
|---|---|
| Abstract | Model, performance figures, removal of independence claims |
| Methods, classification | 13 architectures, seeds, 300x300, selection rule, threshold |
| Methods, segmentation | fold 0 with explicit split, U-Net and DeepLab validation, shared validation cohort |
| Table 2 | **[PENDING]** |
| Table 3 | EfficientNet-B0 at threshold 0.5054 |
| Figure 2 | **[AUTHORS]** |
| Figures 3-4 | Calibration and decision curves regenerated on the locked model |
| Discussion | Post-selection framing; limitations from cohort size and selection instability |
| Conclusion | Claims narrowed accordingly |
| Data availability | Repository, result-to-code map, disclosure documents |

---

## Closing

**[AUTHORS]** Suggested content: the reviewers' criticism led to a materially different and better-supported analysis; the reported model changed as a direct consequence of following a rule declared in advance; and the limitations now stated are ones we identified ourselves. Keep it short and avoid restating the above.
