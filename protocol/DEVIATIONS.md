# Deviations from the analysis plan

Every departure from [`analysis_plan_v1.yaml`](analysis_plan_v1.yaml), with the
reason and the date. Silent deviation invalidates the plan, so items are
recorded here whether or not they are consequential.

---

## D1. nnU-Net rebuild runs in a different software environment

**Date:** 2026-09-19
**Plan says:** a single environment, torch 2.13.0+cu126 (`environment` section).
**What happened:** the published nnU-Net run was produced in `nnunet-env` with
torch 2.9.1+cu128, recorded in
`nnUNet_results/Dataset000_lung/nnUNetTrainer__nnUNetPlans__2d/fold_all/debug.json`.
That environment is no longer functional: it was created at
`/home/tianxi-liang/TianxiLiang/research/china/nnunet-env` and later moved into
the repository, so its console scripts carry shebangs pointing at an interpreter
that does not exist, and `import nnunetv2` fails inside it.

**Consequence:** the rebuild uses the `prism` environment (torch 2.13.0+cu126).
The published nnU-Net result cannot be reproduced bit-for-bit, because the
environment that produced it cannot be instantiated. The plan's claim of a
single environment was wrong when written; the repository has always had two.

**Reported as:** the environment discordance the reviewer identified. Both
environments are now declared rather than one.

---

## D2. Segmentation validation split changed

**Date:** 2026-09-19
**Plan says:** nnU-Net 2d, fold `all` (`segmentation.models.nnunet.fold`).
**What happened:** fold `all` cannot satisfy the requirement immediately above
it in the same plan, that the checkpoint be selected on internal_val Dice. In
nnU-Net v2, fold `all` sets `val_keys = tr_keys`
(`nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py:565-569`), so the run has no
held-out validation. `fold_all/validation/` contains exactly the 600 training
cases and the reported "Mean Validation Dice: 0.9825" is training Dice.

**Consequence:** a new dataset `Dataset001_lungval` contains both cohorts
(600 train, 257 internal_val) with an explicit `splits_final.json`, trained as
fold 0. Stock nnU-Net, no custom trainer. `checkpoint_best.pth` now has a
genuinely held-out selection signal.

**Note:** internal_val case identifiers were renumbered by +1000 because
`case_00001` exists in both cohorts and refers to two different patients. The
mapping is in `nnUNet_raw/Dataset001_lungval/case_id_mapping.json`.

---

## D3. Original plans forced rather than re-planned

**Date:** 2026-09-19
**Plan says:** patch 896x1792, batch 2, CTNormalization.
**What happened:** adding 257 cases changes the dataset fingerprint, so
re-planning would shift those values.

**Consequence:** `nnUNetPlans.json` is copied verbatim from `Dataset000_lung`
with only `dataset_name` patched. Two reasons: it keeps the published
architecture configuration so the retrain differs from the published run in the
validation split alone, and it keeps
`foreground_intensity_properties_per_channel` derived from the 600 training
cases, where re-planning would have computed normalisation statistics over
train plus internal_val.

---

## D4. Primary selection criterion changed to AUC

**Date:** 2026-09-19
**Plan originally said:** accuracy on internal_val at each architecture's own
accuracy-maximising threshold.
**Changed to:** AUC on internal_val, for both checkpoint retention and
architecture ranking; ties broken by optimal-threshold accuracy, then
alphabetically.

**Reason:** matches the authors' written specification. The change was made
before the sweep that feeds the lock was run, and the earlier partial sweep
(9 runs under the accuracy rule) was discarded rather than carried forward.
The accuracy ranking is still computed and reported alongside the AUC ranking,
because the two disagree and that disagreement is itself a finding.

---

## D5. Claim of independent external validation withdrawn

**Date:** 2026-09-19
**Plan says** (`scope.does_not_establish`): the protocol governs the locked
re-analysis and does not retroactively render Centers 2 and 3 unseen.

**What was found:** the internal-only sweep does not identify DenseNet121.
Under the declared rule (argmax internal_val AUC at the primary seed, all 13
architectures) the selected model is **efficientnet_b0**, AUC 0.9454 against
DenseNet121's 0.9119. Averaged across seeds 67/1234/2025, DenseNet121 ranks
**seventh of thirteen** (0.8884 +- 0.0174 against efficientnet_b0's
0.9233 +- 0.0157), its rank varies from second to eighth by seed, and every
per-seed winner is an EfficientNet variant.

**Consequence, two parts.**

First, the reported classification model changes from DenseNet121 to
efficientnet_b0 (seed 67, epoch 4, threshold 0.5054 by Youden on internal_val).
This follows `classification.selection.outcome_contract`, which was declared
before the sweep ran and states that whatever the rule returns is the reported
result. External performance is close to the published figures: Center 2 AUC
0.903 against 0.905, Center 3 0.853 against 0.889.

Second, the claim of independent external validation is withdrawn regardless of
which model is reported. The Center 2 and Center 3 results are described as
post-selection multicentre performance evaluation, the reviewer's own term. No
re-run can alter this, because the limitation lies in the history of the data
relative to the hypothesis rather than in the code: the candidate pool,
hyperparameters and preprocessing were all fixed while external metrics were
visible.

**Superseded note:** an earlier draft of this entry described the ranking under
a mean-across-seeds accuracy rule, which placed DenseNet121 fifth and led to a
"cluster of indistinguishable architectures" reading. That rule was replaced by
the AUC rule in D4 before the lock, and under the AUC rule the leaders do
separate by slightly more than seed noise. The cluster framing is not used.

---

## D6. U-Net learning-rate schedule adjusted for the noisier validation signal

**Date:** 2026-09-20
**Plan says:** U-Net at the published settings, 350 epochs, batch 1, lr 5e-4.
**What happened:** the published run drove `ReduceLROnPlateau(patience=5)` with
*training* Dice, which rises smoothly. Once validation moved to a genuine
held-out cohort (D2), the same schedule saw a noisy signal on 257 cases,
decayed five times within 35 epochs and reached a learning rate of exactly
zero at epoch 35, before the model had converged. Best held-out Dice at that
point was 0.6194 at epoch 30, after which no further learning was possible.

**Two failed attempts, recorded because they affected results that were briefly
produced:**

*Attempt 1* kept the published `patience=5` but drove it with held-out Dice.
The learning rate reached exactly zero at epoch 35. Abandoned;
log retained at `unet_locked/train_log_lr_collapsed.txt`.

*Attempt 2* raised patience to 20 and added `min_lr=1e-6`, still driven by
held-out Dice. This completed 350 epochs but **239 of them ran at the 1e-6
floor**, so the model never converged. It produced a U-Net baseline of 0.661
internal Dice against the published 0.797, and 0.556 against 0.765 on Center 2.
Log retained at `unet_locked/train_log_lr_floor_undertrained.txt`.

Attempt 2 is the more serious error, because a weakened baseline biases the
three-model comparison **in favour of nnU-Net**, the selected model. Any Table 2
generated from it is invalid and must not be reported.

*Attempt 3* separated the two concerns: schedule driven by training **loss** at
the published `patience=5`, checkpoint still selected on held-out Dice. This
also failed. Training diverged at epoch 6 (loss 0.99 to 176.8), the schedule
decayed in response, and the run was pinned at the 1e-6 floor from epoch 22 with
held-out Dice oscillating between 0.23 and 0.61. Stopped at epoch 28. Log at
`unet_locked/train_log.txt`.

**Resolution: the U-Net arm is NOT retrained, and the published weights are
retained. The defect is inert for this arm, which can be shown from the
deposited log.**

The published run's learning rate reached exactly 0.000000 at **epoch 45 of 350**.
For the remaining **306 epochs the model was frozen**: training Dice ranged
0.7468 to 0.7587, standard deviation 0.00201, total drift 0.0119. Every candidate
checkpoint in that window is the same weights to within measurement noise.
Selecting on training Dice rather than held-out Dice therefore chooses from a
pool of identical models and cannot have changed the result.

Retraining is not merely unnecessary here, it is harmful. The published optimiser
configuration (RMSprop, momentum 0.999, learning rate 5e-4, batch size 1) is
unstable on this data: all three retrain attempts collapsed, by divergence or by
learning-rate decay, at different epochs. The published run collapsed too, at
epoch 45; it simply reached a good solution first. Reproducing that outcome is a
matter of optimisation luck, not method. A retrained baseline that is weaker
because we could not reproduce their optimisation would bias the three-model
comparison **in favour of nnU-Net, the selected model**, which is the one
direction that must be avoided.

**This justification does not generalise to the other arms, and is not applied to
them.** nnU-Net and DeepLabv3+ both had live schedules for the whole run, so for
those the defect was real and both were retrained with a genuine held-out split.
nnU-Net's reported Dice falls by at most 0.011 on any evaluation cohort;
DeepLabv3+ falls by 0.034 to 0.065.

Table 2 therefore combines retrained nnU-Net, retrained DeepLabv3+, and the
published U-Net, with the source stated per row. The conclusion is unchanged:
nnU-Net leads every cohort by a wide margin.

Optimiser, learning rate, batch size and epoch budget were never altered in any
attempt.

**Note:** the original conflation was ours, not the reviewer's. The defect was
which checkpoint is *kept*; moving the learning-rate schedule onto the held-out
signal as well was an unforced change that degraded the baseline.

**DeepLabv3+ was checked for the same failure mode and is unaffected:** it uses
`PolyLR`, a fixed polynomial decay over `total_itrs` with no dependence on any
validation signal, so its schedule is identical to the published run. Its
change in Table 2 is attributable to honest checkpoint selection alone.

---

## D7. Post-hoc probability calibration added

**Date:** 2026-09-20
**Plan says:** nothing. The plan fixed a selection rule and an operating
threshold but did not anticipate that the selected model's probabilities would
need rescaling.

**What happened:** the locked model discriminates well (tuning AUC 0.945) but
its probabilities are compressed. Calibration slope on the tuning cohort was
3.46 against an ideal of 1.0, with 2.23 and 1.64 on Centers 2 and 3. The cause
is the retained checkpoint being epoch 4 of 40: AUC-based early stopping
selected a model that ranks cases well before its logits had spread out, so
discrimination and calibration decoupled. Decision-curve analysis assumes
calibrated probabilities, and a reported malignancy probability that is not a
probability cannot support clinical interpretation.

**Consequence:** Platt scaling (`analysis/code/calibrate_locked_model.py`),
fitted on the tuning cohort ONLY and applied unchanged elsewhere, the same
discipline already used for the operating threshold. Platt rather than isotonic
because 257 cases is too few for a non-parametric fit. Each mask variant is
calibrated separately. The locked operating threshold is mapped through the same
monotonic transform rather than re-optimised, so AUC and every classification
decision are unchanged; only the probability scale moves. Both are verified
per cohort in `predictions_calibrated/calibration.json`.

**Outcome, stated plainly: this does not fully work.** The transform fixes the
tuning cohort exactly, which is trivial because that is where it was fitted, and
overcorrects everywhere else, turning slopes above 1 into slopes below 1:

| Cohort | slope before | slope after |
|---|---|---|
| train | 1.48 | 0.43 |
| internal_val | 3.46 | 1.00 |
| external_test1 | 2.23 | 0.64 |
| external_test2 | 1.64 | 0.47 |

Measured as distance from ideal on the log scale, Center 2 improves
(0.80 to 0.45) while Center 3 worsens (0.49 to 0.76). Brier follows the same
mixed pattern. The degree of probability compression therefore differs between
cohorts, and a single scaling fitted at Center 1 cannot correct all three.

**Reported as:** both scales are reported. Raw probabilities remain primary for
discrimination, which calibration cannot affect. Calibrated probabilities are
used for decision-curve analysis, which requires them, with the limitation
stated. The tuning-cohort slope of exactly 1.00 is a fitting artifact and is
labelled as such rather than presented as a result.
