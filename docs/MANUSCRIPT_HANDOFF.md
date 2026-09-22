# Checklist for the manuscript and Supplement authors

No manuscript or Supplement source was provided. The repository changes are
complete for the stated code edits; publication text and pending results still
need the following author decisions. Use [REPRODUCE.md](REPRODUCE.md) for the
file-level map and do not combine original and revision result sets.

| Current/previous wording | Required correction or question |
|---|---|
| ~~nnU-Net three-fold training~~ | Dataset000_lung, 2d, **fold all**, checkpoint_best.pth. The later rerun was Dataset001/fold 0. |
| ~~Internal-validation Dice during fold_all training~~ | **Training-set Dice** (pseudo-Dice where appropriate). Raw log labels remain unchanged. |
| Separate Center 1 tuning Dice | Keep separate: evaluated on 257 tuning cases; do not rename it training Dice. |
| External segmentation Dice | Original saved means: Center 2 0.91347653, Center 3 0.91491973. Wording correction changes neither value. Confirm which CI procedure the table uses. |
| ~~300×300 at inference for already reported predictions~~ | Historical outputs used 224×224 inference with 300×300 training. Current inference is corrected to 300×300, but no new outputs have been generated. Which result set will the revised classification table report? |
| ~~Table 3 threshold 0.502~~ | Deposited DenseNet121 operating points: 0.5483 manual / 0.5000 automatic. Verify table variant. Re-derive the tuning threshold if reporting new 300 px predictions. |
| ~~Six architectures, seeds 42/123/2024, 18 repeats~~ | Repository original: 13 architectures at seed 67. Separate sweeps: 13×3 at seeds 67/1234/2025. Identify which experiment each sentence describes. |
| ~~Original checkpoint selected by internal AUC~~ | Original train.py selected by **internal accuracy**. AUC was used in the later, separate protocol sweep. |
| ~~DenseNet121 was best at another seed~~ | Report the verified original **seed 67, automatic masks, 224 px inference** comparison below. No different seed is asserted. |
| ~~DenseNet121 was best internally and externally~~ | Highest internal **accuracy**; highest External Test 2 accuracy/AUC; **third** on External Test 1 accuracy/AUC. Internal AUC ranks second. Use the paragraph below. |
| ~~Independent external validation established by a rerun~~ | Without prior verifiable lock evidence or untouched data, use **post-selection multicentre performance evaluation** for Centers 2 and 3. |
| ~~All models retrained for this update~~ | No training or inference was run. This update changes logging, inference resolution and documentation. |
| Single software environment | Original nnU-Net debug record and later analysis environment differ; specify the environment belonging to each actual result. |
| U-Net validation split / procedure | Original notes: 350 epochs, batch 1, validation split 0, training reused for training-time validation. Do not substitute later retraining descriptions. |
| Figure 2 | Show nnU-Net fold_all → predicted mask → ROI+10% halo → 2-channel classifier. State the result-specific resolution and threshold; remove the separate fold_0/EfficientNet-B0 path if not reported. |

Questions for colleagues before finalising:

1. Which exact manuscript cells use `analysis/results/` and which, if any, use `protocol/results/`?
2. Will the revised classification results stay historical (224 px, described accurately), or await new 300 px inference, threshold selection, tables and figures?
3. Use seed 67 for the accepted comparison. If a different historical model-selection claim is added, provide dated evidence for that separate claim.
4. Are there original DeepLab weights elsewhere? The current same-named checkpoint is from the later retraining.
5. Where are Figure 2 and any reader-study/interobserver results? Those cannot be mapped from the supplied repository alone.

Suggested accurate nnU-Net wording:

> The nnU-Net model used Dataset000_lung, the 2D configuration and fold all.
> Because this fold reuses the training cases in the framework's validation
> loader, its training-time Dice is described as training-set Dice. Evaluation
> on the separate Center 1 tuning cohort and the two external cohorts is
> reported separately. This terminology correction does not change the saved
> external segmentation results.

For the resolution correction, use future/pending wording until outputs exist:

> Classification inference has been corrected from 224×224 to 300×300 to match
> training. The existing classification results were produced with 224×224
> inference; revised 300×300 results have not yet been generated.

## Approved classification finding for the Supplement / Results

> In the original seed-67 analysis using automatic nnU-Net masks and 224×224
> classification inference, DenseNet121 achieved the highest internal-validation
> accuracy among the 13 architectures (85.60%). On External Test 2, it also
> achieved the highest accuracy (81.91%) and AUC (0.865). On External Test 1,
> its accuracy was 81.48% and AUC was 0.889, ranking third on both metrics.
> The automatic-mask operating threshold was 0.5000, determined on the internal
> tuning cohort and applied unchanged to the external cohorts.

Internal-validation AUC was 0.906 (rank 2). This paragraph reports the observed
comparison; it does not establish when the architecture was selected or justify
calling the external cohorts independent. The original within-run checkpoint
rule used internal accuracy. Do not add “therefore selected before external
evaluation” without dated evidence supporting that sequence.

These are historical 224 px results. The code's new 300 px inference setting
must be described as a subsequent correction with results pending, or the table
must be recomputed before claiming 300 px generated these numbers. No manuscript
source was supplied, so this text is ready for the colleague to insert.
