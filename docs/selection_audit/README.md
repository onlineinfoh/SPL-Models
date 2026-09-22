# Audit of saved DenseNet121 selection evidence

This audit uses existing probabilities and the two saved 39-run sweeps only.
It does not search new seeds, retrain, select new checkpoints, or tune on external
outcomes. Full comparisons, including competitors and ties, are in
[`all_rankings.csv`](all_rankings.csv); inputs are hashed in `sources.json`.

## Original seed 67, 224 px inference

Ranks among all 13 architectures (1 means highest; ties share rank):

| Mask variant | Metric | Internal | External 1 | External 2 |
|---|---|---:|---:|---:|
| Manual | Accuracy | 1 | 5 | 2 |
| Manual | AUC | 3 | 3 | 1 |
| Automatic | Accuracy | 1 | 3 | 1 |
| Automatic | AUC | 2 | 3 | 1 |

The closest match to the recalled result is **seed 67, automatic masks,
accuracy**: 0.856031 internal, 0.814815 External 1, 0.819149 External 2.
External 1 is led by EfficientNet-B0 and B1 (both 0.842593), so DenseNet121 is
not first on both external cohorts. Its automatic-mask External 1 AUC is
0.889201; EfficientNet-B1 has 0.895512.

Supported wording for these stored results:

> In the original seed-67 automatic-mask analysis, DenseNet121 had the highest
> internal-validation accuracy and the highest External Test 2 accuracy and AUC
> among the 13 architectures; it did not lead on External Test 1.

This observation is not evidence of the historical model-selection timeline,
nor does it apply automatically after changing inference to 300 px.

## Saved sweeps: internal ranks

| Experiment | Seed | Accuracy rank | AUC rank |
|---|---:|---:|---:|
| Earlier accuracy sweep | 67 | 1 (tied with EfficientNet-B0) | 2 |
| Earlier accuracy sweep | 1234 | 4 | 4 |
| Earlier accuracy sweep | 2025 | 6 | 7 |
| Revision AUC sweep | 67 | 2 | 2 |
| Revision AUC sweep | 1234 | 6 | 7 |
| Revision AUC sweep | 2025 | 8 | 8 |

Accuracy definitions differ: the earlier sweep records accuracy at 0.5; the
revision table above uses its recorded optimal-threshold accuracy. AUC is
evaluated at the checkpoint retained by each experiment's own rule. The two
experiments must not be combined into one seed comparison.

The earlier sweep chose an architecture by mean internal accuracy over seeds,
not by choosing the most favourable seed. A seed-level tie therefore does not
establish that its declared rule selected DenseNet121.

Neither sweep deposits full all-architecture external predictions. Available
winner/robustness external exports belong to EfficientNet-B0. Missing DenseNet121
external comparisons are **unverified**, not evidence of first place.

No seed outside 67/1234/2025 was found in the saved sweep records or named
checkpoint/output directories inspected. An additional dated run can be audited
if provided. The saved evidence currently does not support “best internally and
in both external cohorts.”
