# Cross-cohort overlap check

1059 images compared across 4 cohorts by image content.

## Why identifiers cannot be used

`case_id` is a within-cohort index. Every cohort numbers independently from `case_00001`, so identifiers collide completely and carry no patient information:

| Cohort pair | Colliding case_ids |
|---|---|
| train vs internal_val | 257 |
| train vs external_test1 | 108 |
| train vs external_test2 | 94 |
| internal_val vs external_test1 | 108 |
| internal_val vs external_test2 | 94 |
| external_test1 vs external_test2 | 94 |

## Cross-cohort separation (the leakage question)

- Exact duplicate pairs across cohorts: **0**
- Near-duplicate pairs across cohorts (lesion-crop r >= 0.99): **1**

**Verdict: SHARED IMAGES ACROSS COHORTS.**

## Within-cohort duplication (the cohort-size question)

Byte-identical images appearing more than once inside one cohort. This is not leakage, but it means the reported cohort size counts some images twice.

| Cohort | Images | Unique | Duplicated |
|---|---|---|---|
| train | 600 | 589 | 11 |
| internal_val | 257 | 257 | 0 |
| external_test1 | 108 | 108 | 0 |
| external_test2 | 94 | 94 | 0 |

## Limitation

Establishes image-level separation only. Patient-level separation cannot be established from this repository because no global patient identifier exists; the same patient could contribute different frames to different cohorts. That must be confirmed against the source records at the contributing centres.
