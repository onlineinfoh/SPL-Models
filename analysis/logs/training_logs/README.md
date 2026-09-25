# Training logs

Per-run logs from the classification stage.

| Directory | Files | Run |
|---|---|---|
| `single_seed_67/` | 13 | One log per architecture, seed 67, from `binary_classification/train.py`. Records train, tuning and external metrics at every epoch, and the retained epoch. |
| `multiseed_sweep/` | 39 | One log per architecture and seed, seeds 67 / 1234 / 2025, from `analysis/code/train_locked.py`. Trains and selects on the tuning cohort only; external cohorts are evaluated once after selection and are not present in these logs. |

## Checkpoint selection

`analysis/code/summarize_training_logs.py` parses each log, recovers the full per-epoch series, and reports the epoch that was retained alongside the epoch that each candidate selection rule would have chosen.

```bash
python3 analysis/code/summarize_training_logs.py
```

Output: `analysis/results/checkpoint_selection_summary.csv` and `.json`.

For the seed-67 run, the retained epoch is the tuning-set accuracy maximum in all 13 architectures. It coincides with the External Test 1 AUC maximum in 3 of 13 and the External Test 2 AUC maximum in 3 of 13. Following the tuning-set rule gives up a mean of 0.0448 External Test 1 AUC and 0.0528 External Test 2 AUC relative to the best epoch available in each run.

## Early stopping

Across the 39 sweep runs the retained epoch has median 4 and range 1 to 12, and 13 of 39 runs retain the epoch 1 or epoch 2 checkpoint. Early stopping with patience 8 on 257 tuning cases is therefore sensitive to the seed, which is worth accounting for when comparing architectures.
