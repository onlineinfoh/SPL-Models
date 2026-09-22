# SPL-Models Notes (nnU-Net v2)

Original repository: https://github.com/MIC-DKFZ/nnUNet

## Data loading
Expected nnU-Net layout (relative to repo root):
- `seg-model-training/nnunet/nnUNet_raw/Dataset000_lung/imagesTr`
- `seg-model-training/nnunet/nnUNet_raw/Dataset000_lung/labelsTr`
- `seg-model-training/nnunet/nnUNet_raw/Dataset000_lung/dataset.json`

## Parameters used (paper run)
From `nnUNet_results/Dataset000_lung/nnUNetTrainer__nnUNetPlans__2d/plans.json`:
- configuration: `2d`
- batch size: 2
- patch size: 896×1792
- spacing: 1.0×1.0
- network: PlainConvUNet (nnU-Net default)

## Train
```bash
nnUNetv2_train Dataset000_lung 2d all -tr nnUNetTrainer -p nnUNetPlans
```

This documents the original run; no retraining was performed for the current
wording and inference corrections. The checkpoint used is `checkpoint_best.pth`.

## Meaning of Dice labels

~~Three-fold nnU-Net; internal-validation Dice during training~~

The original model uses **fold `all`**. nnU-Net sets `val_keys = tr_keys`
for this fold, so training-time "validation"/pseudo-Dice and the final
`fold_all/validation/` score are **training-set metrics**, not held-out internal
validation. Raw framework logs retain their original labels as historical records.

This does not rename the separate Center 1 tuning-cohort evaluation (`val`,
257 cases), nor either external-cohort evaluation (108 and 94 cases).
The deposited benchmark Dice is 0.9824 on train, 0.9203 on tuning,
0.9135 on External Test 1 and 0.9149 on External Test 2. Those saved values and
the original weights/masks are unchanged. Shared-metric Table 2 rounds the
external Dice to 0.913 and 0.915 and uses a different CI procedure.

~~Dataset001_lungval, fold 0, retrained pipeline as the reported model~~ is
superseded for the current scope. Its code, checkpoints and logs remain available;
it is a different experiment. See [result mapping](../../docs/REPRODUCE.md)
and [change record](../../docs/CHANGELOG.md).

## Outputs
Training curves/logs are under:
`nnUNet_results/Dataset000_lung/nnUNetTrainer__nnUNetPlans__2d/fold_all/`
