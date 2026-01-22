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
nnUNetv2_train Dataset000_lung 2d nnUNetTrainer__nnUNetPlans__2d -f all
```

## Outputs
Training curves/logs are under:
`nnUNet_results/Dataset000_lung/nnUNetTrainer__nnUNetPlans__2d/fold_all/`
