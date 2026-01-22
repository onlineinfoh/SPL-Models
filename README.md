# SPL-Models
Code and repository prepared by Tianxi Liang.

Automatic Segmentation and Benign–Malignant Differentiation of Subpleural Pulmonary Lesions (SPLs) using grayscale ultrasound.  
Manuscript under review.

## Highlights
- Two-stage pipeline: segmentation → classification.
- Segmentation models: nnU-Net, U-Net, DeepLabv3.
- Classification models: DenseNet121/201, EfficientNet-b1/b2, ResNet18/50.
- Grad-CAM visualization for interpretability.

## Pipeline
1. **Data preparation**: ultrasound images + lesion masks; internal split and two external test sets.
2. **Segmentation training**: nnU-Net, U-Net, DeepLabv3, UNet++ on lesion masks.
3. **Segmentation evaluation**: Dice, mIoU, FPR, precision, recall.
4. **ROI extraction**: use best segmentation model to crop lesion region.
5. **Classification training**: DenseNet/EfficientNet/ResNet on ROI images.
6. **Classification evaluation**: AUC, sensitivity, specificity, PPV, NPV; Grad-CAM.

## Repository layout
- `seg-model-training/DeepLabV3Plus-Pytorch` – DeepLabv3 training code.
- `seg-model-training/Pytorch-UNet` – U-Net training code.
- `seg-model-training/nnunet` – nnU-Net training pipeline and configs.
- `binary_classification/train.py` – ROI-based classification training (all backbones).
- `binary_classification/heatmap.py` – Grad-CAM visualization.
- `binary_classification/runs/` – output folder (empty by default; created by training).
- `data/` – placeholder only (no clinical data).

## Quickstart
All paths below are relative to the repository root.

### Classification training
```bash
python binary_classification/train.py --arches densenet121 resnet50 --epochs 40 --batch-size 16
```
Outputs go to `binary_classification/runs/`.

### Grad-CAM
```bash
python binary_classification/heatmap.py
```
Outputs go to `binary_classification/heatmaps_tight/`.

## Grad-CAM method
We use Gradient-weighted Class Activation Mapping (Grad-CAM) on the last convolutional block of each classifier backbone to visualize discriminative regions. Heatmaps are overlaid on the cropped ROI and saved per cohort for qualitative inspection. The implementation is in `binary_classification/heatmap.py`.

## Segmentation training
Run the training scripts inside each segmentation folder. Detailed parameters and commands are documented in `README_SPL.md` for each model:
- **nnU-Net**: `seg-model-training/nnunet/README_SPL.md`
  - Training curve example: `seg-model-training/nnunet/nnUNet_results/Dataset000_lung/nnUNetTrainer__nnUNetPlans__2d/fold_all/progress.png`
- **U-Net**: `seg-model-training/Pytorch-UNet/README_SPL.md`
- **DeepLabv3**: `seg-model-training/DeepLabV3Plus-Pytorch/README_SPL.md`

## Data and weights
No clinical data or labels are included.
- Data/labels request: **lpbbl@aliyun.com** (Shanghai Pulmonary Hospital).
- Model weights request: **til4023@med.cornell.edu**.

## What is intentionally excluded
- Clinical data and labels.
- Checkpoints or model weights.
- Environment folders and local artifacts.

## Citation
Please cite the original methods:
- nnU-Net: Isensee et al., *Nature Methods*, 2021.
- U-Net: Ronneberger et al., *MICCAI*, 2015.
- DeepLabv3+: Chen et al., *ECCV*, 2018.
- DenseNet: Huang et al., *CVPR*, 2017.
- EfficientNet: Tan & Le, *ICML*, 2019.
- ResNet: He et al., *CVPR*, 2016.
- Grad-CAM: Selvaraju et al., *ICCV*, 2017.

## License
See individual submodules and `LICENSE` files.
