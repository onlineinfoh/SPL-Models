# SPL-Models Notes (DeepLabV3+)

Original repository: https://github.com/VainF/DeepLabV3Plus-Pytorch

## Data loading
We use the custom `lung` dataset defined in `datasets/lung_ct.py`.
Expected NIfTI layout (relative to repo root):
- `data/train/imagesTr`
- `data/train/labelsTr`

## Parameters used (paper run)
- model: `deeplabv3plus_mobilenet`
- dataset: `lung`
- num classes: 2
- crop size: 512
- batch size: 4
- output stride: 16
- validation interval: 200 iterations
- total iterations: 30000, which is 200 epochs over the 600 training slices
- learning rate: 0.01 (default, poly schedule)

`--total_itrs` must be passed explicitly; the script default is 67000.

## Train
```bash
python main.py --dataset lung \
  --lung_img_dir ../../data/train/imagesTr \
  --lung_mask_dir ../../data/train/labelsTr \
  --num_classes 2 \
  --crop_size 512 \
  --batch_size 4 \
  --val_interval 200 \
  --total_itrs 30000
```

## Outputs
Checkpoints are written to `checkpoints/` (excluded from git).
