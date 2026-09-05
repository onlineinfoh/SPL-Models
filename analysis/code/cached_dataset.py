"""
RAM-cached drop-in replacement for train.py's TightCropDataset.

The original dataset re-decodes two gzipped NIfTI volumes on every __getitem__.
With num_workers=0 that serialises disk I/O against GPU compute and holds GPU
utilisation near 16 percent. The pipeline is

    load -> tight bounding-box crop (10 percent halo) -> resize to 300x300
         -> [augmentation, training only] -> stack -> per-channel standardise

and everything up to and including the resize is deterministic and independent
of the epoch, so it is computed once and cached in RAM. That costs about 720 kB
per case, roughly 760 MB for all 1059 cases.

The augmentation path reproduces the original bit for bit, including a quirk:
cv2.warpAffine on a (H, W, 1) array returns (H, W), so after a rotation the
following `base = img[..., 0]` selects a single column rather than the image
plane. It is left as-is here so the two implementations stay interchangeable.
verify_cached_dataset.py asserts tensor-level equality between them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "binary_classification"))

from train import HALO_FRAC, IMG_SIZE, TightCropDataset  # type: ignore  # noqa: E402


class CachedTightCropDataset(Dataset):
    """Same semantics as TightCropDataset, with the deterministic prefix cached."""

    def __init__(self, image_dir: Path, mask_dir: Path,
                 labels: dict[str, int], is_train: bool = False):
        base = TightCropDataset(image_dir, mask_dir, labels, is_train=is_train)
        self.items = base.items
        self.is_train = is_train
        self._cache: list[tuple[np.ndarray, np.ndarray]] = []

        for img_path, mask_path, _label, _case in self.items:
            self._cache.append(self._deterministic_prefix(img_path, mask_path))

    # -- mirrors train.py lines 117-159 ------------------------------------
    @staticmethod
    def _load(path: Path) -> np.ndarray:
        import nibabel as nib
        from PIL import Image
        if path.suffix.lower() in {".nii", ".gz"} or path.name.endswith(".nii.gz"):
            return np.squeeze(nib.load(str(path)).get_fdata()).astype(np.float32)
        return np.array(Image.open(path)).astype(np.float32)

    @classmethod
    def _deterministic_prefix(cls, img_path: Path,
                              mask_path: Path) -> tuple[np.ndarray, np.ndarray]:
        img = cls._load(img_path)
        is_nii = (mask_path.suffix.lower() in {".nii", ".gz"}
                  or mask_path.name.endswith(".nii.gz"))
        mask = cls._load(mask_path) / (1.0 if is_nii else 255.0)

        if img.ndim == 2:
            img = img[..., None]
        elif img.ndim == 3 and img.shape[-1] > 1:
            img = img.mean(axis=-1, keepdims=True)
        if mask.ndim == 3:
            mask = mask[..., 0]
        mask = (mask > 0.5).astype(np.float32)

        ys, xs = np.where(mask > 0.5)
        if len(xs) == 0:
            # fallback: centred square crop. Not triggered by this dataset,
            # since every case has a non-empty mask.
            h, w = mask.shape
            size = min(h, w)
            y0 = (h - size) // 2
            x0 = (w - size) // 2
            y1, x1 = y0 + size - 1, x0 + size - 1
        else:
            x0, x1 = xs.min(), xs.max()
            y0, y1 = ys.min(), ys.max()
        bw, bh = x1 - x0 + 1, y1 - y0 + 1
        mx, my = int(HALO_FRAC * bw), int(HALO_FRAC * bh)
        x0 = max(0, x0 - mx)
        y0 = max(0, y0 - my)
        x1 = min(mask.shape[1] - 1, x1 + mx)
        y1 = min(mask.shape[0] - 1, y1 + my)

        img = img[y0:y1 + 1, x0:x1 + 1]
        mask = mask[y0:y1 + 1, x0:x1 + 1]

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0.5).astype(np.float32)
        if img.ndim == 2:
            img = img[..., None]
        return img.astype(np.float32), mask.astype(np.float32)

    def __len__(self) -> int:
        return len(self.items)

    # -- mirrors train.py lines 163-208 ------------------------------------
    def __getitem__(self, idx):
        _img_path, _mask_path, label, case = self.items[idx]
        cimg, cmask = self._cache[idx]
        img = cimg.copy()
        mask = cmask.copy()

        if self.is_train:
            if np.random.rand() < 0.5:
                img = np.flip(img, axis=1)
                mask = np.flip(mask, axis=1)
            if np.random.rand() < 0.5:
                angle = np.random.uniform(-20, 20)
                h, w_img = img.shape[:2]
                center = (w_img / 2, h / 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                img = cv2.warpAffine(img, M, (w_img, h), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REFLECT)
                mask = cv2.warpAffine(mask, M, (w_img, h), flags=cv2.INTER_NEAREST,
                                      borderMode=cv2.BORDER_REFLECT)
            # After a rotation img is (H, W), so img[..., 0] is a single column
            # rather than the image plane. Kept as in the original.
            base = img[..., 0]
            if np.random.rand() < 0.7:
                gamma = np.random.uniform(0.6, 1.4)
                base = np.clip(base ** gamma, 0.0, 255.0)
            if np.random.rand() < 0.7:
                factor = 1.0 + np.random.uniform(-0.25, 0.25)
                mean = base.mean()
                base = np.clip((base - mean) * factor + mean, 0.0, 255.0)
            if np.random.rand() < 0.5:
                noise = np.random.normal(0.0, 6.0, base.shape)
                base = np.clip(base + noise, 0.0, 255.0)
            img[..., 0] = base

        if img.ndim == 2:
            img = img[..., None]
        if mask.ndim == 3:
            mask = mask[..., 0]

        img = np.ascontiguousarray(img)
        mask = np.ascontiguousarray(mask)

        x = np.concatenate([img, mask[..., None]], axis=-1)
        x = x.transpose(2, 0, 1)
        x = torch.from_numpy(x).float()
        x = (x - x.mean(dim=(1, 2), keepdim=True)) / (x.std(dim=(1, 2), keepdim=True) + 1e-6)
        mask_tensor = torch.from_numpy(mask).float()
        return x, torch.tensor(label, dtype=torch.float32), mask_tensor, case


_DATASET_CACHE: dict[tuple[str, str, bool, int], CachedTightCropDataset] = {}


def build_cached_loader(img_dir: Path, mask_dir: Path, label_map: dict[str, int],
                        batch_size: int, is_train: bool):
    """
    Mirror of train.build_loader, using the cached dataset.

    The decoded dataset is memoised across calls so a multi-architecture,
    multi-seed sweep decodes each NIfTI exactly once per process.
    """
    key = (str(img_dir), str(mask_dir), is_train, len(label_map))
    ds = _DATASET_CACHE.get(key)
    if ds is None:
        ds = CachedTightCropDataset(img_dir, mask_dir, label_map, is_train=is_train)
        _DATASET_CACHE[key] = ds
    if len(ds) == 0:
        raise ValueError(f"No samples found for {img_dir}")
    if is_train:
        labels = np.array([lbl for _, _, lbl, _ in ds.items], dtype=np.int64)
        counts = np.bincount(labels, minlength=2)
        weights = 1.0 / (counts + 1e-6)
        sampler = WeightedRandomSampler(weights[labels], num_samples=len(labels),
                                        replacement=True)
        loader = DataLoader(ds, batch_size=batch_size, sampler=sampler,
                            num_workers=0, drop_last=False)
    else:
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, drop_last=False)
    return loader, ds
