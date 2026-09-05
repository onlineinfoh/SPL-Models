#!/usr/bin/env python3
"""
Run inference with the tight ROI 2-channel ResNet-18 model and dump probabilities.

Outputs a TXT per split with: case_id, label, prob_malignant, prob_benign.
Splits: train, internal_val, external_test1, external_test2.
"""

from __future__ import annotations

from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tv_models
import nibabel as nib
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
DATA_CROP = ROOT / "new_data"
LABEL_DIR = ROOT / "binary_classification" / "labels"
ARCHES = [
    "inception_v3", "vgg19",
    "resnet18", "resnet50", "resnet101",
    "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3", "efficientnet_b4", "efficientnet_b5",
    "densenet121", "densenet201",
]
SEEDS = [67]
OUT_DIR = ROOT / "binary_classification" / "predictions_tight"
CKPT_ROOT = ROOT / "binary_classification" / "runs"

IMG_SIZE = 224
HALO_FRAC = 0.10  # match train.py

SPLITS = {
    "train": (
        DATA_CROP / "train" / "imagesTr",
        DATA_CROP / "train" / "labelsTr",
        LABEL_DIR / "labels_train.csv",
    ),
    "internal_val": (
        DATA_CROP / "val" / "img_v",
        DATA_CROP / "val" / "seg_v",
        LABEL_DIR / "labels_internal_val.csv",
    ),
    "external_test1": (
        DATA_CROP / "test1" / "img_test1",
        DATA_CROP / "test1" / "seg_test1",
        LABEL_DIR / "labels_external_test1.csv",
    ),
    "external_test2": (
        DATA_CROP / "test2" / "img_test2",
        DATA_CROP / "test2" / "seg_test2",
        LABEL_DIR / "labels_external_test2.csv",
    ),
}


def normalize_case_prefix(name: str) -> str:
    base = Path(name).name
    if base.endswith(".nii.gz"):
        base = base[:-7]
    else:
        base = Path(base).stem
    if base.endswith("_0000"):
        base = base[:-5]
    if base.startswith("case_"):
        return base
    digits = "".join(ch for ch in base if ch.isdigit())
    if digits:
        return f"case_{int(digits):05d}"
    return f"case_{base}"


def load_labels(path: Path) -> dict[str, int]:
    df = pd.read_csv(path)
    return {str(r.case_id): int(r.label) for _, r in df.iterrows()}


def tight_crop(img: np.ndarray, mask: np.ndarray):
    ys, xs = np.where(mask > 0.5)
    if len(xs) == 0:
        h, w = mask.shape
        size = min(h, w)
        y0 = (h - size) // 2
        x0 = (w - size) // 2
        y1 = y0 + size - 1
        x1 = x0 + size - 1
    else:
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    mx, my = int(HALO_FRAC * bw), int(HALO_FRAC * bh)
    x0 = max(0, x0 - mx)
    y0 = max(0, y0 - my)
    x1 = min(mask.shape[1] - 1, x1 + mx)
    y1 = min(mask.shape[0] - 1, y1 + my)
    return x0, y0, x1, y1


def load_image_mask(img_path: Path, mask_path: Path, mask_dir: Path):
    def _load(path):
        if path.suffix.lower() in {".nii", ".gz"} or path.name.endswith(".nii.gz"):
            return np.squeeze(nib.load(str(path)).get_fdata()).astype(np.float32)
        return np.array(Image.open(path)).astype(np.float32)

    img = _load(img_path)
    mask = None
    if mask_path.exists():
        mask = _load(mask_path)
    else:
        # try alternatives for test sets like 1_seg.nii.gz
        case = normalize_case_prefix(img_path.name)
        digits = "".join(ch for ch in case if ch.isdigit())
        if digits:
            num = int(digits)
            candidates = [
                mask_dir / f"{num}_seg.nii.gz",
                mask_dir / f"{num}_seg.nii",
                mask_dir / f"{num:05d}_seg.nii.gz",
                mask_dir / f"{num:05d}_seg.nii",
                mask_dir / f"{case}.nii.gz",
                mask_dir / f"{case}.nii",
            ]
            for c in candidates:
                if c.exists():
                    mask = _load(c)
                    break
    if mask is None:
        raise FileNotFoundError(f"Mask not found for {img_path}")
    if img.ndim == 2:
        img = img[..., None]
    elif img.ndim == 3 and img.shape[-1] > 1:
        img = img.mean(axis=-1, keepdims=True)
    mask = mask if mask.ndim == 2 else mask[..., 0]
    if mask.max() > 1.0:
        mask = mask / 255.0
    mask = (mask > 0.5).astype(np.float32)

    x0, y0, x1, y1 = tight_crop(img[..., 0], mask)
    img = img[y0:y1+1, x0:x1+1]
    mask = mask[y0:y1+1, x0:x1+1]

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 0.5).astype(np.float32)
    if img.ndim == 2:
        img = img[..., None]

    x = np.concatenate([img, mask[..., None]], axis=-1)
    x = x.transpose(2, 0, 1)
    x = torch.from_numpy(x).float()
    x = (x - x.mean(dim=(1, 2), keepdim=True)) / (x.std(dim=(1, 2), keepdim=True) + 1e-6)
    return x


def iter_mask_variants(img_dir: Path, mask_dir: Path):
    """
    Yield (variant_name, mask_dir_path). Always includes ground truth ("gt").
    Adds a "model" variant if a sibling folder <img_dir.name>_model exists.
    """
    yield "gt", mask_dir
    model_dir = img_dir.parent / f"{img_dir.name}_model"
    if model_dir.exists():
        yield "model", model_dir


def _sens_at_spec95(y_true: np.ndarray, y_prob: np.ndarray):
    if len(np.unique(y_true)) < 2:
        return np.nan
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    mask = fpr <= 0.05
    if not np.any(mask):
        return np.nan
    return np.max(tpr[mask])


def _metrics_from_rows(rows: list[tuple[str, int, float, float]], threshold: float = 0.5) -> dict[str, float]:
    if not rows:
        return {
            "acc": np.nan, "auc": np.nan, "ap": np.nan, "prec": np.nan,
            "recall": np.nan, "spec": np.nan, "sens_at_spec95": np.nan,
        }
    y_true = np.array([lbl for _, lbl, _, _ in rows])
    y_prob = np.array([p for _, _, p, _ in rows])
    preds = (y_prob >= threshold).astype(int)
    tp = np.sum((preds == 1) & (y_true == 1))
    tn = np.sum((preds == 0) & (y_true == 0))
    fp = np.sum((preds == 1) & (y_true == 0))
    fn = np.sum((preds == 0) & (y_true == 1))
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    spec = tn / (tn + fp + 1e-8)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    ap = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    sens95 = _sens_at_spec95(y_true, y_prob)
    return {
        "acc": acc,
        "auc": auc,
        "ap": ap,
        "prec": prec,
        "recall": rec,
        "spec": spec,
        "sens_at_spec95": sens95,
    }


def _best_threshold_from_rows(rows: list[tuple[str, int, float, float]], prefer: float = 0.5):
    if not rows:
        return prefer, np.nan
    y_true = np.array([lbl for _, lbl, _, _ in rows])
    y_prob = np.array([p for _, _, p, _ in rows])
    if y_true.size == 0:
        return prefer, np.nan
    probs = np.unique(y_prob)
    if probs.size == 0:
        return prefer, np.nan
    candidates = np.concatenate(([probs.min() - 1e-6], probs, [probs.max() + 1e-6, prefer]))
    best_thr = prefer
    best_acc = -np.inf
    for thr in candidates:
        preds = (y_prob >= thr).astype(int)
        acc = (preds == y_true).mean()
        if acc > best_acc + 1e-12 or (abs(acc - best_acc) < 1e-12 and abs(thr - prefer) < abs(best_thr - prefer)):
            best_acc = acc
            best_thr = thr
    return float(best_thr), float(best_acc)


def _adapt_first_conv(conv: nn.Conv2d) -> nn.Conv2d:
    new_conv = nn.Conv2d(2, conv.out_channels, kernel_size=conv.kernel_size,
                         stride=conv.stride, padding=conv.padding, bias=conv.bias is not None)
    with torch.no_grad():
        new_w = conv.weight.mean(dim=1, keepdim=True)
        new_conv.weight[:, 0:1, ...] = new_w
        new_conv.weight[:, 1:2, ...] = new_w
        if conv.bias is not None:
            new_conv.bias.copy_(conv.bias)
    return new_conv


def load_model(arch: str, checkpoint: Path, device: torch.device):
    arch = arch.lower()
    if arch == "resnet18":
        try:
            weights = tv_models.ResNet18_Weights.IMAGENET1K_V1
        except AttributeError:
            weights = None
        base = tv_models.resnet18(weights=weights)
        base.conv1 = _adapt_first_conv(base.conv1)

        class Model(nn.Module):
            def __init__(self, backbone):
                super().__init__()
                self.stem = nn.Sequential(
                    backbone.conv1,
                    backbone.bn1,
                    backbone.relu,
                    backbone.maxpool,
                    backbone.layer1,
                    backbone.layer2,
                    backbone.layer3,
                    backbone.layer4,
                )
                self.dropout = nn.Dropout(0.4)
                self.head = nn.Linear(backbone.fc.in_features, 1)

            def forward(self, x):
                feat = self.stem(x)
                feat = F.adaptive_avg_pool2d(feat, 1).squeeze(-1).squeeze(-1)
                feat = self.dropout(feat)
                logits = self.head(feat)
                return logits.squeeze(1)

        model = Model(base)
    elif arch.startswith("efficientnet_b"):
        try:
            weights = getattr(tv_models, f"EfficientNet_{arch.split('_')[1].upper()}_Weights").IMAGENET1K_V1
        except Exception:
            weights = None
        base = getattr(tv_models, arch)(weights=weights)
        base.features[0][0] = _adapt_first_conv(base.features[0][0])
        in_feats = base.classifier[1].in_features

        class Model(nn.Module):
            def __init__(self, backbone):
                super().__init__()
                self.features = backbone.features
                self.dropout = nn.Dropout(0.4)
                self.head = nn.Linear(in_feats, 1)

            def forward(self, x):
                feat = self.features(x)
                feat = F.adaptive_avg_pool2d(feat, 1).squeeze(-1).squeeze(-1)
                feat = self.dropout(feat)
                logits = self.head(feat)
                return logits.squeeze(1)

        model = Model(base)
    elif arch.startswith("densenet"):
        try:
            weights = getattr(tv_models, f"DenseNet{arch.split('densenet')[-1]}_Weights").IMAGENET1K_V1
        except Exception:
            weights = None
        base = getattr(tv_models, arch)(weights=weights)
        base.features.conv0 = _adapt_first_conv(base.features.conv0)
        in_feats = base.classifier.in_features

        class Model(nn.Module):
            def __init__(self, backbone):
                super().__init__()
                self.features = backbone.features
                self.dropout = nn.Dropout(0.4)
                self.head = nn.Linear(in_feats, 1)

            def forward(self, x):
                feat = self.features(x)
                feat = F.relu(feat, inplace=True)
                feat = F.adaptive_avg_pool2d(feat, 1).squeeze(-1).squeeze(-1)
                feat = self.dropout(feat)
                logits = self.head(feat)
                return logits.squeeze(1)

        model = Model(base)
    elif arch == "resnet50":
        try:
            weights = tv_models.ResNet50_Weights.IMAGENET1K_V1
        except Exception:
            weights = None
        base = tv_models.resnet50(weights=weights)
        base.conv1 = _adapt_first_conv(base.conv1)
        in_feats = base.fc.in_features

        class Model(nn.Module):
            def __init__(self, backbone):
                super().__init__()
                self.stem = nn.Sequential(
                    backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
                    backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4
                )
                self.dropout = nn.Dropout(0.4)
                self.head = nn.Linear(in_feats, 1)

            def forward(self, x):
                feat = self.stem(x)
                feat = F.adaptive_avg_pool2d(feat, 1).squeeze(-1).squeeze(-1)
                feat = self.dropout(feat)
                logits = self.head(feat)
                return logits.squeeze(1)

        model = Model(base)
    elif arch == "resnet101":
        try:
            weights = tv_models.ResNet101_Weights.IMAGENET1K_V1
        except Exception:
            weights = None
        base = tv_models.resnet101(weights=weights)
        base.conv1 = _adapt_first_conv(base.conv1)
        in_feats = base.fc.in_features

        class Model(nn.Module):
            def __init__(self, backbone):
                super().__init__()
                self.stem = nn.Sequential(
                    backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
                    backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4
                )
                self.dropout = nn.Dropout(0.4)
                self.head = nn.Linear(in_feats, 1)

            def forward(self, x):
                feat = self.stem(x)
                feat = F.adaptive_avg_pool2d(feat, 1).squeeze(-1).squeeze(-1)
                feat = self.dropout(feat)
                logits = self.head(feat)
                return logits.squeeze(1)

        model = Model(base)
    elif arch == "vgg19":
        try:
            weights = tv_models.VGG19_Weights.IMAGENET1K_V1
        except Exception:
            weights = None
        base = tv_models.vgg19(weights=weights)
        base.features[0] = _adapt_first_conv(base.features[0])
        in_feats = base.classifier[0].in_features  # typically 25088
        base.classifier = nn.Sequential(
            nn.Linear(in_feats, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 1),
        )

        class Model(nn.Module):
            def __init__(self, backbone):
                super().__init__()
                self.features = backbone.features
                self.avgpool = backbone.avgpool
                self.classifier = backbone.classifier

            def forward(self, x):
                feat = self.features(x)
                feat = self.avgpool(feat)
                flat = torch.flatten(feat, 1)
                logits = self.classifier(flat)
                return logits.squeeze(1)

        model = Model(base)
    elif arch == "inception_v3":
        try:
            weights = tv_models.Inception_V3_Weights.IMAGENET1K_V1
        except Exception:
            weights = None
        base = tv_models.inception_v3(weights=weights, aux_logits=True if weights is not None else False)
        base.aux_logits = False
        base.AuxLogits = None
        base.Conv2d_1a_3x3.conv = _adapt_first_conv(base.Conv2d_1a_3x3.conv)
        base._transform_input = lambda t: t
        in_feats = base.fc.in_features
        base.fc = nn.Linear(in_feats, 1)

        class Model(nn.Module):
            def __init__(self, backbone):
                super().__init__()
                self.backbone = backbone

            def forward(self, x):
                out = self.backbone(x)
                if hasattr(out, "logits"):
                    out = out.logits
                elif isinstance(out, (tuple, list)):
                    out = out[0]
                logits = out
                return logits.squeeze(1)

        model = Model(base)
    else:
        raise ValueError(f"Unsupported arch {arch}")

    state = torch.load(checkpoint, map_location=device)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def _collect_rows(model, img_dir: Path, mask_dir: Path, labels: dict[str, int], device: torch.device):
    rows = []
    missing = []
    img_paths = sorted(list(img_dir.glob("*.nii.gz")) + list(img_dir.glob("*.nii")) + list(img_dir.glob("*.png")))
    for img_path in img_paths:
        case = normalize_case_prefix(img_path.name)
        if case not in labels:
            missing.append(f"{case}: label missing")
            continue
        mask_path = mask_dir / img_path.name
        try:
            x = load_image_mask(img_path, mask_path, mask_dir).unsqueeze(0).to(device)
        except FileNotFoundError as e:
            missing.append(f"{case}: mask missing ({e})")
            continue
        with torch.no_grad():
            logits = model(x)
            prob = torch.sigmoid(logits).item()
        label = labels[case]
        rows.append((case, label, prob, 1 - prob))
    return rows, missing, img_paths


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = defaultdict(lambda: defaultdict(list))  # variant -> split -> list of (arch, seed, metrics, n, thr)

    for arch in ARCHES:
        for seed in SEEDS:
            ckpt = CKPT_ROOT / arch / "best.pth"
            if not ckpt.exists():
                print(f"Checkpoint not found for {arch}: {ckpt}")
                continue
            model = load_model(arch, ckpt, device)
            thresholds = {}

            # first, find best thresholds on internal_val
            split = "internal_val"
            img_dir, mask_dir, label_file = SPLITS[split]
            labels = load_labels(label_file)
            for variant, mask_dir_variant in iter_mask_variants(img_dir, mask_dir):
                rows, missing, img_paths = _collect_rows(model, img_dir, mask_dir_variant, labels, device)
                if not img_paths:
                    print(f"No images for split {split}")
                    continue
                if not rows:
                    print(f"No samples for {arch} {split} ({variant})")
                    continue
                thr, thr_acc = _best_threshold_from_rows(rows)
                thresholds[variant] = thr
                metrics = _metrics_from_rows(rows, threshold=thr)
                summary[variant][split].append((arch, seed, metrics, len(rows), thr))

                out_sub = OUT_DIR / arch
                out_sub.mkdir(parents=True, exist_ok=True)
                suffix = "" if variant == "gt" else f"_{variant}"
                out_path = out_sub / f"{split}{suffix}_seed{seed}_probs.txt"
                with out_path.open("w") as f:
                    f.write("case_id,label,prob_malignant,prob_benign\n")
                    for case, lbl, p_mal, p_ben in rows:
                        f.write(f"{case},{lbl},{p_mal:.6f},{p_ben:.6f}\n")
                    if missing:
                        f.write("\n# missing entries\n")
                        for m in missing:
                            f.write(f"{m}\n")
                print(
                    f"{arch} seed {seed} {variant}: best_thr={thr:.4f} (val acc={thr_acc:.4f}); "
                    f"saved {len(rows)} rows to {out_path}"
                )

            # run other splits using the internal_val threshold per variant
            for split in ("train", "external_test1", "external_test2"):
                img_dir, mask_dir, label_file = SPLITS[split]
                labels = load_labels(label_file)
                for variant, mask_dir_variant in iter_mask_variants(img_dir, mask_dir):
                    rows, missing, img_paths = _collect_rows(model, img_dir, mask_dir_variant, labels, device)
                    if not img_paths:
                        print(f"No images for split {split}")
                        continue
                    if not rows:
                        print(f"No samples for {arch} {split} ({variant})")
                        continue
                    thr = thresholds.get(variant, 0.5)
                    if variant not in thresholds:
                        print(f"Threshold for {arch} {variant} not found from internal_val; using 0.5")
                    metrics = _metrics_from_rows(rows, threshold=thr)
                    summary[variant][split].append((arch, seed, metrics, len(rows), thr))

                    out_sub = OUT_DIR / arch
                    out_sub.mkdir(parents=True, exist_ok=True)
                    suffix = "" if variant == "gt" else f"_{variant}"
                    out_path = out_sub / f"{split}{suffix}_seed{seed}_probs.txt"
                    with out_path.open("w") as f:
                        f.write("case_id,label,prob_malignant,prob_benign\n")
                        for case, lbl, p_mal, p_ben in rows:
                            f.write(f"{case},{lbl},{p_mal:.6f},{p_ben:.6f}\n")
                        if missing:
                            f.write("\n# missing entries\n")
                            for m in missing:
                                f.write(f"{m}\n")
                    print(
                        f"Saved {len(rows)} rows to {out_path} "
                        f"(thr={thr:.4f}, acc={metrics['acc']:.4f}, auc={metrics['auc']:.4f})"
                    )

    # write grand summary
    summary_path = OUT_DIR / "grand_summary.txt"
    with summary_path.open("w") as f:
        f.write("Grand summary of accuracies (threshold optimized on internal_val)\n")
        for variant in ("gt", "model"):
            if variant not in summary:
                continue
            f.write(f"\n=== {variant.upper()} masks ===\n")
            for split, records in summary[variant].items():
                if not records:
                    continue
                # sort by acc desc then arch name
                records_sorted = sorted(
                    records,
                    key=lambda x: (-(x[2]["acc"] if not np.isnan(x[2]["acc"]) else -1), x[0], x[1])
                )
                best = records_sorted[0]
                bm = best[2]
                f.write(
                    f"{split}: best {best[0]} (seed {best[1]}), "
                    f"thr={best[4]:.4f}, acc={bm['acc']:.4f}, auc={bm['auc']:.4f}, ap={bm['ap']:.4f}, "
                    f"prec={bm['prec']:.4f}, recall={bm['recall']:.4f}, spec={bm['spec']:.4f}, "
                    f"sens_at_spec95={bm['sens_at_spec95']:.4f}, n={best[3]}\n"
                )
                f.write("  all:\n")
                for arch, seed, m, n, thr in records_sorted:
                    f.write(
                        f"    {arch} (seed {seed}, thr={thr:.4f}): "
                        f"acc={m['acc']:.4f}, auc={m['auc']:.4f}, ap={m['ap']:.4f}, "
                        f"prec={m['prec']:.4f}, recall={m['recall']:.4f}, spec={m['spec']:.4f}, "
                        f"sens_at_spec95={m['sens_at_spec95']:.4f}, n={n}\n"
                    )
    print(f"Wrote grand summary to {summary_path}")


if __name__ == "__main__":
    main()
