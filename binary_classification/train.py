from __future__ import annotations

import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np 
import pandas as pd
from PIL import Image
import nibabel as nib
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models as tv_models
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt


       
ROOT = Path(__file__).resolve().parents[1]
                          
DATA_CROP = ROOT / "data"
LABELS_DIR = ROOT / "binary_classification" / "labels"
RUN_ROOT = ROOT / "binary_classification" / "runs"

TRAIN_IMG_DIR = DATA_CROP / "train" / "imagesTr"
TRAIN_MASK_DIR = DATA_CROP / "train" / "labelsTr"
INT_IMG_DIR = DATA_CROP / "val" / "img_v"
INT_MASK_DIR = DATA_CROP / "val" / "seg_v"
EXT1_IMG_DIR = DATA_CROP / "test1" / "img_test1"
EXT1_MASK_DIR = DATA_CROP / "test1" / "seg_test1"
EXT2_IMG_DIR = DATA_CROP / "test2" / "img_test2"
EXT2_MASK_DIR = DATA_CROP / "test2" / "seg_test2"

LABEL_FILES = {
    "train": LABELS_DIR / "labels_train.csv",
    "internal": LABELS_DIR / "labels_internal_val.csv",
    "external1": LABELS_DIR / "labels_external_test1.csv",
    "external2": LABELS_DIR / "labels_external_test2.csv",
}

HALO_FRAC = 0.10                                        
IMG_SIZE = 300
LABEL_SMOOTH = 0.1


def load_labels(path: Path) -> dict[str, int]:
    df = pd.read_csv(path)
    return {str(r.case_id): int(r.label) for _, r in df.iterrows()}


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


class TightCropDataset(Dataset):
    def __init__(self, image_dir: Path, mask_dir: Path, labels: dict[str, int], is_train: bool = False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.labels = labels
        self.is_train = is_train
        self.items = []

        for img_path in sorted(list(self.image_dir.glob("*.nii.gz")) + list(self.image_dir.glob("*.nii")) + list(self.image_dir.glob("*.png"))):
            case = normalize_case_prefix(img_path.name)
            mask_path = self.mask_dir / img_path.name
            if not mask_path.exists():
                                                
                alt = self.mask_dir / (case + ".nii.gz")
                if alt.exists():
                    mask_path = alt
                else:
                    alt = self.mask_dir / (case + ".nii")
                    if alt.exists():
                        mask_path = alt
            if not mask_path.exists():
                                                      
                digits = "".join(ch for ch in case if ch.isdigit())
                if digits:
                    num = int(digits)
                    cand = [
                        self.mask_dir / f"{num}_seg.nii.gz",
                        self.mask_dir / f"{num}_seg.nii",
                        self.mask_dir / f"{num:05d}_seg.nii.gz",
                        self.mask_dir / f"{num:05d}_seg.nii",
                    ]
                    for c in cand:
                        if c.exists():
                            mask_path = c
                            break
            if not mask_path.exists():
                continue
            if case not in labels:
                continue
            self.items.append((img_path, mask_path, labels[case], case))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, mask_path, label, case = self.items[idx]
        def _load(path):
            if path.suffix.lower() in {".nii", ".gz"} or path.name.endswith(".nii.gz"):
                arr = np.squeeze(nib.load(str(path)).get_fdata()).astype(np.float32)
            else:
                arr = np.array(Image.open(path)).astype(np.float32)
            return arr

        img = _load(img_path)
        mask = _load(mask_path) / (255.0 if mask_path.suffix.lower() not in {".nii", ".gz"} and not mask_path.name.endswith(".nii.gz") else 1.0)
        if img.ndim == 2:
            img = img[..., None]
        elif img.ndim == 3 and img.shape[-1] > 1:
            img = img.mean(axis=-1, keepdims=True)
        if mask.ndim == 3:
            mask = mask[..., 0]
        mask = (mask > 0.5).astype(np.float32)

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

        img = img[y0:y1+1, x0:x1+1]
        mask = mask[y0:y1+1, x0:x1+1]

                               
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0.5).astype(np.float32)
        if img.ndim == 2:
            img = img[..., None]

        if self.is_train:
                  
            if np.random.rand() < 0.5:
                img = np.flip(img, axis=1)
                mask = np.flip(mask, axis=1)
                            
            if np.random.rand() < 0.5:
                angle = np.random.uniform(-20, 20)
                h, w_img = img.shape[:2]
                center = (w_img / 2, h / 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                img = cv2.warpAffine(img, M, (w_img, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                mask = cv2.warpAffine(mask, M, (w_img, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
                              
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


def build_loader(img_dir: Path, mask_dir: Path, label_map: dict[str, int], batch_size: int, is_train: bool):
    ds = TightCropDataset(img_dir, mask_dir, label_map, is_train=is_train)
    if len(ds) == 0:
        raise ValueError(f"No samples found for {img_dir} with labels ({len(label_map)} labels present). "
                         "Check that case IDs in the label CSV match image names.")
    if is_train:
        labels = np.array([lbl for _, _, lbl, _ in ds.items], dtype=np.int64)
        class_counts = np.bincount(labels, minlength=2)
        weights = 1.0 / (class_counts + 1e-6)
        sample_weights = weights[labels]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        loader = DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=0, drop_last=False)
    else:
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
    return loader, ds


def _sens_at_spec95(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return np.nan
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    mask = fpr <= 0.05
    if not np.any(mask):
        return np.nan
    return np.max(tpr[mask])


def _metrics_from_arrays(y_true: np.ndarray, y_prob: np.ndarray):
    if y_true.size == 0:
        return {
            "auc": np.nan, "ap": np.nan, "acc": np.nan, "prec": np.nan,
            "recall": np.nan, "spec": np.nan, "ppv": np.nan, "npv": np.nan,
            "sens_at_spec95": np.nan,
        }
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    ap = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    preds = (y_prob >= 0.5).astype(int)
    tp = np.sum((preds == 1) & (y_true == 1))
    tn = np.sum((preds == 0) & (y_true == 0))
    fp = np.sum((preds == 1) & (y_true == 0))
    fn = np.sum((preds == 0) & (y_true == 1))
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    spec = tn / (tn + fp + 1e-8)
    ppv = prec
    npv = tn / (tn + fn + 1e-8)
    sens95 = _sens_at_spec95(y_true, y_prob)
    return {
        "auc": auc,
        "ap": ap,
        "acc": acc,
        "prec": prec,
        "recall": rec,
        "spec": spec,
        "ppv": ppv,
        "npv": npv,
        "sens_at_spec95": sens95,
    }


def eval_model(model, loader, device, return_arrays: bool = False):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y, _, _ in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits)
            ys.append(y.cpu().numpy())
            ps.append(probs.cpu().numpy())
    if not ys:
        empty = _metrics_from_arrays(np.array([]), np.array([]))
        return (empty, (np.array([]), np.array([]))) if return_arrays else empty
    y_true = np.concatenate(ys)
    y_prob = np.concatenate(ps)
    metrics = _metrics_from_arrays(y_true, y_prob)
    if return_arrays:
        return metrics, (y_true, y_prob)
    return metrics


def plot_auc(history: dict[str, list[float]], out_path: Path):
    plt.figure(figsize=(8, 5))
    for key, vals in history.items():
        if not vals:
            continue
        plt.plot(range(1, len(vals) + 1), vals, label=key)
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_roc_curves(curves: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], out_path: Path):
    plt.figure(figsize=(6, 6))
    for name, trio in curves.items():
        if trio is None:
            continue
        fpr, tpr, _ = trio
        if fpr is None or tpr is None:
            continue
        plt.plot(fpr, tpr, label=name)
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("1 - Specificity (FPR)")
    plt.ylabel("Sensitivity (TPR)")
    plt.title("ROC Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def _adapt_first_conv(conv: nn.Conv2d, in_ch: int) -> nn.Conv2d:
    new_conv = nn.Conv2d(in_ch, conv.out_channels, kernel_size=conv.kernel_size,
                         stride=conv.stride, padding=conv.padding, bias=conv.bias is not None)
    with torch.no_grad():
        avg_w = conv.weight.mean(dim=1, keepdim=True)
        for c in range(in_ch):
            new_conv.weight[:, c:c+1, ...] = avg_w
        if conv.bias is not None:
            new_conv.bias.copy_(conv.bias)
    return new_conv


def build_model(arch: str):
    arch = arch.lower()
    if arch == "resnet18":
        try:
            weights = tv_models.ResNet18_Weights.IMAGENET1K_V1
        except AttributeError:
            weights = None
        base = tv_models.resnet18(weights=weights)
        base.conv1 = _adapt_first_conv(base.conv1, 2)

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

        return Model(base)

    if arch.startswith("efficientnet_b"):
        try:
            weights = getattr(tv_models, f"EfficientNet_{arch.split('_')[1].upper()}_Weights").IMAGENET1K_V1
        except Exception:
            weights = None
        base = getattr(tv_models, arch)(weights=weights)
        base.features[0][0] = _adapt_first_conv(base.features[0][0], 2)
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

        return Model(base)

    if arch.startswith("densenet"):
        try:
            weights = getattr(tv_models, f"DenseNet{arch.split('densenet')[-1]}_Weights").IMAGENET1K_V1
        except Exception:
            weights = None
        base = getattr(tv_models, arch)(weights=weights)
        base.features.conv0 = _adapt_first_conv(base.features.conv0, 2)
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

        return Model(base)

    if arch == "resnet50":
        try:
            weights = tv_models.ResNet50_Weights.IMAGENET1K_V1
        except Exception:
            weights = None
        base = tv_models.resnet50(weights=weights)
        base.conv1 = _adapt_first_conv(base.conv1, 2)
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

        return Model(base)

    if arch == "resnet101":
        try:
            weights = tv_models.ResNet101_Weights.IMAGENET1K_V1
        except Exception:
            weights = None
        base = tv_models.resnet101(weights=weights)
        base.conv1 = _adapt_first_conv(base.conv1, 2)
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

        return Model(base)

    if arch == "vgg19":
        try:
            weights = tv_models.VGG19_Weights.IMAGENET1K_V1
        except Exception:
            weights = None
        base = tv_models.vgg19(weights=weights)
                          
        base.features[0] = _adapt_first_conv(base.features[0], 2)
        in_feats = base.classifier[0].in_features                   
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

        return Model(base)

    if arch == "inception_v3":
        try:
            weights = tv_models.Inception_V3_Weights.IMAGENET1K_V1
        except Exception:
            weights = None
        base = tv_models.inception_v3(weights=weights, aux_logits=True if weights is not None else False)
                                                             
        base.aux_logits = False
        base.AuxLogits = None
        base.Conv2d_1a_3x3.conv = _adapt_first_conv(base.Conv2d_1a_3x3.conv, 2)
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

        return Model(base)

    raise ValueError(f"Unsupported arch: {arch}")


def main():
    parser = argparse.ArgumentParser(description="Tight ROI binary classification (image + mask channels).")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--arches", type=str, nargs="+",
                        default=[
                            "inception_v3", "vgg19",
                            "resnet18", "resnet50", "resnet101",
                            "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3", "efficientnet_b4", "efficientnet_b5",
                            "densenet121", "densenet201",
                        ],
                        help="Backbones to train (space-separated list)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[67])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                 
    lbl_train = load_labels(LABEL_FILES["train"])
    lbl_int = load_labels(LABEL_FILES["internal"])
    lbl_ext1 = load_labels(LABEL_FILES["external1"])
    lbl_ext2 = load_labels(LABEL_FILES["external2"])

                                                      
    train_map = {normalize_case_prefix(p.name): lbl_train[normalize_case_prefix(p.name)]
                 for p in TRAIN_IMG_DIR.glob("*.nii.gz") if normalize_case_prefix(p.name) in lbl_train}
    if len(train_map) == 0:
        raise ValueError("No train cases matched labels. Rebuild labels with build_labels.py and check file naming.")

    results = []
    for arch in args.arches:
        for seed in args.seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)

            seed_dir = RUN_ROOT / arch
            seed_dir.mkdir(parents=True, exist_ok=True)
            log_path = seed_dir / "train_log.txt"
            ckpt_path = seed_dir / "best.pth"

            train_loader, _ = build_loader(TRAIN_IMG_DIR, TRAIN_MASK_DIR, train_map, args.batch_size, is_train=True)
            val_loader, _ = build_loader(INT_IMG_DIR, INT_MASK_DIR, lbl_int, args.batch_size, is_train=False)
            ext1_loader, _ = build_loader(EXT1_IMG_DIR, EXT1_MASK_DIR, lbl_ext1, args.batch_size, is_train=False)
            ext2_loader, _ = build_loader(EXT2_IMG_DIR, EXT2_MASK_DIR, lbl_ext2, args.batch_size, is_train=False)

            model = build_model(arch).to(device)
            criterion = nn.BCEWithLogitsLoss(reduction="none")
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

            best_val_acc = -np.inf
            best_epoch = None
            history = defaultdict(list)
            patience_ctr = 0

                                                     
            last_train_pair = (np.array([]), np.array([]))
            last_val_pair = (np.array([]), np.array([]))
            last_ext1_pair = (np.array([]), np.array([]))
            last_ext2_pair = (np.array([]), np.array([]))

            with log_path.open("w") as logf:
                logf.write(f"=== {arch} Seed {seed} ===\n")
                for epoch in range(1, args.epochs + 1):
                    model.train()
                    ys_train, ps_train = [], []
                    for x, y, m, _ in train_loader:
                        x, y, m = x.to(device), y.to(device), m.to(device)
                        optimizer.zero_grad()
                        logits = model(x)
                                         
                        y_smooth = y * (1 - LABEL_SMOOTH) + 0.5 * LABEL_SMOOTH
                                                                                          
                        lesion_weight = (m.mean(dim=(1, 2)) * 0.9 + 0.1).clamp(min=0.1)
                        loss = (criterion(logits, y_smooth) * lesion_weight).mean()
                        loss.backward()
                        optimizer.step()
                        probs = torch.sigmoid(logits)
                        ys_train.append(y.detach().cpu().numpy())
                        ps_train.append(probs.detach().cpu().numpy())
                    scheduler.step()

                                   
                    y_train = np.concatenate(ys_train) if ys_train else np.array([])
                    p_train = np.concatenate(ps_train) if ps_train else np.array([])
                    train_metrics = _metrics_from_arrays(y_train, p_train)
                    train_auc = train_metrics["auc"]
                    last_train_pair = (y_train, p_train)

                    val_metrics, (y_val, p_val) = eval_model(model, val_loader, device, return_arrays=True)
                    ext1_metrics, (y_e1, p_e1) = eval_model(model, ext1_loader, device, return_arrays=True)
                    ext2_metrics, (y_e2, p_e2) = eval_model(model, ext2_loader, device, return_arrays=True)
                    last_val_pair = (y_val, p_val)
                    last_ext1_pair = (y_e1, p_e1)
                    last_ext2_pair = (y_e2, p_e2)

                    history["train_auc"].append(train_auc)
                    history["val_auc"].append(val_metrics["auc"])
                    history["ext1_auc"].append(ext1_metrics["auc"])
                    history["ext2_auc"].append(ext2_metrics["auc"])

                    log_line = (
                        f"[{arch} Seed {seed}] Epoch {epoch:02d} "
                        f"train_auc={train_auc:.4f} val_auc={val_metrics['auc']:.4f} "
                        f"ext1_auc={ext1_metrics['auc']:.4f} ext2_auc={ext2_metrics['auc']:.4f}"
                    )
                    print(log_line)
                    logf.write(log_line + "\n")
                    logf.write(
                        f"    train: acc={train_metrics['acc']:.4f} auc={train_metrics['auc']:.4f} "
                        f"ap={train_metrics['ap']:.4f} spec={train_metrics['spec']:.4f} "
                        f"ppv={train_metrics['ppv']:.4f} npv={train_metrics['npv']:.4f} "
                        f"sens_at_spec95={train_metrics['sens_at_spec95']:.4f}\n"
                    )
                    logf.write(
                        f"    val: acc={val_metrics['acc']:.4f} auc={val_metrics['auc']:.4f} "
                        f"ap={val_metrics['ap']:.4f} spec={val_metrics['spec']:.4f} "
                        f"ppv={val_metrics['ppv']:.4f} npv={val_metrics['npv']:.4f} "
                        f"sens_at_spec95={val_metrics['sens_at_spec95']:.4f}\n"
                    )
                    logf.write(
                        f"    ext1: acc={ext1_metrics['acc']:.4f} auc={ext1_metrics['auc']:.4f} "
                        f"ap={ext1_metrics['ap']:.4f} spec={ext1_metrics['spec']:.4f} "
                        f"ppv={ext1_metrics['ppv']:.4f} npv={ext1_metrics['npv']:.4f} "
                        f"sens_at_spec95={ext1_metrics['sens_at_spec95']:.4f}\n"
                    )
                    logf.write(
                        f"    ext2: acc={ext2_metrics['acc']:.4f} auc={ext2_metrics['auc']:.4f} "
                        f"ap={ext2_metrics['ap']:.4f} spec={ext2_metrics['spec']:.4f} "
                        f"ppv={ext2_metrics['ppv']:.4f} npv={ext2_metrics['npv']:.4f} "
                        f"sens_at_spec95={ext2_metrics['sens_at_spec95']:.4f}\n"
                    )
                    logf.flush()

                                                    
                    val_acc = val_metrics["acc"]
                    if not np.isnan(val_acc) and val_acc > best_val_acc + 1e-6:
                        best_val_acc = val_acc
                        best_epoch = epoch
                        patience_ctr = 0
                        torch.save(model.state_dict(), ckpt_path)
                    else:
                        patience_ctr += 1
                        if patience_ctr >= args.patience:
                            break
            results.append((arch, best_val_acc))

            best_metrics = {"train": None, "val": None, "ext1": None, "ext2": None}
            ckpt_ready = best_epoch is not None and ckpt_path.exists()
            if ckpt_ready:
                state = torch.load(ckpt_path, map_location=device)
                if "model_state_dict" in state:
                    state = state["model_state_dict"]
                model.load_state_dict(state, strict=True)
                train_eval_loader, _ = build_loader(
                    TRAIN_IMG_DIR, TRAIN_MASK_DIR, train_map, args.batch_size, is_train=False
                )
                best_metrics["train"] = eval_model(model, train_eval_loader, device)
                best_metrics["val"] = eval_model(model, val_loader, device)
                best_metrics["ext1"] = eval_model(model, ext1_loader, device)
                best_metrics["ext2"] = eval_model(model, ext2_loader, device)

                                                                                     
            def _fmt_summary(tag, metrics):
                if metrics is None:
                    return f"{tag}: no valid metrics"
                return (
                    f"{tag}: epoch={best_epoch} "
                    f"acc={metrics['acc']:.4f} auc={metrics['auc']:.4f} ap={metrics['ap']:.4f} "
                    f"spec={metrics['spec']:.4f} ppv={metrics['ppv']:.4f} npv={metrics['npv']:.4f} "
                    f"sens_at_spec95={metrics['sens_at_spec95']:.4f}"
                )

            if not ckpt_ready or not np.isfinite(best_val_acc):
                best_line = "best_epoch=None (no checkpoint saved)"
            else:
                best_line = f"best_epoch={best_epoch} val_acc={best_val_acc:.4f}"

            summary_lines = [
                "=== Best checkpoint (val acc) ===",
                best_line,
                _fmt_summary("train", best_metrics["train"]),
                _fmt_summary("internal_val", best_metrics["val"]),
                _fmt_summary("external_test1", best_metrics["ext1"]),
                _fmt_summary("external_test2", best_metrics["ext2"]),
            ]
            with (seed_dir / "train_log.txt").open("a") as logf:
                logf.write("\n".join(summary_lines) + "\n")
            for line in summary_lines:
                print(line)

                                           
            plot_auc({
                "train (train set)": history["train_auc"],
                "internal (val)": history["val_auc"],
                "external_test1": history["ext1_auc"],
                "external_test2": history["ext2_auc"],
            }, seed_dir / "auc.png")
                                                   
            from sklearn.metrics import roc_curve
            curves = {}
            def _roc(y, p):
                if y is None or p is None or len(np.unique(y)) < 2:
                    return None, None, None
                return roc_curve(y, p)
            fpr, tpr, thr = _roc(*last_train_pair); curves["train (train set)"] = (fpr, tpr, thr)
            fpr, tpr, thr = _roc(*last_val_pair); curves["internal (val)"] = (fpr, tpr, thr)
            fpr, tpr, thr = _roc(*last_ext1_pair); curves["external_test1"] = (fpr, tpr, thr)
            fpr, tpr, thr = _roc(*last_ext2_pair); curves["external_test2"] = (fpr, tpr, thr)
            plot_roc_curves(curves, seed_dir / "roc.png")


if __name__ == "__main__":
    main()
