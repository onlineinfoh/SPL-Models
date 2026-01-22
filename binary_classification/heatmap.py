from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import nibabel as nib
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tv_models
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA_CROP = ROOT / "data"
LABEL_DIR = ROOT / "binary_classification" / "labels"
ARCHES = [
    "inception_v3", "vgg19",
    "resnet18", "resnet50", "resnet101",
    "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3", "efficientnet_b4", "efficientnet_b5",
    "densenet121", "densenet201",
]
SEEDS = [67]
OUT_ROOT = ROOT / "binary_classification" / "heatmaps_tight"
CKPT_ROOT = ROOT / "binary_classification" / "runs"

IMG_SIZE = 224
HALO_FRAC = 0.10

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


def load_predictions(arch: str, split: str, seed: int, variant: str = "gt") -> dict[str, float]:
    suffix = "" if variant == "gt" else f"_{variant}"
    pred_file = ROOT / "binary_classification" / "predictions_tight" / arch / f"{split}{suffix}_seed{seed}_probs.txt"
    if not pred_file.exists():
        return {}
    df = pd.read_csv(pred_file, comment="#")
    return {str(r.case_id): float(r.prob_malignant) for _, r in df.iterrows()}


def best_threshold_from_preds(labels: dict[str, int], preds: dict[str, float], prefer: float = 0.5):
    if not preds or not labels:
        return prefer, np.nan
    y_true = []
    y_prob = []
    for case, prob in preds.items():
        if case in labels and np.isfinite(prob):
            y_true.append(labels[case])
            y_prob.append(prob)
    if not y_true:
        return prefer, np.nan
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    probs = np.unique(y_prob)
    if probs.size == 0:
        return prefer, np.nan
    candidates = np.concatenate(([probs.min() - 1e-6], probs, [probs.max() + 1e-6, prefer]))
    best_thr = prefer
    best_acc = -np.inf
    for thr in candidates:
        preds_bin = (y_prob >= thr).astype(int)
        acc = (preds_bin == y_true).mean()
        if acc > best_acc + 1e-12 or (abs(acc - best_acc) < 1e-12 and abs(thr - prefer) < abs(best_thr - prefer)):
            best_acc = acc
            best_thr = thr
    return float(best_thr), float(best_acc)


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


def load_image_mask(img_path: Path, mask_path: Path):
    def _load(path):
        if path.suffix.lower() in {".nii", ".gz"} or path.name.endswith(".nii.gz"):
            return np.squeeze(nib.load(str(path)).get_fdata()).astype(np.float32)
        return np.array(Image.open(path)).astype(np.float32)

    full_img = _load(img_path)
    full_mask = _load(mask_path)
    if full_img.ndim == 2:
        full_img = full_img[..., None]
    elif full_img.ndim == 3 and full_img.shape[-1] > 1:
        full_img = full_img.mean(axis=-1, keepdims=True)
    if full_mask.ndim == 3:
        full_mask = full_mask[..., 0]
    if full_mask.max() > 1.0:
        full_mask = full_mask / 255.0
    full_mask = (full_mask > 0.5).astype(np.float32)

    x0, y0, x1, y1 = tight_crop(full_img[..., 0], full_mask)
    crop_img = full_img[y0:y1+1, x0:x1+1]
    crop_mask = full_mask[y0:y1+1, x0:x1+1]

    crop_img_resized = cv2.resize(crop_img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    crop_mask_resized = cv2.resize(crop_mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    crop_mask_resized = (crop_mask_resized > 0.5).astype(np.float32)
    if crop_img_resized.ndim == 2:
        crop_img_resized = crop_img_resized[..., None]

    x = np.concatenate([crop_img_resized, crop_mask_resized[..., None]], axis=-1)
    x = x.transpose(2, 0, 1)
    x = torch.from_numpy(x).float()
    x = (x - x.mean(dim=(1, 2), keepdim=True)) / (x.std(dim=(1, 2), keepdim=True) + 1e-6)
    return full_img[..., 0], full_mask, (x0, y0, x1, y1), crop_img_resized, crop_mask_resized, x.unsqueeze(0)


def iter_mask_variants(img_dir: Path, mask_dir: Path):
    """
    Yield (variant_name, mask_dir_path). Always includes ground truth ("gt").
    Adds a "model" variant if a sibling folder <img_dir.name>_model exists.
    """
    yield "gt", mask_dir
    model_dir = img_dir.parent / f"{img_dir.name}_model"
    if model_dir.exists():
        yield "model", model_dir


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


def build_model(arch: str, device: torch.device):
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
                return logits.squeeze(1), feat

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
                return logits.squeeze(1), feat

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
                return logits.squeeze(1), feat

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
                return logits.squeeze(1), feat

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
                return logits.squeeze(1), feat

        model = Model(base)
    elif arch == "vgg19":
        try:
            weights = tv_models.VGG19_Weights.IMAGENET1K_V1
        except Exception:
            weights = None
        base = tv_models.vgg19(weights=weights)
        base.features[0] = _adapt_first_conv(base.features[0])
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
                return logits.squeeze(1), flat

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
                return logits.squeeze(1), None

        model = Model(base)
    else:
        raise ValueError(f"Unsupported arch {arch}")

    return model.to(device)


def compute_cam(model, x, device):
    x = x.to(device)
    activations = []
    gradients = []

    def f_hook(module, inp, out):
        activations.append(out)

    def b_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

                                                              
    target = None
    for m in reversed(list(model.modules())):
        if isinstance(m, nn.Conv2d):
            target = m
            break
    if target is None:
        target = list(model.modules())[-1]

    handle_f = target.register_forward_hook(f_hook)
    handle_b = target.register_backward_hook(b_hook)

    logits, _ = model(x)
    score = logits[0]
    model.zero_grad()
    score.backward()

    handle_f.remove()
    handle_b.remove()

    if not activations or not gradients:
        return None
    act = activations[0]
    grad = gradients[0]
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = (weights * act).sum(dim=1)
    cam = cam.squeeze(0).detach().cpu().numpy()
    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam


def overlay_heatmap(full_img: np.ndarray, full_mask: np.ndarray, bbox, crop_cam: np.ndarray,
                    crop_mask: np.ndarray, crop_img: np.ndarray, title: str, subtitle: str, out_path: Path):
    x0, y0, x1, y1 = bbox
    full_h, full_w = full_img.shape[:2]
                                                                        
    full_norm = (full_img - full_img.min()) / (full_img.max() - full_img.min() + 1e-6)
    full_rgb = np.repeat(full_norm[..., None], 3, axis=2)
    contours, _ = cv2.findContours(full_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(full_rgb, contours, -1, (1.0, 0.0, 0.0), 2)
    cv2.rectangle(full_rgb, (x0, y0), (x1, y1), (1.0, 1.0, 0.0), 2)

                                                                                     
    cam_crop = cv2.resize(crop_cam, (x1 - x0 + 1, y1 - y0 + 1))
    cam_full = np.zeros((full_h, full_w), dtype=np.float32)
    cam_full[y0:y1+1, x0:x1+1] = cam_crop
    if cam_full.max() > 0:
        cam_full /= cam_full.max()
    cmap = plt.get_cmap("jet")
    cam_color_full = cmap(cam_full)[..., :3]
    overlay_full = 0.5 * full_rgb + 0.5 * cam_color_full

                                               
    cam_resized = cv2.resize(crop_cam, (crop_mask.shape[1], crop_mask.shape[0]))
    if cam_resized.max() > 0:
        cam_resized = cam_resized / cam_resized.max()
    cam_color = cmap(cam_resized)[..., :3]
                                                      
    crop_gray = np.squeeze(crop_img)
    if crop_gray.max() > crop_gray.min():
        base_gray = (crop_gray - crop_gray.min()) / (crop_gray.max() - crop_gray.min())
    else:
        base_gray = np.zeros_like(crop_gray)
    base_rgb = np.repeat(base_gray[..., None], 3, axis=2)
    overlay_crop = 0.5 * base_rgb + 0.5 * cam_color
    contours_crop, _ = cv2.findContours(crop_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay_crop, contours_crop, -1, (1.0, 0.0, 0.0), 2)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(overlay_full)
    axs[0].axis("off")
    axs[0].set_title("Full image + ROI heat")
    axs[1].imshow(overlay_crop)
    axs[1].axis("off")
    axs[1].set_title("Cropped heatmap")
    fig.suptitle(f"{title}\n{subtitle}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", format="jpeg")
    plt.close()
    print(f"Saved {out_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    for arch in ARCHES:
        for seed in SEEDS:
            ckpt = CKPT_ROOT / arch / "best.pth"
            if not ckpt.exists():
                print(f"Skipping {arch}: checkpoint not found.")
                continue
            model = build_model(arch, device)
            state = torch.load(ckpt, map_location=device)
            if "model_state_dict" in state:
                state = state["model_state_dict"]
            model.load_state_dict(state, strict=True)
            model.eval()

            val_img_dir, val_mask_dir, val_label_file = SPLITS["internal_val"]
            val_labels = load_labels(val_label_file)
            thresholds = {}
            for variant, _ in iter_mask_variants(val_img_dir, val_mask_dir):
                val_preds = load_predictions(arch, "internal_val", seed, variant=variant)
                thr, thr_acc = best_threshold_from_preds(val_labels, val_preds)
                thresholds[variant] = thr
                if np.isnan(thr_acc):
                    print(f"[WARN] {arch} seed {seed} {variant}: no val preds; using thr={thr:.3f}")
                else:
                    print(f"{arch} seed {seed} {variant}: best_thr={thr:.3f} (val acc={thr_acc:.4f})")

            for split, (img_dir, mask_dir, label_file) in SPLITS.items():
                labels = load_labels(label_file)
                for variant, mask_dir_variant in iter_mask_variants(img_dir, mask_dir):
                    preds = load_predictions(arch, split, seed, variant=variant)
                    img_paths = sorted(list(img_dir.glob("*.nii.gz")) + list(img_dir.glob("*.nii")) + list(img_dir.glob("*.png")))
                    if not img_paths:
                        print(f"No images for split {split}")
                        continue
                    suffix = "" if variant == "gt" else f"_{variant}"
                    out_dir = OUT_ROOT / arch / f"{split}{suffix}"
                    out_dir.mkdir(parents=True, exist_ok=True)

                    from tqdm import tqdm
                    for img_path in tqdm(img_paths, desc=f"{arch} {split}{suffix} heatmaps", leave=False):
                        case = normalize_case_prefix(img_path.name)
                        if case not in labels or case not in preds:
                            continue
                        mask_path = mask_dir_variant / img_path.name
                        if not mask_path.exists():
                            alt = mask_dir_variant / f"{case}.nii.gz"
                            if not alt.exists():
                                alt = mask_dir_variant / f"{case}.nii"
                            if alt.exists():
                                mask_path = alt
                        if not mask_path.exists():
                            digits = "".join(ch for ch in case if ch.isdigit())
                            if digits:
                                num = int(digits)
                                candidates = [
                                    mask_dir_variant / f"{num}_seg.nii.gz",
                                    mask_dir_variant / f"{num}_seg.nii",
                                    mask_dir_variant / f"{num:05d}_seg.nii.gz",
                                    mask_dir_variant / f"{num:05d}_seg.nii",
                                ]
                                for c in candidates:
                                    if c.exists():
                                        mask_path = c
                                        break
                        if not mask_path.exists():
                            continue

                        full_img, full_mask, bbox, crop_img, crop_mask, x = load_image_mask(img_path, mask_path)
                        cam = compute_cam(model, x, device)
                        if cam is None:
                            print(f"[WARN] CAM failed for {case} in {split}{suffix}")
                            continue
                        label = labels[case]
                        prob = preds.get(case, float("nan"))
                        thr = thresholds.get(variant, 0.5)
                        pred_cls = 1 if prob >= thr else 0
                        correct = (pred_cls == label)
                        subtitle = (
                            f"label={label} prob={prob:.3f} pred={pred_cls} thr={thr:.3f} "
                            f"({'correct' if correct else 'wrong'})"
                        )
                        out_path = out_dir / f"{case}_cam.jpg"
                        overlay_heatmap(full_img, full_mask, bbox, cam, crop_mask, crop_img, f"{arch} {split}{suffix} {case}", subtitle, out_path)


if __name__ == "__main__":
    main()
