#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
MPLCONFIGDIR = REPO_ROOT / ".tmp" / "matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))


def _candidate_cuda_lib_dirs() -> list[Path]:
    roots = []
    for raw in (os.environ.get("CONDA_PREFIX"), sys.prefix, str(REPO_ROOT / "nnunet-env")):
        if raw:
            roots.append(Path(raw))

    dirs: list[Path] = []
    seen = set()
    for root in roots:
        for pattern in (
            "lib/python*/site-packages/nvidia/cudnn/lib",
            "lib/python*/site-packages/nvidia/cublas/lib",
            "lib/python*/site-packages/nvidia/cuda_runtime/lib",
        ):
            for path in root.glob(pattern):
                resolved = path.resolve()
                if resolved.is_dir() and resolved not in seen:
                    seen.add(resolved)
                    dirs.append(resolved)
    return dirs


def _ensure_cuda_loader_paths():
    candidate_dirs = _candidate_cuda_lib_dirs()
    if not candidate_dirs:
        return

    current = [entry for entry in os.environ.get("LD_LIBRARY_PATH", "").split(":") if entry]
    missing = [str(path) for path in candidate_dirs if str(path) not in current]
    if not missing:
        return

    os.environ["LD_LIBRARY_PATH"] = ":".join(missing + current)
    if os.environ.get("BINARY_AUC_LD_REEXEC") == "1":
        return
    os.environ["BINARY_AUC_LD_REEXEC"] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], os.environ)


_ensure_cuda_loader_paths()

import cv2
import nibabel as nib
import numpy as np
import torch
from nibabel.fileholders import FileHolder
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from binary_classification.train import build_model


HALO_FRAC = 0.10
IMG_SIZE = 300


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run binary classification sweeps on test images + predicted segmentations and compute AUC."
    )
    parser.add_argument("--image-dir", type=Path, default=SCRIPT_DIR / "tmp_in")
    parser.add_argument("--mask-dir", type=Path, default=SCRIPT_DIR / "tmp_out")
    parser.add_argument("--arch", type=str, default=None, help="Optional single architecture override.")
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Optional single checkpoint override. If omitted, sweeps all best.pth files under binary_classification/runs.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-csv", type=Path, default=SCRIPT_DIR / "binary_classification_predictions.csv")
    parser.add_argument("--summary-csv", type=Path, default=SCRIPT_DIR / "binary_classification_model_summary.csv")
    return parser.parse_args()


def load_nifti(path: Path) -> np.ndarray:
    try:
        image = nib.load(str(path))
        array = np.asanyarray(image.dataobj)
    except nib.filebasedimages.ImageFileError:
        if path.suffixes[-2:] != [".nii", ".gz"]:
            raise
        with path.open("rb") as handle:
            file_map = {"image": FileHolder(fileobj=handle), "header": FileHolder(fileobj=handle)}
            image = nib.Nifti1Image.from_file_map(file_map)
            array = np.asanyarray(image.dataobj)
    return np.squeeze(array).astype(np.float32, copy=False)


def case_id_from_image_name(name: str) -> str:
    base = name
    if base.endswith(".nii.gz"):
        base = base[:-7]
    elif base.endswith(".nii"):
        base = base[:-4]
    if base.endswith("_0000"):
        base = base[:-5]
    return base


def label_from_case_id(case_id: str) -> int:
    if case_id.startswith("b_"):
        return 0
    if case_id.startswith("m_"):
        return 1
    raise ValueError(f"Unsupported case prefix for {case_id}")


class TightCropInferenceDataset(Dataset):
    def __init__(self, image_dir: Path, mask_dir: Path):
        self.items: list[tuple[Path, Path, int, str]] = []
        for image_path in sorted(list(image_dir.glob("*.nii.gz")) + list(image_dir.glob("*.nii"))):
            case_id = case_id_from_image_name(image_path.name)
            mask_candidates = [
                mask_dir / f"{case_id}.nii.gz",
                mask_dir / f"{case_id}.nii",
                mask_dir / image_path.name,
            ]
            mask_path = next((candidate for candidate in mask_candidates if candidate.exists()), None)
            if mask_path is None:
                continue
            self.items.append((image_path, mask_path, label_from_case_id(case_id), case_id))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        image_path, mask_path, label, case_id = self.items[idx]
        image = load_nifti(image_path)
        mask = load_nifti(mask_path)

        if image.ndim == 2:
            image = image[..., None]
        elif image.ndim == 3 and image.shape[-1] > 1:
            image = image.mean(axis=-1, keepdims=True)

        if mask.ndim == 3:
            mask = mask[..., 0]
        mask = (mask > 0.5).astype(np.float32)

        ys, xs = np.where(mask > 0.5)
        if len(xs) == 0:
            height, width = mask.shape
            size = min(height, width)
            y0 = (height - size) // 2
            x0 = (width - size) // 2
            y1 = y0 + size - 1
            x1 = x0 + size - 1
        else:
            x0, x1 = xs.min(), xs.max()
            y0, y1 = ys.min(), ys.max()

        box_width, box_height = x1 - x0 + 1, y1 - y0 + 1
        margin_x, margin_y = int(HALO_FRAC * box_width), int(HALO_FRAC * box_height)
        x0 = max(0, x0 - margin_x)
        y0 = max(0, y0 - margin_y)
        x1 = min(mask.shape[1] - 1, x1 + margin_x)
        y1 = min(mask.shape[0] - 1, y1 + margin_y)

        image = image[y0:y1 + 1, x0:x1 + 1]
        mask = mask[y0:y1 + 1, x0:x1 + 1]

        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0.5).astype(np.float32)
        if image.ndim == 2:
            image = image[..., None]

        image = np.ascontiguousarray(image)
        mask = np.ascontiguousarray(mask)
        x = np.concatenate([image, mask[..., None]], axis=-1).transpose(2, 0, 1)
        x = torch.from_numpy(x).float()
        x = (x - x.mean(dim=(1, 2), keepdim=True)) / (x.std(dim=(1, 2), keepdim=True) + 1e-6)
        y = torch.tensor(label, dtype=torch.float32)
        return x, y, case_id


def infer(model: torch.nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    case_ids: list[str] = []
    y_true: list[np.ndarray] = []
    y_prob: list[np.ndarray] = []
    with torch.no_grad():
        for x, y, case_batch in tqdm(loader, desc="Batches", leave=False):
            x = x.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()
            y_true.append(y.numpy())
            y_prob.append(probs)
            case_ids.extend(case_batch)
    return case_ids, np.concatenate(y_true), np.concatenate(y_prob)


def discover_model_specs(args) -> list[tuple[str, Path]]:
    if args.arch or args.weights:
        if not args.arch or not args.weights:
            raise ValueError("--arch and --weights must be provided together for single-model mode.")
        return [(args.arch, args.weights)]

    specs = []
    for weights_path in sorted((REPO_ROOT / "binary_classification" / "runs").glob("*/best.pth")):
        specs.append((weights_path.parent.name, weights_path))
    if not specs:
        raise RuntimeError("No best.pth checkpoints found under binary_classification/runs.")
    return specs


def main():
    args = parse_args()
    requested_device = args.device or os.environ.get("DEVICE")
    if requested_device:
        device = torch.device(requested_device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = TightCropInferenceDataset(args.image_dir, args.mask_dir)
    if len(dataset) == 0:
        raise RuntimeError("No matched image/mask pairs found.")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)
    model_specs = discover_model_specs(args)

    summary_rows = []
    best_result = None

    for arch, weights_path in tqdm(model_specs, desc="Models"):
        model = build_model(arch).to(device)
        state = torch.load(weights_path, map_location=device)
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state, strict=True)

        case_ids, y_true, y_prob = infer(model, loader, device)
        auc = roc_auc_score(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        y_pred = (y_prob >= 0.5).astype(np.int64)
        acc = float((y_pred == y_true).mean())

        result = {
            "arch": arch,
            "weights": weights_path,
            "case_ids": case_ids,
            "y_true": y_true,
            "y_prob": y_prob,
            "y_pred": y_pred,
            "auc": auc,
            "ap": ap,
            "acc": acc,
        }
        summary_rows.append(result)
        if best_result is None or result["auc"] > best_result["auc"]:
            best_result = result

    assert best_result is not None

    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["arch", "weights", "auc", "average_precision", "accuracy_at_0.5"])
        for row in summary_rows:
            writer.writerow([row["arch"], row["weights"], f"{row['auc']:.6f}", f"{row['ap']:.6f}", f"{row['acc']:.6f}"])

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case_id", "label", "prob_malignant", "pred_label"])
        for case_id, label, prob, pred in zip(
            best_result["case_ids"], best_result["y_true"], best_result["y_prob"], best_result["y_pred"]
        ):
            writer.writerow([case_id, int(label), float(prob), int(pred)])

    print(f"Image dir:         {args.image_dir}")
    print(f"Mask dir:          {args.mask_dir}")
    print(f"Device:            {device}")
    print(f"Matched cases:     {len(best_result['case_ids'])}")
    print(f"Models evaluated:  {len(summary_rows)}")
    print(f"Best architecture: {best_result['arch']}")
    print(f"Best weights:      {best_result['weights']}")
    print(f"Best AUC:          {best_result['auc']:.6f}")
    print(f"Best AP:           {best_result['ap']:.6f}")
    print(f"Best Acc@0.5:      {best_result['acc']:.6f}")
    print(f"Best CSV:          {args.output_csv}")
    print(f"Summary CSV:       {args.summary_csv}")


if __name__ == "__main__":
    main()
