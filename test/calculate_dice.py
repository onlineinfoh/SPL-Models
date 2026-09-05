#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.fileholders import FileHolder


def parse_args():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Calculate Dice scores between predicted NIfTI masks and ground-truth NIfTI masks."
    )
    parser.add_argument("--gt-dir", type=Path, default=script_dir / "gt")
    parser.add_argument("--pred-dir", type=Path, default=script_dir / "tmp_out")
    parser.add_argument("--csv", type=Path, default=None, help="Optional path to write per-case Dice CSV.")
    return parser.parse_args()


def load_mask(path: Path) -> np.ndarray:
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

    if array.ndim > 3:
        raise ValueError(f"Unsupported ndim={array.ndim} for {path}")
    return (array > 0).astype(np.uint8, copy=False).reshape(-1)


def dice_score(gt: np.ndarray, pred: np.ndarray) -> float:
    intersection = np.count_nonzero(gt & pred)
    gt_sum = np.count_nonzero(gt)
    pred_sum = np.count_nonzero(pred)
    denominator = gt_sum + pred_sum
    if denominator == 0:
        return 1.0
    return (2.0 * intersection) / denominator


def main():
    args = parse_args()
    gt_files = sorted(args.gt_dir.glob("*.nii.gz"))
    if not gt_files:
        raise FileNotFoundError(f"No ground-truth files found in {args.gt_dir}")

    rows = []
    total_intersection = 0
    total_gt = 0
    total_pred = 0
    missing_predictions = []
    shape_mismatches = []

    for gt_path in gt_files:
        pred_path = args.pred_dir / gt_path.name
        if not pred_path.exists():
            missing_predictions.append(gt_path.name)
            continue

        gt = load_mask(gt_path)
        pred = load_mask(pred_path)

        if gt.shape != pred.shape:
            shape_mismatches.append((gt_path.name, gt.shape[0], pred.shape[0]))
            continue

        score = dice_score(gt, pred)
        rows.append((gt_path.name, score))

        total_intersection += np.count_nonzero(gt & pred)
        total_gt += np.count_nonzero(gt)
        total_pred += np.count_nonzero(pred)

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["case", "dice"])
            writer.writerows(rows)

    print(f"GT dir:   {args.gt_dir}")
    print(f"Pred dir: {args.pred_dir}")
    print(f"Matched cases: {len(rows)}")
    print(f"Missing predictions: {len(missing_predictions)}")
    print(f"Shape mismatches: {len(shape_mismatches)}")

    if missing_predictions:
        print("First missing predictions:", ", ".join(missing_predictions[:10]))
    if shape_mismatches:
        preview = ", ".join(f"{name}({gt_len}!={pred_len})" for name, gt_len, pred_len in shape_mismatches[:10])
        print("First shape mismatches:", preview)

    if not rows:
        raise RuntimeError("No comparable cases found.")

    per_case_dice = np.array([score for _, score in rows], dtype=np.float64)
    print(f"Mean per-case Dice:   {per_case_dice.mean():.6f}")
    print(f"Median per-case Dice: {np.median(per_case_dice):.6f}")
    print(f"Min per-case Dice:    {per_case_dice.min():.6f}")
    print(f"Max per-case Dice:    {per_case_dice.max():.6f}")

    denominator = total_gt + total_pred
    overall_dice = 1.0 if denominator == 0 else (2.0 * total_intersection) / denominator
    print(f"Overall Dice:         {overall_dice:.6f}")


if __name__ == "__main__":
    main()
