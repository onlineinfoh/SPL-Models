#!/usr/bin/env python3
"""
Quick sanity checker for the new data at data/.

What it does:
- Walks all subdirectories under data/
- Counts image files (.nii/.nii.gz/.png/.jpg/.jpeg)
- Prints a few sample shapes per folder (nifti via nibabel; png/jpg via PIL)
- Lists any label tables (.csv/.xlsx/.json) with their columns and first 3 rows
"""

from __future__ import annotations

from pathlib import Path
import json
import csv
import sys

try:
    import nibabel as nib
except ImportError:
    nib = None

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import pandas as pd
except ImportError:
    pd = None


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"


IMG_EXTS = {".nii", ".nii.gz", ".png", ".jpg", ".jpeg"}
TABLE_EXTS = {".csv", ".xlsx", ".xls", ".json"}


def list_files():
    files = []
    for p in DATA_ROOT.rglob("*"):
        if p.is_file():
            files.append(p)
    return files


def print_tables(files):
    table_files = [f for f in files if f.suffix.lower() in TABLE_EXTS or f.name.endswith(".nii.gz")]
    print("\n=== Label/Metadata Tables ===")
    for tf in sorted(table_files):
        if tf.suffix.lower() in {".csv"}:
            print(f"[CSV] {tf}")
            if pd is not None:
                try:
                    df = pd.read_csv(tf)
                    print(f"  Columns: {list(df.columns)}")
                    print(df.head(3))
                except Exception as e:
                    print(f"  Could not read: {e}")
        elif tf.suffix.lower() in {".xlsx", ".xls"}:
            print(f"[XLSX] {tf}")
            if pd is not None:
                try:
                    df = pd.read_excel(tf)
                    print(f"  Columns: {list(df.columns)}")
                    print(df.head(3))
                except Exception as e:
                    print(f"  Could not read: {e}")
        elif tf.suffix.lower() == ".json":
            print(f"[JSON] {tf}")
            try:
                with tf.open("r", encoding="utf-8") as f:
                    obj = json.load(f)
                if isinstance(obj, dict):
                    print(f"  Keys: {list(obj.keys())[:5]}")
            except Exception as e:
                print(f"  Could not read: {e}")


def sample_shape(path: Path):
    ext = path.suffix.lower()
    if path.name.endswith(".nii.gz"):
        ext = ".nii.gz"
    if ext in {".nii", ".nii.gz"}:
        if nib is None:
            return "N/A (nibabel missing)"
        try:
            img = nib.load(str(path))
            data = img.get_fdata()
            return f"shape={tuple(data.shape)}"
        except Exception as e:
            return f"error: {e}"
    if ext in {".png", ".jpg", ".jpeg"}:
        if Image is None:
            return "N/A (PIL missing)"
        try:
            with Image.open(path) as im:
                return f"shape={im.size[::-1]} channels={len(im.getbands())}"
        except Exception as e:
            return f"error: {e}"
    return "unknown"


def summarize_images(files):
    print("\n=== Image Summary by Folder ===")
    by_dir = {}
    for f in files:
        ext = f.suffix.lower()
        if f.name.endswith(".nii.gz"):
            ext = ".nii.gz"
        if ext not in IMG_EXTS:
            continue
        by_dir.setdefault(f.parent, []).append(f)

    for d, flist in sorted(by_dir.items()):
        print(f"\n[{d}] count={len(flist)}")
        for sample in flist[:3]:
            print(f"  {sample.name}: {sample_shape(sample)}")


def main():
    if not DATA_ROOT.exists():
        print(f"Data root not found: {DATA_ROOT}")
        sys.exit(1)
    files = list_files()
    print(f"Found {len(files)} total files under {DATA_ROOT}")
    summarize_images(files)
    print_tables(files)


if __name__ == "__main__":
    main()
