#!/usr/bin/env python3
"""
Rebuild label CSVs from data/ without external dependencies.

Reads:
  data/train/train.xlsx    (columns: 文件名, 良恶性)
  data/val/validation.xlsx (columns: 序列, 良恶性)
  data/test1/test1.xlsx    (columns: 序号, 良恶性)
  data/test2/test2.xlsx    (columns: 序号, 良恶性)

Writes:
  binary_classification/labels/labels_train.csv
  binary_classification/labels/labels_internal_val.csv
  binary_classification/labels/labels_external_test1.csv
  binary_classification/labels/labels_external_test2.csv

Uses a minimal XLSX reader (zip + XML), no pandas/openpyxl required.
"""

from __future__ import annotations

from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET
import csv

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT_DIR = ROOT / "binary_classification/labels"


def normalize_case(name: str) -> str:
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


def map_label(val) -> int:
    v = str(val).strip().lower()
    if v in {"1", "恶性", "malignant", "cancer", "positive", "pos"}:
        return 1
    if v in {"0", "良性", "benign", "negative", "neg"}:
        return 0
    raise ValueError(f"Unrecognized label value: {val}")


def read_xlsx_first_sheet(path: Path) -> list[list[str]]:
    """Minimal XLSX reader: returns rows as list of cell strings."""
    with zipfile.ZipFile(path, "r") as zf:
        # shared strings
        shared = []
        if "xl/sharedStrings.xml" in zf.namelist():
            ss = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in ss.findall(".//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}si"):
                t = si.find(".//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t")
                shared.append(t.text if t is not None else "")
        # sheet1
        sheet_name = "xl/worksheets/sheet1.xml"
        if sheet_name not in zf.namelist():
            raise FileNotFoundError(f"sheet1.xml not found in {path}")
        sheet = ET.fromstring(zf.read(sheet_name))
        rows = []
        for row in sheet.findall(".//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}row"):
            vals = []
            for c in row.findall("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}c"):
                t = c.get("t")
                v = c.find("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}v")
                if v is None or v.text is None:
                    vals.append("")
                    continue
                if t == "s":
                    idx = int(v.text)
                    vals.append(shared[idx] if idx < len(shared) else "")
                else:
                    vals.append(v.text)
            rows.append(vals)
        # normalize to same length
        max_len = max((len(r) for r in rows), default=0)
        rows = [r + [""] * (max_len - len(r)) for r in rows]
        return rows


def write_csv(path: Path, rows: list[tuple[str, int]]):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "label"])
        w.writerows(rows)
    print(f"Saved {len(rows)} labels -> {path}")


def build_train():
    xl = DATA / "train" / "train.xlsx"
    rows = read_xlsx_first_sheet(xl)
    header = rows[0]
    try:
        idx_fname = header.index("文件名")
        idx_label = header.index("良恶性")
    except ValueError:
        raise ValueError(f"Unexpected columns in {xl}: {header}")
    out_rows = []
    for r in rows[1:]:
        fname = r[idx_fname].strip()
        if not fname:
            continue
        case = normalize_case(fname)
        lbl = map_label(r[idx_label])
        out_rows.append((case, lbl))
    write_csv(OUT_DIR / "labels_train.csv", out_rows)


def build_internal():
    xl = DATA / "val" / "validation.xlsx"
    rows = read_xlsx_first_sheet(xl)
    header = rows[0]
    try:
        idx_seq = header.index("序列")
        idx_label = header.index("良恶性")
    except ValueError:
        raise ValueError(f"Unexpected columns in {xl}: {header}")
    out_rows = []
    for r in rows[1:]:
        if not r[idx_seq]:
            continue
        num = int(float(r[idx_seq]))
        case = f"case_{num:05d}"
        lbl = map_label(r[idx_label])
        out_rows.append((case, lbl))
    write_csv(OUT_DIR / "labels_internal_val.csv", out_rows)


def build_ext(which: str):
    xl = DATA / which / f"{which}.xlsx"
    rows = read_xlsx_first_sheet(xl)
    header = rows[0]
    try:
        idx_seq = header.index("序号")
        idx_label = header.index("良恶性")
    except ValueError:
        raise ValueError(f"Unexpected columns in {xl}: {header}")
    out_rows = []
    for r in rows[1:]:
        if not r[idx_seq]:
            continue
        num = int(float(r[idx_seq]))
        case = f"case_{num:05d}"
        lbl = map_label(r[idx_label])
        out_rows.append((case, lbl))
    write_csv(OUT_DIR / f"labels_external_{which}.csv", out_rows)


def main():
    build_train()
    build_internal()
    build_ext("test1")
    build_ext("test2")


if __name__ == "__main__":
    main()
