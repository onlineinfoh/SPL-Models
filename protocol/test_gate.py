"""Smoke test for protocol/gate.py. Read-only on real data; harden/soften on a temp tree."""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path("/home/tianxi-liang/TianxiLiang/research/china/SPL-Models")
sys.path.insert(0, str(REPO))

from protocol import gate  # noqa: E402

PASS, FAIL = [], []


def check(name, fn, want_raise: bool):
    try:
        fn()
        ok = not want_raise
        detail = "no exception"
    except gate.ExternalDataLocked:
        ok = want_raise
        detail = "ExternalDataLocked"
    except Exception as e:
        ok = False
        detail = f"WRONG EXCEPTION {type(e).__name__}: {e}"
    (PASS if ok else FAIL).append(f"{name}: {detail}")


print("=== initial status ===")
print(json.dumps(gate.status(), indent=2))
assert gate.is_locked(), "expected locked (locked_pipeline.json absent)"

# pick a real gated file
gated_files = sorted((REPO / "data" / "test1" / "img_test1").glob("*"))
assert gated_files, "no files under data/test1/img_test1 to test with"
GF = gated_files[0]
print(f"\ngated sample file: {GF.relative_to(REPO)}")

UNGATED = REPO / "README.md"

gate.install()
print("guard installed\n")

# --- reads that MUST be blocked -------------------------------------------
check("builtins.open(gated)", lambda: open(GF, "rb").close(), True)
check("Path.open(gated)", lambda: GF.open("rb").close(), True)
check("Path.read_bytes(gated)", lambda: GF.read_bytes(), True)
check("os.listdir(gated dir)", lambda: os.listdir(REPO / "data" / "test1"), True)
check("os.scandir(gated dir)", lambda: list(os.scandir(REPO / "data" / "test2")), True)
check("os.open(gated)", lambda: os.close(os.open(GF, os.O_RDONLY)), True)
check("os.stat(gated)", lambda: os.stat(GF), True)
check("gated label csv", lambda: open(
    REPO / "binary_classification" / "labels" / "labels_external_test1.csv").close(), True)


def _numpy_read():
    import numpy as np
    np.loadtxt(REPO / "binary_classification" / "labels" / "labels_external_test2.csv",
               delimiter=",", dtype=str)


check("numpy.loadtxt(gated)", _numpy_read, True)

# --- bypasses found by external audit; each MUST now be blocked --------------
import io as _io  # noqa: E402
import subprocess as _sp  # noqa: E402

check("open(file=KEYWORD)", lambda: open(file=GF, mode="rb").close(), True)
check("io.open(gated)", lambda: _io.open(GF, "rb").close(), True)
check("io.open(file=KEYWORD)", lambda: _io.open(file=GF, mode="rb").close(), True)
check("os.open(path=KEYWORD)",
      lambda: os.close(os.open(path=str(GF), flags=os.O_RDONLY)), True)
check("subprocess.run(gated in argv)",
      lambda: _sp.run(["cat", str(GF)], capture_output=True), True)
check("subprocess.Popen(gated in argv)",
      lambda: _sp.Popen(["cat", str(GF)], stdout=_sp.DEVNULL).wait(), True)
check("subprocess.check_output(gated)",
      lambda: _sp.check_output(["ls", str(REPO / "data" / "test1")]), True)

# --- tier 2: prior external RESULTS must also be held out -------------------
check("prior ext AUC training logs", lambda: os.listdir(
    REPO / "analysis" / "logs" / "training_logs" / "single_seed_67"), True)
check("task5_architecture_ranking.csv", lambda: open(
    REPO / "analysis" / "results" / "task5_architecture_ranking.csv").close(), True)
check("checkpoint_selection_summary.csv", lambda: open(
    REPO / "analysis" / "results" / "checkpoint_selection_summary.csv").close(), True)
check("predictions_tight/ (ext probs)", lambda: os.listdir(
    REPO / "binary_classification" / "predictions_tight"), True)


def _pandas_read():
    import pandas as pd
    pd.read_csv(REPO / "binary_classification" / "labels" / "labels_external_test1.csv")


check("pandas.read_csv(gated)", _pandas_read, True)


def _nib_read():
    import nibabel as nib
    nib.load(str(GF)).get_fdata()


if GF.name.endswith(".nii.gz"):
    check("nibabel.load(gated)", _nib_read, True)

# --- reads that MUST still work -------------------------------------------
check("builtins.open(ungated)", lambda: open(UNGATED).close(), False)
check("os.listdir(ungated)", lambda: os.listdir(REPO / "data" / "train"), False)
check("internal_val images", lambda: os.listdir(REPO / "data" / "val" / "img_v"), False)
check("train images", lambda: os.listdir(REPO / "data" / "train" / "imagesTr"), False)
check("protocol yaml", lambda: (REPO / "protocol" / "analysis_plan_v1.yaml").read_text(), False)

# --- release must refuse while unlocked -----------------------------------
check("release() while locked", gate.release, True)

gate.uninstall()

# --- harden / soften round trip on a temp tree ----------------------------
tmp = Path(tempfile.mkdtemp())
try:
    (tmp / "test1").mkdir()
    (tmp / "test1" / "x.txt").write_text("data")
    original_mode = oct(os.stat(tmp / "test1").st_mode & 0o777)

    saved_gated, saved_resolved = gate.GATED, gate._GATED_RESOLVED
    saved_backup = gate.MODE_BACKUP
    gate.GATED = [tmp / "test1"]
    gate._GATED_RESOLVED = [os.path.realpath(tmp / "test1")]
    gate.MODE_BACKUP = tmp / ".modes.json"

    gate.harden()
    hardened_mode = oct(os.stat(tmp / "test1").st_mode & 0o777)
    kernel_blocked = False
    try:
        os.listdir(tmp / "test1")
    except PermissionError:
        kernel_blocked = True

    gate.soften()
    restored_mode = oct(os.stat(tmp / "test1").st_mode & 0o777)

    (PASS if hardened_mode == "0o0" else FAIL).append(f"harden sets 000: got {hardened_mode}")
    (PASS if kernel_blocked else FAIL).append("harden blocks at kernel level (covers cv2/SimpleITK)")
    (PASS if restored_mode == original_mode else FAIL).append(
        f"soften restores mode: {original_mode} -> {restored_mode}")

    gate.GATED, gate._GATED_RESOLVED, gate.MODE_BACKUP = saved_gated, saved_resolved, saved_backup
finally:
    shutil.rmtree(tmp, ignore_errors=True)

# --- audit log ------------------------------------------------------------
audit = REPO / "protocol" / "gate_audit.jsonl"
if audit.exists():
    lines = [json.loads(l) for l in audit.read_text().splitlines() if l.strip()]
    denied = [l for l in lines if l["event"] == "DENIED"]
    (PASS if denied else FAIL).append(f"audit log records denials: {len(denied)} entries")
else:
    FAIL.append("audit log not written")

print("\n" + "=" * 62)
for p in PASS:
    print(f"  PASS  {p}")
for f in FAIL:
    print(f"  FAIL  {f}")
print("=" * 62)
print(f"{len(PASS)} passed, {len(FAIL)} failed")
sys.exit(1 if FAIL else 0)
