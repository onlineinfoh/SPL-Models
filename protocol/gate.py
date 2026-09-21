#!/usr/bin/env python3
"""
Runtime hold-out enforcement for the external cohorts.

While `protocol/locked_pipeline.json` is absent, any read of a gated path
raises `ExternalDataLocked` and the process stops. The gate is released only
after the lock record exists, which the Phase D lock step writes from internal
results alone. Every install, release and attempted access is appended to
`protocol/gate_audit.jsonl`, so the audit trail is a produced artifact rather
than a claim in prose.

Usage, at the top of any training or selection entry point:

    from protocol import gate
    gate.install()

Coverage and its limits
-----------------------
The Python-level guard patches `builtins.open`, `os.open`, `os.scandir`,
`os.listdir`, `os.stat` and the `pathlib.Path` read methods. That covers numpy,
pandas, nibabel, csv and every ordinary file read.

It does NOT cover libraries that call the C `open()` directly, notably
`cv2.imread` and `SimpleITK.ReadImage`. For those, use `harden()`, which sets
mode 000 on the gated directories so enforcement happens in the kernel. Phase C
should run under `harden()`; `release()` restores the original modes.
"""

from __future__ import annotations

import builtins
import io
import json
import os
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PROTOCOL = REPO / "protocol"
LOCK_RECORD = PROTOCOL / "locked_pipeline.json"
AUDIT_LOG = PROTOCOL / "gate_audit.jsonl"
MODE_BACKUP = PROTOCOL / ".gate_modes.json"

# Paths held out until the lock record exists. Kept in sync with
# holdout.gated_paths in analysis_plan_v1.yaml.
GATED = [
    # Tier 1: the external cohorts themselves.
    REPO / "data" / "test1",
    REPO / "data" / "test2",
    REPO / "binary_classification" / "labels" / "labels_external_test1.csv",
    REPO / "binary_classification" / "labels" / "labels_external_test2.csv",

    # Tier 2: PRIOR EXTERNAL RESULTS. Holding out the images is not sufficient.
    # The December run recorded per-epoch ext1/ext2 AUC for all 13 architectures,
    # and those numbers sit readable in the repository. A selection made while
    # they are reachable is not blind to external performance even if no external
    # image is opened, so they are held out for the duration of Phase C and D.
    REPO / "analysis" / "logs" / "training_logs" / "single_seed_67",
    REPO / "analysis" / "results" / "task5_architecture_ranking.csv",
    REPO / "analysis" / "results" / "checkpoint_selection_summary.csv",
    REPO / "analysis" / "results" / "checkpoint_selection_summary.json",
    REPO / "analysis" / "results" / "locked_selection_and_external.json",
    REPO / "analysis" / "results" / "table2_corrected.csv",
    REPO / "analysis" / "results" / "table2_corrected.md",
    REPO / "binary_classification" / "predictions_tight",
]


class ExternalDataLocked(RuntimeError):
    """Raised when gated data is touched before the pipeline is locked."""


# Saved before patching so the guard itself can touch the filesystem.
_real = {
    "open": builtins.open,
    "io_open": io.open,
    "os_open": os.open,
    "scandir": os.scandir,
    "listdir": os.listdir,
    "stat": os.stat,
    "path_open": Path.open,
    "read_bytes": Path.read_bytes,
    "read_text": Path.read_text,
    "iterdir": Path.iterdir,
    "sp_popen": subprocess.Popen,
    "sp_run": subprocess.run,
    "sp_call": subprocess.call,
    "sp_check_output": subprocess.check_output,
}

_installed = False
_checking = False  # re-entrancy guard: the audit writer must not self-trip


# ---------------------------------------------------------------------------
# Lock state
# ---------------------------------------------------------------------------

def is_locked() -> bool:
    """True while the external cohorts are held out."""
    try:
        _real["stat"](LOCK_RECORD)
        return False
    except OSError:
        return True


def _audit(event: str, **fields) -> None:
    global _checking
    _checking = True
    try:
        record = {
            "utc": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "pid": os.getpid(),
            "argv": " ".join(sys.argv[:3]),
            **fields,
        }
        with _real["open"](AUDIT_LOG, "a") as fh:
            fh.write(json.dumps(record) + "\n")
    finally:
        _checking = False


# ---------------------------------------------------------------------------
# Path classification
# ---------------------------------------------------------------------------

_GATED_RESOLVED = [os.path.realpath(p) for p in map(str, GATED)]


def _is_gated(target) -> bool:
    if _checking:
        return False
    try:
        if isinstance(target, int):          # already-open fd
            return False
        p = os.path.realpath(os.fspath(target))
    except (TypeError, ValueError, OSError):
        return False
    for g in _GATED_RESOLVED:
        if p == g or p.startswith(g + os.sep):
            return True
    return False


def _deny(target, via: str):
    _audit("DENIED", path=str(target), via=via)
    raise ExternalDataLocked(
        f"\n"
        f"  Blocked read of held-out data via {via}:\n"
        f"    {target}\n\n"
        f"  The external cohorts are held out until the pipeline is locked.\n"
        f"  {LOCK_RECORD.relative_to(REPO)} does not exist yet.\n\n"
        f"  If this fired during training or model selection, the protocol is\n"
        f"  working as intended: that step must not see external data.\n"
        f"  Run the Phase D lock step first, then re-run under a released gate.\n"
    )


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

def _guard(fn_key: str, via: str, arg_index: int = 0, kwarg: str | None = None):
    """
    Wrap a read function.

    Both the positional slot AND the keyword form must be checked. An earlier
    version inspected only args[arg_index], which left open(file=p), io.open(p)
    and os.open(path=p) as working bypasses; an external audit demonstrated all
    three against the real cohort. Every call form the function accepts has to be
    covered or the guard is decorative.
    """
    real = _real[fn_key]

    def wrapper(*args, **kwargs):
        if is_locked():
            target = None
            if len(args) > arg_index:
                target = args[arg_index]
            elif kwarg is not None and kwarg in kwargs:
                target = kwargs[kwarg]
            if target is not None and _is_gated(target):
                _deny(target, via)
        return real(*args, **kwargs)

    return wrapper


def _guard_subprocess(fn_key: str, via: str):
    """
    Block a subprocess whose command line names a gated path.

    External tooling (the nnUNetv2_* CLIs, SimpleITK-backed converters) runs
    outside this process entirely, so the Python patches never see those reads.
    Scanning the argv is the only interception point available.
    """
    real = _real[fn_key]

    def wrapper(*args, **kwargs):
        if is_locked():
            cmd = kwargs.get("args", args[0] if args else None)
            parts = ([cmd] if isinstance(cmd, (str, bytes, os.PathLike))
                     else list(cmd) if isinstance(cmd, (list, tuple)) else [])
            for part in parts:
                try:
                    s = os.fspath(part) if not isinstance(part, bytes) else part.decode()
                except (TypeError, ValueError):
                    continue
                for g in _GATED_RESOLVED:
                    if g in str(s):
                        _deny(s, f"{via} (subprocess argv)")
        return real(*args, **kwargs)

    return wrapper


def _guard_method(fn_key: str, via: str):
    real = _real[fn_key]

    def wrapper(self, *args, **kwargs):
        if is_locked() and _is_gated(self):
            _deny(self, via)
        return real(self, *args, **kwargs)

    return wrapper


def install() -> None:
    """Patch the Python read paths. Idempotent."""
    global _installed
    if _installed:
        return

    builtins.open = _guard("open", "builtins.open", kwarg="file")
    io.open = _guard("io_open", "io.open", kwarg="file")
    os.open = _guard("os_open", "os.open", kwarg="path")
    os.scandir = _guard("scandir", "os.scandir", kwarg="path")
    os.listdir = _guard("listdir", "os.listdir", kwarg="path")
    os.stat = _guard("stat", "os.stat", kwarg="path")
    Path.open = _guard_method("path_open", "Path.open")
    Path.read_bytes = _guard_method("read_bytes", "Path.read_bytes")
    Path.read_text = _guard_method("read_text", "Path.read_text")
    Path.iterdir = _guard_method("iterdir", "Path.iterdir")

    # External tooling runs outside this interpreter, so argv is the only hook.
    subprocess.Popen = _guard_subprocess("sp_popen", "subprocess.Popen")
    subprocess.run = _guard_subprocess("sp_run", "subprocess.run")
    subprocess.call = _guard_subprocess("sp_call", "subprocess.call")
    subprocess.check_output = _guard_subprocess("sp_check_output", "subprocess.check_output")

    _installed = True
    _audit("INSTALLED", locked=is_locked(),
           gated=[str(Path(p).relative_to(REPO)) for p in _GATED_RESOLVED])


def uninstall() -> None:
    """Restore the original functions. Used by the Phase E runner after lock."""
    global _installed
    if not _installed:
        return
    builtins.open = _real["open"]
    io.open = _real["io_open"]
    os.open = _real["os_open"]
    os.scandir = _real["scandir"]
    os.listdir = _real["listdir"]
    os.stat = _real["stat"]
    Path.open = _real["path_open"]
    Path.read_bytes = _real["read_bytes"]
    Path.read_text = _real["read_text"]
    Path.iterdir = _real["iterdir"]
    subprocess.Popen = _real["sp_popen"]
    subprocess.run = _real["sp_run"]
    subprocess.call = _real["sp_call"]
    subprocess.check_output = _real["sp_check_output"]
    _installed = False
    _audit("UNINSTALLED")


# ---------------------------------------------------------------------------
# Kernel-level enforcement, for cv2 / SimpleITK which bypass Python
# ---------------------------------------------------------------------------

def harden() -> None:
    """Set mode 000 on gated directories. Original modes are saved."""
    if not is_locked():
        raise RuntimeError("refusing to harden: pipeline is already locked")
    modes = {}
    for p in GATED:
        if p.exists():
            modes[str(p)] = stat.S_IMODE(_real["stat"](p).st_mode)
            os.chmod(p, 0o000)
    with _real["open"](MODE_BACKUP, "w") as fh:
        json.dump(modes, fh, indent=2)
    _audit("HARDENED", paths=sorted(modes))


def soften() -> None:
    """Restore the modes saved by harden()."""
    if not MODE_BACKUP.exists():
        _audit("SOFTEN_NOOP")
        return
    with _real["open"](MODE_BACKUP) as fh:
        modes = json.load(fh)
    for p, mode in modes.items():
        # _real["stat"], not os.path.exists: the gated paths are exactly what the
        # installed guard blocks, and soften() must work while it is still closed.
        try:
            _real["stat"](p)
        except OSError:
            continue
        os.chmod(p, mode)
    MODE_BACKUP.unlink()
    _audit("SOFTENED", paths=sorted(modes))


# ---------------------------------------------------------------------------
# Release
# ---------------------------------------------------------------------------

def release() -> None:
    """
    Open the gate. Only legitimate once the lock record exists and validates.

    Called by the Phase E external inference runner, never by training.
    """
    if is_locked():
        raise ExternalDataLocked(
            f"cannot release: {LOCK_RECORD.relative_to(REPO)} does not exist. "
            f"Run the Phase D lock step first."
        )
    with _real["open"](LOCK_RECORD) as fh:
        record = json.load(fh)
    required = {"selected_architecture", "selected_seed", "threshold",
                "git_commit", "weights_sha256"}
    missing = required - record.keys()
    if missing:
        raise ExternalDataLocked(f"lock record incomplete, missing: {sorted(missing)}")

    soften()
    uninstall()
    _audit("RELEASED", **{k: record[k] for k in
                          ("selected_architecture", "selected_seed",
                           "threshold", "git_commit")})


def status() -> dict:
    return {
        "locked": is_locked(),
        "guard_installed": _installed,
        "hardened": MODE_BACKUP.exists(),
        "lock_record": str(LOCK_RECORD.relative_to(REPO)),
        "gated": [str(Path(p).relative_to(REPO)) for p in _GATED_RESOLVED],
    }


if __name__ == "__main__":
    print(json.dumps(status(), indent=2))
