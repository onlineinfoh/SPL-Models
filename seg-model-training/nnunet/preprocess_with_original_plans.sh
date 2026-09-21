#!/usr/bin/env bash
# Preprocess Dataset001_lungval using Dataset000_lung's ORIGINAL plans.
#
# Two reasons not to re-plan:
#
#   1. Concordance. The manuscript documents patch 896x1792, batch 2, spacing
#      1.0x1.0, CTNormalization. Re-planning on 857 cases instead of 600 would
#      shift those values, so the retrained model would differ from the
#      published one in architecture AND validation split at once. Forcing the
#      plans leaves the validation split as the single changed variable.
#
#   2. Correctness. foreground_intensity_properties_per_channel drives
#      CTNormalization and was computed on the 600 TRAINING cases. Re-planning
#      would recompute it over train + internal_val, leaking validation-set
#      intensity statistics into the normalisation applied at training time.
#
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NN="$REPO/seg-model-training/nnunet"

export nnUNet_raw="$NN/nnUNet_raw"
export nnUNet_preprocessed="$NN/nnUNet_preprocessed"
export nnUNet_results="$NN/nnUNet_results"
# nnunet-env is NOT usable. It was created at
#   /home/tianxi-liang/TianxiLiang/research/china/nnunet-env
# and later moved into the repository, so every console script still carries a
# shebang pointing at an interpreter that no longer exists, and `import
# nnunetv2` fails inside it. The original nnU-Net run (fold_all/debug.json,
# torch 2.9.1+cu128) was produced in that environment and cannot be reproduced
# in it. The rebuild therefore runs in the prism environment
# (torch 2.13.0+cu126). This is a genuine deviation from the published run and
# is recorded in protocol/DEVIATIONS.md.
PY="$HOME/venvs/prism/bin/python"
BIN="$HOME/venvs/prism/bin"

SRC_PLANS="$nnUNet_preprocessed/Dataset000_lung/nnUNetPlans.json"
DST_DIR="$nnUNet_preprocessed/Dataset001_lungval"

echo "=== 1. fingerprint (shapes only; plans are overwritten below) ==="
"$BIN/nnUNetv2_extract_fingerprint" -d 1 --verify_dataset_integrity

echo "=== 2. install original plans, dataset_name patched ==="
mkdir -p "$DST_DIR"
# nnUNetv2_plan_experiment normally copies dataset.json into the preprocessed
# folder. We skip planning (the whole point is to reuse the published plans),
# so it has to be copied explicitly or the preprocessor cannot find it.
cp "$nnUNet_raw/Dataset001_lungval/dataset.json" "$DST_DIR/dataset.json"
"$PY" - "$SRC_PLANS" "$DST_DIR/nnUNetPlans.json" <<'PYEOF'
import json, sys
src, dst = sys.argv[1], sys.argv[2]
p = json.load(open(src))
assert p["configurations"]["2d"]["patch_size"] == [896, 1792], "unexpected source patch size"
p["dataset_name"] = "Dataset001_lungval"
p["_provenance"] = (
    "Copied verbatim from Dataset000_lung/nnUNetPlans.json with dataset_name "
    "patched. Not re-planned: this preserves the published architecture "
    "configuration and keeps CTNormalization statistics derived from the 600 "
    "training cases only."
)
json.dump(p, open(dst, "w"), indent=2)
print(f"  patch_size {p['configurations']['2d']['patch_size']}  "
      f"batch {p['configurations']['2d']['batch_size']}  "
      f"norm {p['configurations']['2d']['normalization_schemes']}")
PYEOF

echo "=== 3. preprocess 2d with those plans ==="
"$BIN/nnUNetv2_preprocess" -d 1 -c 2d -plans_name nnUNetPlans -np 8

echo "=== 4. install the explicit train/val split ==="
cp "$NN/splits_final.json" "$DST_DIR/splits_final.json"
"$PY" - "$DST_DIR/splits_final.json" <<'PYEOF'
import json, sys
s = json.load(open(sys.argv[1]))
assert len(s) == 1, "expected a single fold"
tr, va = set(s[0]["train"]), set(s[0]["val"])
assert not tr & va, "train/val overlap in splits_final.json"
print(f"  fold 0: {len(tr)} train / {len(va)} val, disjoint")
PYEOF

echo
echo "ready. train with:"
echo "  nnUNetv2_train 1 2d 0"
