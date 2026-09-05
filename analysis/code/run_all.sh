#!/usr/bin/env bash
# Run the classification analysis end to end.
#
#   bash analysis/code/run_all.sh
#
# Reads only from binary_classification/predictions_tight/ (never written to).
# Writes only to analysis/results/ and analysis/figures/.
# Override the interpreter with PY=/path/to/python.
set -euo pipefail

PY="${PY:-$HOME/venvs/prism/bin/python}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

echo "Interpreter: $PY"
"$PY" -c "import numpy,pandas,scipy,sklearn,matplotlib;print('numpy',numpy.__version__,'pandas',pandas.__version__,'scipy',scipy.__version__,'sklearn',sklearn.__version__,'mpl',matplotlib.__version__)"

for s in task0_data_check.py \
         task1_confidence_intervals.py \
         task2_threshold_policy.py \
         task3_calibration.py \
         task4_dca.py \
         task5_model_selection.py \
         task6_gt_vs_model_mask.py ; do
  echo
  echo "############################################################"
  echo "# $s"
  echo "############################################################"
  "$PY" "$s"
done

echo
echo "Done. Results in analysis/results/, figures in analysis/figures/."
