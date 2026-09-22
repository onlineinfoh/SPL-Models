# Retained revision experiment

~~Active reported pipeline~~ — superseded for the current scope.

The files in this directory remain unedited records of the later re-analysis:
Dataset001_lungval/fold_0 segmentation and an internal-AUC classification sweep
that selected EfficientNet-B0. The analysis plan, lock, deviations, predictions,
training logs and checkpoints are preserved. Statements of current status inside
those historical documents describe that experiment, not the active model path.

The current configuration uses Dataset000_lung/fold_all and the original
classification checkpoints. See [the current result map](../docs/REPRODUCE.md).
Do not run `watchdog.sh`, `finalize_rebuild.sh`, or `run_protocol.py` as part of
the current correction; they orchestrate the separate revision experiment.
