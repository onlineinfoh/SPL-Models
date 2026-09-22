# Scope correction — 2026-09-22 UTC

The Claude session `49a41542-fcea-4220-ab07-0a42c71aa7cb` was checked. It records
the request to return to `6475d07`, but only uncommitted inference changes were
reverted; HEAD remained `9f18f16`. The present update restores the original model
path with explicitly requested corrections. It is **not** a byte-for-byte Git
rollback: revision code, results and logs remain available.

## Changes to active code

| Previous setting/status | Current setting | Effect |
|---|---|---|
| ~~Original `train.py` superseded by protocol runner~~ | Original classifier training path is active | Removed only the stale status header; checkpoint selection remains internal accuracy |
| ~~External AUC text printed/written each epoch and in final training summary~~ | Those text statements suppressed | External evaluation, history arrays and plots remain; no historical logs edited |
| ~~224 px classification inference~~ | 300 px | Same checkpoints, preprocessing procedure and threshold-selection rule apart from resize; predictions may change |
| ~~Write corrected probabilities over `predictions_tight/`~~ | Write to `predictions_tight_300/` | Original deposited predictions preserved |
| ~~224 px default Grad-CAM / old prediction source~~ | 300 px; new prediction/output directories | Keeps default visualisation aligned with corrected inference |
| ~~Dataset001/fold_0 as the active nnU-Net prediction driver~~ | Dataset000/fold_all, checkpoint_best.pth | Previous driver retained in archive; original external results untouched |
| ~~Disable test-time mirroring in the revision driver~~ | Original benchmark default mirroring enabled in fold_all CLI | Future prediction only; no new masks generated or equivalence claimed |
| ~~Internal-validation Dice during fold_all training~~ | Training-set Dice / pseudo-Dice as appropriate | Interpretation correction; separate tuning/external evaluations retain their labels |
| ~~Revision README claims all reported results follow one locked pipeline~~ | Result-specific map and explicit provenance gaps | Historical and corrected outputs distinguished |

`predict_all_cohorts.sh` also fails on empty input cohorts or prediction failures.
The documented training CLI uses the correct positional fold argument. The
classification shell driver now labels its already-disabled training step as
using existing checkpoints, instead of announcing training. New
derived-mask/prediction/heatmap directories are ignored by Git, consistent with
the existing policy for local derived image data.

The dead inference directory `new_data/` is **not** restored: the existing
correction to `data/` is retained. Shared analysis helpers, including native-size
U-Net inference and optional environment overrides, are preserved; their
historical configuration differences are described in REPRODUCE.md.

## Preservation

The exact previous versions of every changed model entry point and the main
documentation are in
[`history/2026-09-22-before-scope-correction/`](history/2026-09-22-before-scope-correction/manifest.json).
These are source records, not scripts to execute from the archive: their paths
are relative to their original locations.

`preserved_artifacts.json` records SHA256 values of 427 existing tracked
result/configuration/log files before edits. They are checked again after edits.
No model weights, masks, original prediction files or training logs were edited.
The `protocol/` plan, lock and historical deviations are untouched; its new
README marks that experiment superseded for the present scope.

No code or historical log was deleted. The earlier staged deletions of
`DISCLOSURE.md` and `docs/RESPONSE_LETTER_DRAFT.md` predate this work and were not
changed. No reset, commit, push, retraining or inference was performed.

## Computation performed

`analysis/code/audit_existing_selection.py` read existing probabilities and both
saved 39-run sweep records, recomputed all original accuracy/AUC rankings, and
wrote the complete results to `docs/selection_audit/`. It does not train, infer,
change thresholds using external data, or search seeds. Ties are explicit.

`analysis/code/write_provenance_manifest.py` hashes source/configuration files,
existing result artifacts and available original checkpoints. Its manifest
identifies bytes currently available, not a retroactive pre-analysis lock.

## Pending

The authors subsequently chose the verified original seed-67 automatic-mask
comparison for reporting. No different seed is asserted. The saved evidence
does not establish DenseNet121 as first on both external cohorts. Corrected 300 px inference and
its downstream tables have not been run. Manuscript/Supplement/Figure 2 source
files were unavailable; the colleague checklist identifies remaining decisions.
