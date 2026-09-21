# Note on removed and restored files

*DRAFT - review before publication.*

This note exists so that a reader inspecting the commit history has the context
next to it.

## What happened

Commit `6475d07` ("Remove scratch evaluation, resolve paths relative to the
repository", 2026-09-05) deleted the `test/` directory and
`analysis/logs/train_locked_sweep.log`, and added `test/` and `**/*.log` to
`.gitignore`.

The deleted material was an exploratory evaluation carried out on a cohort that
is **not part of this study**. Those images were never used for training,
validation, model selection, threshold derivation or any reported result. The
evaluation was scratch work.

The commit message gave as its reason that the material "would be read as
contradicting the reported results." That phrasing describes a presentational
concern rather than the actual reason the material is out of scope, which is
simply that the cohort is not part of the study. Removing the files was the
wrong way to handle it.

## Current state

All deleted files are restored and tracked:

- `test/` - the exploratory evaluation scripts and their outputs
- `analysis/logs/train_locked_sweep.log` - console record of the internal-only
  architecture sweep

The `**/*.log` ignore rule was too broad: it also excluded analysis logs that
belong in the repository. It has been narrowed to scratch environment logs only.

Nothing described here is excluded from the public repository.

## Scope

The exploratory cohort is not reported in the manuscript, because it is not part
of the study. The restored files are retained so the repository record is
complete, not because they bear on the reported results.

The study cohorts are the 600-patient training set and the 257-patient tuning
set from Center 1, and the external cohorts of 108 patients (Center 2) and 94
patients (Center 3). Those, and only those, are described in the manuscript.

## Related

- [`protocol/DEVIATIONS.md`](protocol/DEVIATIONS.md) - departures from the
  pre-committed analysis plan
- [`docs/REPRODUCE.md`](docs/REPRODUCE.md) - every reported result mapped to the
  code that produced it
