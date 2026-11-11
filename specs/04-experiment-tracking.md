# Spec 04 — Experiment tracking

## Goal
Any result can be reproduced from its run ID alone.

## Tool
MLflow. Local file store by default (`./mlruns`), Drive on Colab, `/kaggle/working` on Kaggle, the managed service on Databricks. The wrapper in `src/docextract/tracking.py` falls back to a JSONL log under `artifacts/runs/` when MLflow is not installed, so tests and smoke runs never depend on it.

## Requirements
- **TR-1** Every `train`, `predict` and `evaluate` command opens a run.
- **TR-2** Each run logs as parameters: the fully resolved config (flattened), git SHA, dirty flag, data version, platform name, base model ID.
- **TR-3** Each run logs as metrics: the metrics in `specs/03-evaluation.md`, plus training loss per logging step, wall-clock seconds, peak GPU memory.
- **TR-4** Each run logs as artifacts: resolved config YAML, `report.json`, predictions file, and for training runs the adapter or model weights.
- **TR-5** Runs are tagged `stage` = `smoke`, `dev` or `release`; and `baseline` = `true` for baseline-model runs.
- **TR-6** A model is registered only from a run whose `stage` is `release`.

## Naming
Experiment: `invoice-receipt-extraction` (suffix `-smoke` for smoke runs). Run name: `<stage>-<short description>-<git sha7>`.

## Comparing runs
`mlflow ui --backend-store-uri ./mlruns` locally. The comparison table required for the README is produced from tracked runs only (task T-11).

## Acceptance
- A fresh clone can reproduce a logged metric within the tolerance in `specs/03-evaluation.md` using only the run's logged config, data version and git SHA.
