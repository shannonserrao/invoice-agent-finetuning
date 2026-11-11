# CLAUDE.md — working rules for this repository

This project is spec-driven. The documents are the source of truth; code follows them.

## Read order
1. `SPEC.md` — what we are building and how success is measured.
2. `PLAN.md` — architecture and technical decisions.
3. `docs/architecture.md` — diagrams of the pipeline, inference path and release flow.
4. `specs/` — one detailed spec per component (data, training, evaluation, tracking, feedback, serving).
5. `TASKS.md` — the ordered work list. Each task names the spec section it implements and its acceptance check.
6. `BUILD.md` — how to run everything on local, Colab, Kaggle and Databricks.

## Rules
- Work one task from `TASKS.md` at a time, in order. Tick it off in the same commit that completes it.
- If code and spec disagree, stop and fix the spec first (with a line in `CHANGELOG.md`), then the code.
- Every behaviour change needs a test in `tests/` or an evaluation result logged to the tracker.
- No number goes into `README.md` or `docs/model_card.md` unless it comes from a tracked run. Cite the run ID.
- Never commit data, model weights, credentials or `.env`. Only manifests and configs are versioned in git.
- All parameters live in `configs/`. No hard-coded paths, model names or hyperparameters in `src/`.
- Platform differences (Colab, Kaggle, Databricks, local) are handled only in `platforms/` and by config overrides.
- The test split is touched only by `make eval`. Never use it for training, threshold tuning or prompt tuning.

## Commands
- `make setup` — install the package with dev extras.
- `make lint test` — must pass before any commit.
- `make data train eval` — full pipeline with `configs/base.yaml`.
- `make smoke` — 2-minute CPU run on a tiny subset; use this to check wiring.

## Implemented vs stubbed
Implemented and unit-tested: `config.py`, `schema.py`, `metrics.py`, `evaluate.py`, `tracking.py`, `feedback.py`, `data.py`, `baseline.py`, `routing.py`, `confidence.py`, `review.py`, `bundle.py`, `cli.py`.
Implemented, tested on helpers only (end-to-end run still to verify, T-07): `train.py`.
Stub that raises `NotImplementedError` with a task ID: `predict.py` (T-08, T-09). Implement it by following `TASKS.md`.

## Commit style
Conventional Commits (`feat:`, `fix:`, `docs:`, `data:`, `exp:`). Reference the task, e.g. `feat(data): T-03 build manifest`.
