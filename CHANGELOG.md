# Changelog

All notable changes to specs, data, code and models. Newest first.

## [Unreleased]
- feedback: T-15 `make review-page` builds `feedback/review.html` from the review queue: image beside editable JSON, schema check in the page, verdict buttons, download of `reviewed.csv` in the ingest format. Local file only; nothing is uploaded. Fix: `schema.schema_errors` now rejects a non-string line-item `description`, as `schema/document.schema.json` requires.
- data: T-13 corrections merge in `data.py` (`merge_corrections`). Corrections go to the training split only; validation/test ids, leaked and duplicate images are excluded and reported in the manifest; data version becomes `<version>-fb<fingerprint>`. Spec 01 §4 order changed: caps first, then corrections, so corrections are never capped away. `feedback-ingest` now validates `corrected` against the schema (HF-2). New config `feedback.include_confirmed` (default false).
- routing: T-10 `routing.py` (`fit_threshold`, `make fit-threshold`, `cascade`) and T-11 tooling `confidence.py` (risk-coverage AURC vs random, AUROC, permutation test). `evaluate` now reports `confidence` and, with `routing.escalate_to_baseline`, `cascade`. Spec 06 updated.
- baseline: `baseline.py` and `make baseline` (T-06 code). Zero-shot `claude-opus-5-5` with schema in the prompt, on-disk cache keyed by model/prompt hash/image hash, token and cost accounting, run tagged `baseline=true`. Config gains `prompt.*` and `baseline.*`. Not yet run live: needs an API key.
- eval: per-dataset scoring scope. `evaluate.py` now scores only `data.datasets.<name>.fields` and skips fields listed in `unscored_if_null` when gold is null, so correct predictions on unknown gold (SROIE ambiguous dates) are not counted as false positives. Spec 03 updated.
- data: T-05 SROIE mapper (`map_sroie`, `parse_date`). SROIE is eval-only: `data.datasets.sroie.eval_only` writes `data/processed/test_sroie.jsonl` from the published test split (361) and is excluded from train/validation/test. Ambiguous numeric dates become null (125 of 361).
- data: implemented `data.py` (CORD mapper, `build_dataset`, leakage and schema checks, gold arithmetic report, manifest). Pinned CORD and Qwen2.5-VL-3B revisions; `data.datasets.cord.fields` no longer lists vendor_name (CORD does not label it). Added `data.version`.
- Initial scaffold: specs, plan, tasks, architecture diagrams, config loader, metrics, evaluation, tracking and feedback modules with tests.
- Stubs awaiting implementation: `train.py`, `predict.py` (see `TASKS.md`).
