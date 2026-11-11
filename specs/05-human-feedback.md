# Spec 05 — Human feedback loop

## Goal
People check the predictions the model is least sure of, plus a random sample, and their corrections become training data for the next model version.

## Flow
See diagram 3 in `docs/architecture.md`.
1. `predict` appends every prediction to `feedback/predictions_log.jsonl` with `id`, `input_ref`, `prediction`, `confidence`, `model_version`.
2. `make feedback-export` writes `feedback/review_queue.csv` containing:
   - every item with confidence below `feedback.review_threshold` (reason `low_confidence`);
   - a random `feedback.audit_rate` share of the rest (reason `random_audit`).
3. A reviewer fills in `verdict`, `corrected`, `reviewer`, `note`. `input_ref` is the document image path. Either open the CSV in a spreadsheet and view the image beside the predicted JSON, or run `make review-page` and open `feedback/review.html`: it shows the image beside editable JSON, checks the edit against the schema, and downloads `reviewed.csv` in the format step 4 reads.
4. Save as `feedback/reviewed.csv` and run `make feedback-ingest`. Valid rows are appended to `feedback/corrections.jsonl`, which is committed.
5. The next `make data` merges corrections into the training split and bumps the data version (see spec 01 §4).

## Requirements
- **HF-1** `verdict` is one of `correct`, `incorrect`, `unsure`.
- **HF-2** `incorrect` requires `corrected`: the full corrected JSON for the document (must validate against `schema/document.schema.json`). `make feedback-ingest` rejects a row whose `corrected` is not valid JSON or fails the schema, and `make data` re-checks.
- **HF-3** `reviewer` is mandatory. Rows with a blank verdict are skipped, not rejected.
- **HF-4** An item already in `corrections.jsonl` is never queued or ingested twice.
- **HF-5** Corrections are added only to the training split. Corrected items that came from the test split are excluded from training and reported separately.
- **HF-6** The random-audit sample is the only unbiased estimate of live accuracy. Low-confidence items must never be used to estimate accuracy.
- **HF-7** At least 10% of reviewed items are reviewed by a second person; inter-reviewer agreement is reported.

## Section 4 — Promotion gate
A `challenger` model becomes `champion` only when document-level agreement on the random-audit sample is at least 90% over 100 or more reviewed documents, and no release gate in `SPEC.md` section 5 has regressed.

## Reports
`make feedback-ingest` prints: items added, agreement overall, agreement on the audit sample. Log these to the tracker with each retrain.

## Implemented in
`src/docextract/feedback.py`, tested in `tests/test_feedback.py`. Merge into data: `src/docextract/data.py` (`merge_corrections`, tested in `tests/test_data.py`). Open work: the optional review UI (T-15).
