# Spec 01 — Data

## 1. Goal
A deterministic, versioned dataset of document images paired with canonical JSON, with a manifest that proves what was used.

## 2. Sources
| Dataset | Config key | Use | To verify in T-02 |
|---|---|---|---|
| CORD | `data.datasets.cord` | Primary train, validation, test | Hub ID, revision, counts (about 800 / 100 / 100), label format, licence |
| SROIE | `data.datasets.sroie` | Second test set, header fields | Source location, counts, licence |
| Invoice set | `data.datasets.invoices` | Stretch | Choose one with a licence that fits your use |

## 3. Canonical schema and mappers
`schema/document.schema.json` is the contract. Each dataset has one mapper function `map_<dataset>(raw) -> dict`:
- copy fields with a direct equivalent;
- parse numbers with `schema.parse_number` (handles thousands separators and currency symbols);
- parse dates to ISO 8601; if ambiguous or unparseable, set `null`;
- set fields the source does not label to `null` and list them in the data card;
- never infer a value the source does not contain.

Fields that a dataset does not label are excluded from that dataset's metrics (`data.datasets.<name>.fields`).

## 4. Processing
1. Load the pinned revision.
2. Save images as files under `data/processed/images/<dataset>/<id>.<ext>` without resizing.
3. Apply the mapper; validate each label against the schema.
4. Keep the published splits where they exist; otherwise split by `data.seed`.
5. Apply `data.max_train_samples` and `data.max_eval_samples` when set (to the source data).
6. Merge corrections per spec 05 HF-5, after the caps so a correction is never capped away: `incorrect` rows with a valid corrected document relabel a train row with the same id, or add a new train row (image copied to `images/feedback/`); `correct` rows are used only with `feedback.include_confirmed`; `unsure` never. Ids from validation or test, images equal to a validation or test image (leak), duplicates of a train image and missing images are excluded and listed in the manifest.

## 5. Output files
- `data/processed/{train,validation,test}.jsonl`: `{"id": str, "dataset": str, "image": "relative/path", "document": {...}}`.
- `data/manifest.json` (committed): data version, source revisions, seed, row count and SHA-256 per split file, SHA-256 of the sorted list of image hashes, corrections applied (`corrections_applied`, ids) and excluded (`corrections_excluded`, by reason: validation, test, leak, duplicate_of_train, image_missing). With corrections applied the data version is `<data.version>-fb<fingerprint>`, where the fingerprint hashes the applied ids and documents, so it changes whenever the corrections change.

## 6. Checks (fail the command)
- Every label validates against the schema.
- No image hash appears in more than one split.
- Report, do not fail: share of gold documents passing the arithmetic checks. This is the ceiling for the same check on predictions.

## 7. Acceptance
- Two consecutive `make data` runs produce identical hashes.
- `tests/test_data.py` covers each mapper on a small fixture, without network access.
