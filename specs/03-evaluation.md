# Spec 03 — Evaluation

## 1. Goal
One command produces every number in `SPEC.md` section 5 for the small model, the baseline and the routed system, from files on disk.

## 2. Inputs
- `artifacts/predictions.jsonl`: `{"id", "document" (object or null), "raw_valid_json" (bool), "confidence", "seconds"}`.
- `data/processed/test.jsonl`: gold documents.
- Optional `artifacts/baseline_predictions.jsonl`, same format plus `cost_usd`.

## 3. Normalisation (implemented in `schema.py`)
Applied identically to gold and prediction before comparison:
- strings: Unicode NFKC, lower-case, collapse whitespace, strip surrounding punctuation;
- numbers: parsed to float and rounded to 2 decimals; `"1,234.50"` and `1234.5` are equal;
- dates: compared as ISO strings;
- `null`, missing and empty string are the same (absent).
- Scope per dataset: only `data.datasets.<name>.fields` are scored. A field listed in `unscored_if_null` is skipped for a document whose gold value is null (unknown, not absent).

## 4. Metrics (implemented in `metrics.py`)
| Name | Function | Definition |
|---|---|---|
| Field F1 | `field_f1` | Flatten each document to a multiset of (path, value). Header paths are field names. Line items are matched to gold items greedily by description similarity, then compared field by field; unmatched items count as misses or extras. Micro-averaged over documents. |
| Per-field accuracy | `per_field_accuracy` | Exact match per header field over documents where gold has a value |
| Document exact match | `exact_match` | Flattened multisets identical |
| JSON validity | from `raw_valid_json` and schema check | |
| Arithmetic consistency | `schema.run_checks` | Line items sum to subtotal; subtotal plus tax equals total; tolerance `eval.amount_tolerance` |
| Routing quality | `routing_quality` | Straight-through rate; precision of `auto` on key fields |
| Cost per 1,000 | `cost_per_1k` | GPU-hour price and measured documents per second |
| Monthly projection | `business_projection` | From the `business` config |

## 5. Baseline protocol
- The prompt contains the schema and the same instruction as the fine-tuned model's prompt.
- Frozen and committed before the first test run; responses cached by model, prompt hash and image hash.
- Unparseable output scores zero for that document and is counted in the validity rate.
- Backends: an API key (`baseline.backend: anthropic`, for example an organisation key in a Databricks secret scope) or a Databricks serving endpoint (`databricks`). Request settings (backend, effort, token limit, image preparation) are part of the cache key.
- Images over `baseline.max_image_bytes` or `baseline.max_image_edge` are shrunk deterministically before sending, because the API rejects images over 5 MB; the count is reported as `images_resized`.
- Run on Databricks: `make baseline-bundle` packs the documents of one split (ids and images only, no gold labels), the prompt and schema, and the notebook `platforms/databricks/baseline/baseline_notebook.py`. The notebook writes one results zip. `make baseline-import` rejects a dry run, a different set of ids, a changed prompt or schema, a different data version, or inconsistent token totals; otherwise it installs `artifacts/baseline_predictions.jsonl` (`..._<split>` for non-test splits), `baseline_summary.json` and the response cache, scores the baseline against the gold labels, and logs a run tagged `baseline=true`. `make eval` then reads the baseline predictions and the baseline cost per document from those files.

## 6. Statistical care
- 95% bootstrap confidence intervals over documents for field F1 and key-field accuracy. With about 100 test documents the intervals are wide; state them.
- Paired bootstrap for small model against baseline.
- Report the second dataset separately; do not pool.

## 7. Report
`artifacts/report.json`: `{"metrics": {...}, "per_field": {...}, "routing": {...}, "business": {...}, "meta": {...}}`.

## 8. Rules
- `evaluate` never fits a threshold. It reads `artifacts/threshold.json` (fitted on validation) or the config value.
- Prediction ids must equal gold ids exactly.

## 9. Acceptance
`tests/test_schema.py`, `tests/test_metrics.py` and `tests/test_evaluate.py` pass; hand-computed examples match.
