# SPEC — Invoice and Receipt Extraction

## 1. Problem
Finance teams key data from receipts and invoices into their systems, or pay a large hosted model to do it. Both are costly per document, and the documents contain sensitive commercial data. Most documents are routine. We want a small in-house model to extract them, checks that prove an extraction is internally consistent, and a clear rule for when a person must look.

## 2. Users
- **Accounts-payable lead:** wants more documents processed with no manual keying, and a known error rate.
- **ML engineer (you):** wants a reproducible multimodal fine-tuning and evaluation pipeline to learn from.
- **Reviewer:** corrects flagged extractions.

## 3. Scope
In scope:
- Single-page receipt and invoice images, printed, English (and the languages present in the chosen datasets).
- Output: one JSON object per document following `schema/document.schema.json`.
- Field-level confidence proxy, validation checks, and a cascade to an expensive model or a human.

Out of scope: handwriting, multi-page documents, table-heavy purchase orders, posting to an accounting system, production deployment.

## 4. Functional requirements
- **FR-1** Given an image, return `{document, valid, checks, confidence, model_version, route}`.
- **FR-2** `document` validates against the schema: vendor name, date (ISO 8601), currency, line items (description, quantity, unit price, amount), subtotal, tax, total.
- **FR-3** Output is always parseable JSON. Use constrained decoding or repair-and-retry; report the raw parse-failure rate.
- **FR-4** Validation checks: schema valid; line-item amounts sum to the subtotal within tolerance; subtotal plus tax equals total within tolerance; date parses.
- **FR-5** `route` is `auto` when all checks pass and confidence is at least the threshold, else `review`.
- **FR-6** The pipeline runs unchanged on local, Colab, Kaggle and Databricks.
- **FR-7** Every prediction is logged for human review (`specs/05-human-feedback.md`).
- **FR-8** A zero-shot baseline with the expensive model is evaluated on the identical test documents.

## 5. Success metrics and release gates
Measured on the untouched test split. Baseline = zero-shot expensive vision model.

| Metric | Definition | Gate for release |
|---|---|---|
| Field F1 | Micro F1 over normalised (field path, value) pairs | At least baseline minus 2 points, and at least 0.90 on the receipt set |
| Key-field accuracy | Exact match on total, date, vendor after normalisation | Total at least 0.95; date at least 0.93 |
| Document exact match | All fields correct | Reported; no gate |
| JSON validity | Parses and passes schema | At least 0.99 |
| Arithmetic consistency | Share of documents passing both sum checks | Reported for gold and predictions |
| Straight-through rate | Share routed `auto` | At least 0.60 |
| Precision of `auto` | Share of `auto` documents with all key fields correct | At least 0.98 |
| Cost per 1,000 documents | From GPU-hour price and measured throughput | At most 20% of baseline |
| Throughput | Documents per minute on the target GPU | Reported |

The numeric gates are targets informed by published results on these datasets, not measured results of this project. Record the measured baseline in `docs/model_card.md` before training and change gates only through a `CHANGELOG.md` entry.

## 6. Business value model
Inputs, set in `configs/base.yaml` under `cost` and `business`:
- documents per month; manual handling minutes per document; clerk cost per hour;
- baseline cost per document; small-model cost per document; straight-through rate; precision of `auto`; cost of an undetected error.

Outputs produced by `evaluate` in `report.json`:
- monthly cost: fully manual, baseline model with review, small model with review;
- net monthly saving and the break-even volume.

These are projections from test-set results and stated volumes. Label them as projections wherever they are shown.

## 7. Non-functional requirements
- **NFR-1 Reproducibility:** fixed seeds; greedy decoding for evaluation; rerun field F1 within 1 point.
- **NFR-2 Compute:** LoRA training fits a single 24 GB GPU (16 GB with 4-bit loading) and finishes in under 4 hours on the receipt set; smoke run under 5 minutes.
- **NFR-3 Privacy:** no document leaves the environment except in the explicit baseline and escalation calls.
- **NFR-4 Traceability:** every reported number has a run ID.

## 8. Risks
| Risk | Mitigation |
|---|---|
| Public receipts differ from your invoices | State the limit; add your documents through the feedback loop |
| Datasets use different field schemas | One canonical schema and a documented mapper per dataset (spec 01) |
| Model invents plausible values | Arithmetic checks, key-field precision gate, review routing |
| Generative models give no native confidence | Use mean token log-probability plus checks; validate that it ranks errors (spec 06) |
| Value normalisation hides real errors | Normalisation rules are written in spec 03 and unit-tested |
| Model or dataset licence blocks commercial use | Record licences in the data and model cards before sharing results |

## 9. Open questions
- Which expensive model is the baseline, and at what price? Set in `configs/base.yaml: baseline`.
- Which small vision-language model? Default candidate in `PLAN.md`; confirm ID, licence and memory in T-02.
- Are line items required for the first release, or header fields only? Default: both, reported separately.
