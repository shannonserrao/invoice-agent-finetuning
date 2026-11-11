# PLAN — Invoice and Receipt Extraction

Diagrams are in `docs/architecture.md`.

## 1. Approach
Fine-tune a small vision-language model with LoRA to read a document image and write the canonical JSON. Validate the JSON with a schema and arithmetic checks. Route documents that fail a check or fall below a confidence threshold to review or to the expensive model.

## 2. Components
| Module | Responsibility | Spec |
|---|---|---|
| `schema.py` | Canonical document type, normalisation, validation checks | 03 |
| `data.py` | Download pinned datasets, map each to the canonical schema, split, write manifest, merge corrections | 01 |
| `train.py` | LoRA fine-tuning, checkpoints, tracker logging | 02 |
| `predict.py` | Batch generation, JSON parsing and repair, confidence, routing, feedback log | 06 |
| `metrics.py` | Field F1, key-field accuracy, exact match, validity, cost model | 03 |
| `evaluate.py` | Join predictions with gold, compute metrics, routing quality, business projection | 03 |
| `tracking.py` | MLflow wrapper with JSONL fallback | 04 |
| `feedback.py` | Review queue export, correction ingest | 05 |
| `baseline.py` (to add) | Zero-shot extraction with the expensive model, cached responses | 03 |

## 3. Model choices
| Role | Default candidate | Notes |
|---|---|---|
| Primary | A 2–4B open vision-language model, e.g. `Qwen/Qwen2.5-VL-3B-Instruct` | Confirm ID, licence and memory in T-02; set in `model.base_id` |
| Light alternative | A sub-1B document model such as Donut | Cheaper comparison point; also useful for smoke runs |
| Baseline | Set in `baseline.model` | Zero-shot with the schema in the prompt |

## 4. Data
- Primary: CORD (receipts; about 800 train, 100 validation, 100 test images with structured labels).
- Second: SROIE (receipts; four header fields) for cross-dataset testing.
- Stretch: an invoice dataset, once licence and schema mapping are confirmed.
- Each dataset gets a mapper into the canonical schema; unmapped source fields are listed in the data card.

## 5. Training
LoRA on attention and projection layers; 4-bit loading when memory is short; gradient accumulation to an effective batch of 16; image resolution capped by `model.max_image_pixels`. Target text is the canonical JSON with fixed key order. Validation every N steps on a fixed subset using field F1.

## 6. Inference
Greedy decoding with a JSON-constrained grammar when the serving stack supports it; otherwise parse, repair once, retry once. Confidence is the mean token log-probability of the generated JSON, mapped to 0–1. Checks run on every output.

## 7. Routing
`auto` requires all checks to pass and confidence at or above a threshold fitted on validation to meet the precision gate. Everything else is `review`. The cascade option sends `review` documents to the expensive model first and to a person only if that output also fails checks.

## 8. Evaluation
`evaluate` reads `artifacts/predictions.jsonl` and `data/processed/test.jsonl`. It reports the metrics in `SPEC.md` section 5, a per-field table, routing quality and the business projection.

## 9. Order of work
Schema and mappers, baseline measurement, training, constrained inference, evaluation, routing, feedback merge, release. Details in `TASKS.md`.
