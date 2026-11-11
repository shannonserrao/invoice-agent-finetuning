# Spec 06 — Inference, checks, routing and serving

## 1. Goal
Generate valid JSON for each document, attach evidence of reliability, and decide whether a person must look.

## 2. Generation (`predict.py`)
- Load base model plus adapter from `artifacts/model/`.
- Greedy decoding, `predict.max_new_tokens`.
- Preferred: grammar- or schema-constrained decoding when the serving stack supports it (`predict.constrained: true`).
- Fallback: parse; on failure apply one deterministic repair (trim to the outermost braces, close brackets); on second failure retry generation once; then give up with `document: null`.
- `raw_valid_json` records whether the first, unrepaired output parsed. This is the number reported as JSON validity.

## 3. Confidence
- `confidence = exp(mean token log-probability)` over the generated tokens.
- This is a ranking signal, not a probability. T-11 must show it ranks errors better than random before it is used for routing. `confidence.py` computes the area under the risk-coverage curve (AURC) against a random ordering (whose AURC equals the error rate), AUROC, and a permutation p-value; the claim holds only if AURC is lower than random and p < 0.05. If it does not, route on checks only and say so in the model card.

## 4. Checks
`schema.run_checks(document, tolerance)` returns `{schema_valid, items_sum_ok, total_ok, date_ok}`. A check whose inputs are absent is `null` and treated as not passed for routing, unless `routing.allow_missing_checks` lists it.

## 5. Routing
- `auto` if every required check passes and confidence is at least the threshold; else `review`.
- Threshold fitting (`routing.select: validation`): on validation, among documents passing checks, choose the lowest threshold at which key-field precision meets `routing.target_precision`. Save `artifacts/threshold.json`. Command: `make fit-threshold`, which reads only `data/processed/validation.jsonl` and `artifacts/predictions_validation.jsonl` (produced by `predict` with `--set predict.split=validation`). If no threshold reaches the target, `feasible` is false and the threshold is 1.01, so nothing goes to `auto`.
- Cascade option (`routing.escalate_to_baseline: true`): `review` documents go to the expensive model; its output is accepted only if it passes the same checks; otherwise the document goes to the human queue. `evaluate` reports this as `cascade` (small-model auto, accepted from baseline, to human, precision of accepted, baseline calls and cost) using the cached baseline outputs.

## 6. Performance measurement
Documents per minute over the test set at `predict.batch_size`; seconds per document p50 and p95; device, precision and image cap recorded.

## 7. Serving (stretch, T-18)
`POST /extract` with an image returns the FR-1 payload. Use an inference server with batching and adapter support. No authentication in the proof of concept; do not expose it publicly.

## 8. Acceptance
- The routing decision is a pure function of checks, confidence and threshold (unit test in the scaffold).
- Repair never changes a value, only structure (unit test to add with T-09).
