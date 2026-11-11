# TASKS — Invoice and Receipt Extraction

Work top to bottom. Tick the box in the commit that completes the task. Each task lists the spec it implements and the check that proves it is done.

## Done in the scaffold
- [x] **T-00** Repository structure, specs, schema, configs, CI for GitHub and GitLab.
- [x] **T-01** `config.py`, `schema.py`, `metrics.py`, `evaluate.py`, `tracking.py`, `feedback.py`, `cli.py` with unit tests. Check: `make lint test`.

## Data
- [x] **T-02** Verify dataset and model IDs resolve; pin revisions; record licences in `docs/data_card.md` and `docs/model_card.md`. Spec 01 §2. Check: a script prints image and label counts per split.
- [x] **T-03** Implement the CORD mapper and `data.build_dataset`: images to `data/processed/images/`, canonical labels to `{train,validation,test}.jsonl`, `data/manifest.json`. Spec 01 §3–5. Check: `make data` twice gives identical hashes; `tests/test_data.py` covers the mapper on a fixture.
- [x] **T-04** Data checks: gold labels validate against the schema; gold arithmetic-consistency rate reported; no image shared between splits (perceptual or file hash). Spec 01 §6.
- [x] **T-05** SROIE mapper (header fields only) as a second test set. Spec 01 §2.

## Baseline
- [ ] **T-06** Add `baseline.py`: zero-shot extraction with the expensive model, schema in the prompt, on-disk cache, token and cost accounting. Spec 03 §4. Check: baseline run logged with `baseline=true`; numbers copied to `docs/model_card.md`. Status: code, Databricks notebook and local bundle/import tooling are done and tested (`platforms/databricks/baseline/README.md`); stays open until the notebook has been run on Databricks, `make baseline-import` has logged the run, and the numbers are in the model card.

## Training
- [ ] **T-07** Implement `train.train`: prompt template, image preprocessing, LoRA fine-tuning, resumable checkpoints, tracker logging. Spec 02. Check: `make smoke` finishes and logs a run; adapter files exist.
- [ ] **T-08** Validation loop computing field F1 on a fixed subset during training; keep the best adapter. Spec 02 §4.

## Inference and evaluation
- [ ] **T-09** Implement `predict.predict`: batch generation, constrained decoding or parse-repair-retry, confidence from token log-probabilities, checks, `route`, throughput measurement, predictions and feedback log. Spec 06 §2–4. Check: `make predict eval` writes `artifacts/report.json`; JSON validity reported.
- [x] **T-10** Threshold fitting on validation for the precision gate; routing quality on test; cascade with cached baseline outputs. Spec 06 §5. Check: threshold fitted without reading test (unit test).
- [ ] **T-11** Confidence validation: show that confidence ranks errors (area under the risk-coverage curve against random). Spec 06 §3. Status: tooling and tests done (`confidence.py`, reported by `make eval` as `confidence_aurc`, `confidence_auroc` and `report.confidence`); stays open until it has been run on real predictions and the verdict is in `docs/model_card.md`.
- [ ] **T-12** Results table and cost-versus-accuracy chart from tracked runs into `docs/results.md`. Spec 04.

## Release and feedback
- [x] **T-13** Merge `feedback/corrections.jsonl` into the training split; bump data version. Spec 05 HF-5.
- [ ] **T-14** Register the adapter, tag the release, fill in `docs/model_card.md`. `CONTRIBUTING.md` release checklist.

## Stretch
- [x] **T-15** Side-by-side review page (image beside editable JSON) or Label Studio export.
- [ ] **T-16** Quantisation comparison: 4-bit against full precision, accuracy and throughput.
- [ ] **T-17** Second adapter for invoices and an adapter router (mixture of LoRA experts) compared with one combined adapter.
- [ ] **T-18** Serve with an inference server that supports batching and LoRA adapters; endpoint `POST /extract`.
- [ ] **T-19** Light model comparison point (Donut-class).
