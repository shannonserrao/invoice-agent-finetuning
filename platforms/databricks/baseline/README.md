# Baseline on Databricks (task T-06)

The baseline is the expensive model reading the same documents with the same prompt, with no fine-tuning. The small
model is judged against it, so it has to be run once, on the held-out documents, and recorded.

This folder holds the Databricks notebook. The workflow has three parts: **build** a bundle locally, **run** the
notebook on Databricks, **import** the results back into the repo.

```
 local repo                     Databricks                         local repo
 make baseline-bundle  ──►  upload zip + run notebook  ──►  make baseline-import
 (documents, no labels)      (calls the expensive model)      (validate, install, log run)
```

No GPU is needed anywhere in this task. The model runs behind an API; the notebook only sends requests.

## What leaves your machine
The images of the documents in one split (100 CORD test receipts by default) and the prompt. **Gold labels are not
included**: the bundle's `<split>.jsonl` has only `id`, `dataset` and `image`. The images are sent to the model
provider, which is the explicit baseline call that SPEC NFR-3 allows. Do not use this with documents you may not send.

## Part A. Build the bundle (local)

Prerequisite: `make data` has been run (the processed splits and `data/manifest.json` exist).

```bash
make baseline-bundle
```

It writes `artifacts/baseline_bundle_test.zip` and prints its size and SHA-256. Expect about 235 MB for the 100 CORD
test images. The zip holds the notebook, the package code, `configs/prompt.txt`, the schema, the images and ids, and
`bundle.json` (split, document count, prompt hash, data version). It is reproducible: the same inputs give the same
zip.

To bundle the second test set instead (361 SROIE receipts; only if you want that baseline too), call the CLI
directly: `python -m docextract.cli baseline-bundle --config configs/base.yaml --set predict.split=test_sroie`.

## Part B. Run on Databricks

### B1. Import the notebook
Workspace → your folder → **Import** → upload `platforms/databricks/baseline/baseline_notebook.py`.
Databricks recognises the `# Databricks notebook source` header and opens it as a notebook with cells.

### B2. Upload the bundle to a Volume
Catalog → pick (or create) a Volume, for example `main.default.baseline` → **Upload to this volume** →
`baseline_bundle_test.zip`. Its path is `/Volumes/main/default/baseline/baseline_bundle_test.zip`. Also decide an
output folder on the same Volume, for example `/Volumes/main/default/baseline/out` (the notebook creates it).

### B3. Choose a backend and set it up

**Backend `anthropic`: an organisation-provisioned API key.** The key goes in a secret scope, never in the notebook.
With the Databricks CLI (current syntax):
```bash
databricks secrets create-scope baseline
databricks secrets put-secret baseline anthropic-key      # it prompts for the value; paste the org key
```
Widgets 5 and 6 are then `baseline` and `anthropic-key`. Your compute needs outbound internet access to the Anthropic
API. If your workspace blocks outbound traffic, use the second backend.

**Backend `databricks`: a Claude model-serving endpoint in your workspace.** Open **Serving**, find a Claude
endpoint that accepts images (a Foundation Model endpoint or an external-model endpoint you or your admin created),
and copy its name into widget 7. No key is needed; access follows your workspace permissions.

### B4. Compute
Attach **serverless** compute, or a small all-purpose cluster on Databricks Runtime 14 or newer. A CPU node is enough.

### B5. Fill the widgets (top of the notebook)

| # | Widget | Value |
|---|---|---|
| 1 | Bundle zip path | `/Volumes/.../baseline_bundle_test.zip` |
| 2 | Output folder | `/Volumes/.../out` |
| 3 | Backend | `anthropic` or `databricks` |
| 4 | Model | `claude-opus-5-5` (anthropic backend) |
| 5, 6 | Secret scope, key | as created in B3 (anthropic backend) |
| 7 | Serving endpoint | endpoint name (databricks backend) |
| 8, 9 | USD per 1M input / output tokens | your real price. Defaults are the public list price of `claude-opus-5-5` (4.0 / 20.0) at the time of writing. Through a Databricks endpoint the price is in DBUs, so convert and enter the equivalent. |
| 10 | Documents to run | **`3` first (dry run)**, then `0` for all |
| 11 | Parallel requests | `4`; lower it if you see rate-limit errors |

### B6. Dry run, then the full run
1. Run all cells with widget 10 = `3`. Check the table: `parsed` should be true, `stop_reason` normal
   (`end_turn` or `stop`), tokens non-zero. The printout shows a projected cost for all documents.
2. Set widget 10 to `0` and run all cells again. The 3 dry-run answers are reused, not paid for twice.
3. A restart or error part-way is safe: answers are cached on the Volume and a rerun continues where it stopped.

The last cell writes `baseline_results_test.zip` into the output folder (and `baseline_results_test_dry_run.zip`
for the dry run, which cannot be imported).

### B7. Download the results
Catalog → your Volume → `out` → `baseline_results_test.zip` → **Download**.

## Part C. Bring the results back into the repo (local)

Put the zip anywhere in your checkout, for example `artifacts/`, then:

```bash
make baseline-import RESULTS=artifacts/baseline_results_test.zip
```

The import **validates** before it writes anything:
- it is a full run, not a dry run;
- the document ids equal the local split exactly;
- the prompt hash equals the local prompt and schema (so a prompt edited after bundling is caught);
- the data version equals `data/manifest.json`;
- token totals in the summary match the predictions.

It then **installs** the files where the rest of the pipeline looks for them:

| File | Where | Used by |
|---|---|---|
| `baseline_predictions.jsonl` | `artifacts/` | `make eval` (baseline columns, cascade) |
| `baseline_summary.json` | `artifacts/` | `make eval` (baseline cost per document), model card |
| response cache | `artifacts/baseline_cache/` | later local reruns, free of charge |

(For another split, file names carry the split, for example `baseline_predictions_test_sroie.jsonl`.) It also
scores the baseline against the gold labels, **logs a tracked run tagged `baseline=true`**, and prints the run ID,
the scores and the cost per document.

If a file already exists with different content the import stops; pass `--force`
(`python -m docextract.cli baseline-import --config configs/base.yaml --results ... --force`) to replace it.

### Record the numbers
`CLAUDE.md` rule: no number in the model card without a tracked run. From the import output and
`artifacts/baseline_summary.json`, fill the **Baseline (zero-shot)** column of `docs/model_card.md`:

| Model card row | Source |
|---|---|
| Field F1 | `baseline_field_f1` |
| Total / Date / Vendor accuracy | `baseline_accuracy_total`, `baseline_accuracy_date`, `baseline_accuracy_vendor_name` |
| Document exact match | `baseline_document_exact_match` |
| JSON validity (unrepaired) | `raw_json_validity` in the summary |
| Cost per 1,000 documents | `baseline_usd_per_1k` in the summary |
| Base model and run ID | `model` in the summary, and the run ID printed by the import |

Then tick **T-06** in `TASKS.md`, add a `CHANGELOG.md` line, and commit the docs (the data, predictions and cache are
git-ignored).

## Troubleshooting
| Symptom | Cause and fix |
|---|---|
| Widget check fails | The message names the widget. Paths must be Volume paths that exist. |
| `401` or authentication error | Wrong secret scope or key name, or the key lacks access to the model. |
| `429` or rate-limit errors | Lower widget 11. The SDK also retries on its own. |
| Connection errors on the `anthropic` backend | Compute has no outbound internet. Use the `databricks` backend. |
| `images_resized` is above 0 | Expected: images over 3.5 MB or wider than 1568 px are shrunk before sending, because the API rejects images over 5 MB and downsizes large ones itself. For the 100 CORD test images this applies to 21 of them (12 are over 5 MB). |
| `refusals` above 0 | The provider declined a document. Those score as unparsed; the count is reported, not hidden. |
| Import says "ids differ" | The bundle was built from a different split or data version. Rebuild and rerun. |
| Import says "prompt changed" | `configs/prompt.txt` or the schema changed after bundling. Rebuild and rerun; never edit the prompt after seeing test results. |
| `models_served` shows more than one model | A fallback model answered some documents. Mention it in the model card or rerun with `refusal_fallback: false`. |

## Rules to keep
- The prompt is frozen by hash before the test run. Tune it on validation only
  (`--set predict.split=validation` when building the bundle), never on test.
- Do not commit the bundle, results zip, predictions or cache.
