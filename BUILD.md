# BUILD.md — how to build and run Invoice and Receipt Extraction

All platforms run the same commands; only the bootstrap differs. Python 3.10 or newer.

## 1. Local (CPU or your own GPU)
```bash
git clone <your-remote>/invoice-receipt-extraction.git && cd invoice-receipt-extraction
python -m venv .venv && source .venv/bin/activate
make setup              # pip install -e ".[dev,train]"
cp .env.example .env    # fill in tokens you need
make lint test          # should pass on a fresh clone
make smoke              # tiny end-to-end run on CPU (after T-06)
make data train eval    # full run
```

## 2. Google Colab
1. Open a new notebook, set the runtime to GPU.
2. Run the cell in `platforms/colab_bootstrap.py` (paste it, or `%run` it after cloning).
3. It clones the repo, installs it, mounts Drive for `artifacts/` and runs `make data train eval` with `configs/platform/colab.yaml`.

Colab sessions are temporary. Artifacts and the MLflow store are written to Drive so a disconnected session can resume from the last checkpoint.

## 3. Kaggle
```bash
pip install kaggle
kaggle kernels push -p platforms/kaggle     # uses kernel-metadata.json
kaggle kernels status <user>/invoice-receipt-extraction
kaggle kernels output <user>/invoice-receipt-extraction -p artifacts/kaggle
```
Edit `platforms/kaggle/kernel-metadata.json` first: set your username and attach the repo as a dataset or enable internet. Kaggle caps a session at a fixed number of hours, so `configs/platform/kaggle.yaml` saves checkpoints frequently.

## 4. Databricks
```bash
pip install databricks-cli
databricks bundle validate -t dev    # from platforms/databricks/
databricks bundle deploy   -t dev
databricks bundle run      -t dev train_job
```
MLflow tracking is native on Databricks; `configs/platform/databricks.yaml` sets `tracking.uri: databricks`. Fill in the cluster node type and workspace host in `platforms/databricks/databricks.yml` before deploying. The bundle file is a starting template; validate it against your workspace.

## 5. Make targets
| Target | What it does | Spec |
|---|---|---|
| `make setup` | Install package and dev tools | — |
| `make lint` / `make test` | Ruff and pytest | — |
| `make data` | Download, split, write manifest | `specs/01-data.md` |
| `make train` | Fine-tune and log to the tracker | `specs/02-training.md` |
| `make eval` | Score predictions, write report | `specs/03-evaluation.md` |
| `make predict` | Batch inference with confidence | `specs/06-serving-and-routing.md` |
| `make feedback-export` | Write the human review queue | `specs/05-human-feedback.md` |
| `make review-page` | Build the local side-by-side review page from the queue | `specs/05-human-feedback.md` |
| `make feedback-ingest` | Merge reviewed corrections | `specs/05-human-feedback.md` |
| `make smoke` | Tiny CPU run of the whole pipeline | — |
| `make release VERSION=x.y.z` | Tag code and register model | `CONTRIBUTING.md` |

## 6. Configuration
`configs/base.yaml` holds every parameter. A platform file extends it:
```bash
python -m docextract.cli train --config configs/platform/colab.yaml --set train.epochs=1
```
`--set key.sub=value` overrides any single value and is recorded with the run.

## 7. Outputs
```
data/            raw and processed data (git-ignored) + manifest.json (committed)
artifacts/       checkpoints, predictions, reports (git-ignored)
mlruns/          local MLflow store (git-ignored)
feedback/        review queue and corrections (corrections.jsonl is committed)
```
