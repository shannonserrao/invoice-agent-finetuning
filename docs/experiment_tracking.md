# Experiment tracking guide

Short how-to. The requirements are in `specs/04-experiment-tracking.md`.

```bash
make train                                  # opens a run automatically
mlflow ui --backend-store-uri ./mlruns      # browse at http://localhost:5000
```

## In code
```python
from docextract.tracking import Tracker

with Tracker(cfg, run_name="dev-lr-sweep", stage="dev") as t:
    t.log_params({"train.lr": 2e-5})
    t.log_metrics({"loss": 0.41}, step=100)
    t.log_artifact("artifacts/report.json")
```

## Per platform
| Platform | Store | Set by |
|---|---|---|
| Local | `./mlruns` | default |
| Colab | Google Drive folder | `configs/platform/colab.yaml` |
| Kaggle | `/kaggle/working/mlruns` (download with kernel output) | `configs/platform/kaggle.yaml` |
| Databricks | Workspace MLflow | `configs/platform/databricks.yaml` |

## Run hygiene
- One hypothesis per run. Write it in the run description.
- Never delete failed runs; tag them `status=failed` with the reason.
- Record each sweep's conclusion as one line in `CHANGELOG.md`.
