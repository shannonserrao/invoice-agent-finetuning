# ADR 0001 — Tooling choices

Status: accepted (initial scaffold)

| Decision | Choice | Why | Alternative left open |
|---|---|---|---|
| Config | YAML with `extends` and `--set` overrides | No extra dependency, readable diffs | Hydra |
| Tracking | MLflow | Runs locally and is native on Databricks | Weights & Biases |
| Data versioning | Manifest with SHA-256 plus pinned dataset revision | Enough for public datasets, zero infrastructure | DVC, git-LFS |
| Model registry | MLflow registry with aliases | Same tool as tracking | Hugging Face Hub |
| Human review | CSV queue plus JSONL corrections in git | Reviewable in a spreadsheet, auditable | Label Studio, Argilla |
| CI | Lint and unit tests only | Training in CI is slow and costly | Scheduled GPU evaluation job |

Add a new ADR file for any change to these.
