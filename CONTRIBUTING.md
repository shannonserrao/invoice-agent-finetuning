# Contributing, versioning and releases

## Remotes: GitHub and GitLab
The repository works on either host, or both as mirrors.
```bash
git init -b main
git remote add origin  git@github.com:<org>/invoice-receipt-extraction.git
git remote add gitlab  git@gitlab.com:<org>/invoice-receipt-extraction.git
git push -u origin main && git push gitlab main
```
- GitHub CI: `.github/workflows/ci.yml` (lint and tests on every push and pull request).
- GitLab CI: `.gitlab-ci.yml` (same jobs).
- Training never runs in CI. CI only checks code, configs and the metric functions.

## Branching
- `main` is always releasable. Protect it; require CI to pass.
- Feature branches: `feat/T-07-train-loop`. Experiment branches: `exp/<short-name>`.
- One task from `TASKS.md` per merge request where practical.

## What is versioned where
| Thing | Where | Identifier |
|---|---|---|
| Code, configs, specs | git | commit SHA, tag `vX.Y.Z` |
| Dataset | `data/manifest.json` in git; files outside git | `data-vN` plus SHA-256 per file |
| Experiments | MLflow | run ID |
| Models | MLflow Model Registry (optionally Hugging Face Hub) | `model-X.Y.Z` with aliases `challenger`, `champion` |
| Human corrections | `feedback/corrections.jsonl` in git | commit SHA |

Every run logs the git SHA, the data version and the resolved config, so any model can be traced to its exact inputs.

## Version numbers
- **Code** `vMAJOR.MINOR.PATCH`: MAJOR for a changed output schema or label set, MINOR for a new capability, PATCH for fixes.
- **Data** `data-vN`: bump whenever the manifest changes (new source, new split, merged corrections).
- **Model** `model-X.Y.Z`: X follows the code MAJOR, Y increments for each retrain on a new data version, Z for a re-run with the same data.

## Release checklist
1. `make lint test` passes.
2. `make eval` on the test split; report attached to the run.
3. The release gates in `SPEC.md` section 5 are met, or the exception is written in `CHANGELOG.md`.
4. `docs/model_card.md` updated with run ID and numbers.
5. `make release VERSION=x.y.z` (tags git, registers the model, sets alias `challenger`).
6. Promote to `champion` only after the human review in `specs/05-human-feedback.md` section 4.

## Large files
Do not use git for data or weights. If you want file-level data versioning beyond the manifest, add DVC or git-LFS; this is recorded as an open option in `docs/decisions/0001-tooling.md`.
