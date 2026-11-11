# Invoice and Receipt Extraction

Turn receipt and invoice images into validated JSON with a LoRA-fine-tuned small vision-language model, escalating only uncertain documents to an expensive LLM.

**Status:** work in progress. Data pipeline (CORD and SROIE), baseline harness, training code, routing, confidence validation, feedback merge and the review page are implemented and unit-tested. Training has not yet been run end to end, inference (`predict.py`) is still a stub, and the baseline has not been run. No results exist yet; see `TASKS.md`.

## What this proves
A proof of concept with three measurable claims, each tested against a zero-shot large-model baseline on the same held-out documents:
1. **Accuracy:** a LoRA-fine-tuned small vision-language model reaches field-level accuracy close to or above the baseline on receipts.
2. **Cost and privacy:** it does so at a fraction of the cost per 1,000 documents, with documents staying in-house.
3. **Straight-through processing:** validation checks plus confidence decide which documents need no human touch, and the share that pass is measured.

Targets are in `SPEC.md` section 5. They are hypotheses until a tracked run confirms them.

## Start here
| Document | Purpose |
|---|---|
| `SPEC.md` | Requirements, metrics, release gates |
| `PLAN.md` | Architecture and technical choices |
| `docs/architecture.md` | Diagrams |
| `TASKS.md` | Ordered implementation tasks |
| `BUILD.md` | Run on local, Colab, Kaggle, Databricks |
| `CLAUDE.md` | Rules for AI-assisted development |
| `CONTRIBUTING.md` | Git, GitHub and GitLab, versioning, releases |
| `specs/` | Component specs |
| `schema/document.schema.json` | The output contract |
| `docs/` | Data card, model card, tracking, reviewer guide, decisions |

## Quick start
```bash
make setup && make lint test     # works now
make data train predict eval     # works after T-03 to T-09
```

## Layout
```
SPEC.md PLAN.md TASKS.md BUILD.md CLAUDE.md CONTRIBUTING.md CHANGELOG.md
configs/         base, smoke and per-platform YAML
schema/          JSON schema of the extracted document
specs/           01-data ... 06-serving-and-routing
docs/            architecture, data card, model card, guides, decisions
src/docextract/  config, schema, data, train, predict, metrics, evaluate, tracking, feedback, cli
tests/           unit tests for the implemented modules
platforms/       Colab, Kaggle and Databricks launchers
data/ artifacts/ feedback/
```
