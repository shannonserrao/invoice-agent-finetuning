PY ?= python
CONFIG ?= configs/base.yaml
CLI = $(PY) -m docextract.cli

.PHONY: setup lint test data baseline baseline-bundle baseline-import fit-threshold train eval predict feedback-export review-page feedback-ingest smoke release clean

setup:
	$(PY) -m pip install -e ".[dev,train]"

lint:
	$(PY) -m ruff check src tests

test:
	$(PY) -m pytest -q

data:
	$(CLI) data --config $(CONFIG)

fit-threshold:
	$(CLI) fit-threshold --config $(CONFIG)

baseline-bundle:
	$(CLI) baseline-bundle --config $(CONFIG)

baseline-import:
	@test -n "$(RESULTS)" || (echo "usage: make baseline-import RESULTS=path/to/baseline_results.zip" && exit 1)
	$(CLI) baseline-import --config $(CONFIG) --results $(RESULTS)

baseline:
	$(CLI) baseline --config $(CONFIG)

train:
	$(CLI) train --config $(CONFIG)

predict:
	$(CLI) predict --config $(CONFIG)

eval:
	$(CLI) evaluate --config $(CONFIG)

feedback-export:
	$(CLI) feedback-export --config $(CONFIG)

review-page:
	$(CLI) review-page --config $(CONFIG)

feedback-ingest:
	$(CLI) feedback-ingest --config $(CONFIG) --reviewed feedback/reviewed.csv

smoke:
	$(CLI) data --config configs/smoke.yaml
	$(CLI) train --config configs/smoke.yaml
	$(CLI) predict --config configs/smoke.yaml
	$(CLI) evaluate --config configs/smoke.yaml

release:
	@test -n "$(VERSION)" || (echo "usage: make release VERSION=x.y.z" && exit 1)
	git tag -a v$(VERSION) -m "release v$(VERSION)"
	@echo "Tagged v$(VERSION). Push with: git push --tags. Then register the model (TASKS.md T-12)."

clean:
	rm -rf artifacts/* .pytest_cache .ruff_cache
