.DEFAULT_GOAL := help
SHELL := /bin/bash

PYTHON_VERSION ?= 3.11
CONFIG ?= configs/attention.yaml
RUN ?=
BEAM ?= 3
UV ?= uv
RUN_PY := $(UV) run

.PHONY: help setup lint format test test-all data features train eval export app clean-cache

help: ## List targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

setup: ## Create the virtualenv from uv.lock and install pre-commit hooks
	$(UV) sync --python $(PYTHON_VERSION) --frozen
	-$(RUN_PY) pre-commit install

lint: ## ruff format --check, ruff check, mypy (strict on src/)
	$(RUN_PY) ruff format --check .
	$(RUN_PY) ruff check .
	$(RUN_PY) mypy src

format: ## Auto-format and auto-fix
	$(RUN_PY) ruff format .
	$(RUN_PY) ruff check --fix .

test: ## Unit tests with coverage (fast, CPU)
	$(RUN_PY) pytest -m "not integration" --cov=captioner --cov-report=term-missing --cov-fail-under=80

test-all: ## Unit + integration tests
	$(RUN_PY) pytest --cov=captioner --cov-report=term-missing --cov-fail-under=80

data: ## Download + verify Flickr8k, build manifest and vocab
	$(RUN_PY) captioner download --config $(CONFIG)
	$(RUN_PY) captioner prepare --config $(CONFIG)

features: ## Extract frozen ResNet-50 features to HDF5
	$(RUN_PY) captioner extract --config $(CONFIG)

train: ## Train a model: make train CONFIG=configs/attention.yaml
	$(RUN_PY) captioner train --config $(CONFIG)

eval: ## Evaluate a run on the test split: make eval RUN=<run_id> (default: latest)
	$(RUN_PY) captioner evaluate --config $(CONFIG) $(if $(RUN),--run $(RUN),)

export: ## Export a run to artifacts/release: make export RUN=<run_id>
	$(RUN_PY) captioner export --config $(CONFIG) $(if $(RUN),--run $(RUN),)

app: ## Launch the Streamlit demo
	$(RUN_PY) streamlit run app/streamlit_app.py

clean-cache: ## Remove Python / tool caches
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
