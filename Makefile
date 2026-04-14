# ---- Config ----
PYTHON := .venv/bin/python
UV := uv

# ---- Default ----
.DEFAULT_GOAL := help

# ---- Setup ----
.PHONY: init
init: ## Create environment and install dependencies
	$(UV) sync

.PHONY: reset
reset: ## Remove env and reinstall everything
	rm -rf .venv uv.lock
	$(UV) lock
	$(UV) sync

# ---- Environment ----
.PHONY: shell
shell: ## Activate virtual environment
	@echo "Run: source .venv/bin/activate"

.PHONY: python
python: ## Run Python inside uv env
	$(UV) run python

# ---- Dependencies ----
.PHONY: add
add: ## Add dependency (usage: make add pkg=<package>)
	$(UV) add $(pkg)

.PHONY: remove
remove: ## Remove dependency (usage: make remove pkg=<package>)
	$(UV) remove $(pkg)

.PHONY: lock
lock: ## Regenerate lock file
	$(UV) lock

.PHONY: sync
sync: ## Sync environment with lock file
	$(UV) sync

# ---- Quality ----
.PHONY: format
format: ## Format code with black
	$(UV) run black .

.PHONY: lint
lint: ## Lint code with ruff
	$(UV) run ruff check .

.PHONY: lint-fix
lint-fix: ## Auto-fix lint issues
	$(UV) run ruff check . --fix

# ---- Testing ----
.PHONY: test
test: ## Run tests
	$(UV) run pytest

.PHONY: test-verbose
test-verbose: ## Run tests (verbose)
	$(UV) run pytest -v

# ---- Notebook / Dev ----
.PHONY: notebook
notebook: ## Start Jupyter
	$(UV) run python -m ipykernel install --user --name=uv-env
	$(UV) run jupyter notebook

# ---- Cleanup ----
.PHONY: clean
clean: ## Remove cache files
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# ---- Help ----
.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "%-20s %s\n", $$1, $$2}'