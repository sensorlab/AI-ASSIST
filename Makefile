# HELP
# This will output the help for each task
# thanks to https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
ifeq ($(VERBOSE),1)
  Q =
else
  Q = @
endif

PORT := 5000

.PHONY: all
all: help

UV ?= /snap/bin/uv

.DEFAULT_GOAL := help


.PHONY: install
install: ## Install the virtual environment and install the pre-commit hooks
	$(Q)echo "🚀 Creating virtual environment using uv"
	$(Q)$(UV) python install
	$(Q)$(UV) venv --clear
	$(Q)$(UV) pip install -Ur pyproject.toml --all-extras --exact --editable .
	$(Q)$(UV) run pre-commit install


.PHONY: update
update: ## Update development dependencies.
	$(Q)echo "🚀 Update virtual environment using uv"
	$(Q)$(UV) pip install -Ur pyproject.toml --all-extras --refresh --exact --editable .
	$(Q)$(UV) lock --upgrade

.PHONY: upgrade
upgrade: install update ## Upgrade installation.
	$(Q)pre-commit autoupdate



.PHONY: help
help: ## This help.
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)


.PHONY: clean
clean: ## Clean up cache files.
	$(Q)find . -type d -name "__pycache__" -exec rm -rf {} +
	$(Q)find . -type f -name "*.pyc" -delete

	$(Q)find ./data -type f -name "*.pkl*" -print -delete
	$(Q)find ./data -type f -name "*.joblib*" -print -delete

	$(Q)echo "🚀 Cleanup complete!"


.PHONY: purge
purge: clean
	$(Q)find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +


.PHONY: debug
debug:
	@echo "PATH=$$PATH"
	@echo "VIRTUAL_ENV=$$VIRTUAL_ENV"
