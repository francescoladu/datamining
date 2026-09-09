# -----------------------------------------------------------------------------
# Project configuration
# -----------------------------------------------------------------------------

DATA_DIR = data
ZIP_FILE = $(DATA_DIR)/phishing_websites.zip
URL = https://archive.ics.uci.edu/static/public/327/phishing+websites.zip
STAMP_FILE = $(DATA_DIR)/.extracted

# Experiment 1: raw data
EXP1_TRAIN = $(DATA_DIR)/experiment_1_raw_train.csv
EXP1_TEST = $(DATA_DIR)/experiment_1_raw_test.csv

# Experiment 2: exact deduplication
EXP2_TRAIN = $(DATA_DIR)/experiment_2_standard_dedup_train.csv
EXP2_TEST = $(DATA_DIR)/experiment_2_standard_dedup_test.csv

# Experiment 3: weighted deduplication
EXP3_TRAIN = $(DATA_DIR)/train_cleaned.csv
EXP3_TEST = $(DATA_DIR)/test_cleaned.csv

REQUIREMENTS = requirements.txt
VENV = venv
PYTHON = $(VENV)/bin/python

# Internal experiment variable.
# Public targets set it automatically.
EXPERIMENT ?=

# Generated outputs
PREPROCESS_OUTPUT_DIR = code/data_preprocessing/outputs
MODEL_OUTPUT_DIR = code/model_selection/outputs
EXPLAIN_OUTPUT_DIR = code/explainability/outputs
FINAL_EVAL_OUTPUT_DIR = code/final_evaluation/outputs

export PYTHONPATH := code


.PHONY: \
	all \
	help \
	setup \
	dependencies \
	install1 \
	install2 \
	install3 \
	analyze1 \
	analyze2 \
	analyze3 \
	train1 \
	train2 \
	train3 \
	evaluate1 \
	evaluate2 \
	evaluate3 \
	explain \
	clean \
	_install \
	_analyze \
	_train \
	_evaluate \
	check-venv \
	check-experiment \
	check-selected-train-data \
	check-selected-data \
	check-exp3-data


# -----------------------------------------------------------------------------
# Default target
# -----------------------------------------------------------------------------

all: help


# -----------------------------------------------------------------------------
# Help
# -----------------------------------------------------------------------------

help:
	@echo ""
	@echo "Available commands:"
	@echo ""
	@printf "  %-22s %s\n" "COMMAND" "DESCRIPTION"
	@printf "  %-22s %s\n" "--------------------" "------------------------------------------------------------"
	@printf "  %-22s %s\n" "make help" "Show this help message."
	@printf "  %-22s %s\n" "make setup" "Create the virtual environment and install dependencies."
	@printf "  %-22s %s\n" "make dependencies" "Update dependencies in the existing virtual environment."
	@printf "  %-22s %s\n" "make install1" "Prepare Experiment 1: raw data."
	@printf "  %-22s %s\n" "make install2" "Prepare Experiment 2: exact deduplication."
	@printf "  %-22s %s\n" "make install3" "Prepare Experiment 3: weighted deduplication."
	@printf "  %-22s %s\n" "make analyze1" "Run EDA on Experiment 1 raw training data."
	@printf "  %-22s %s\n" "make analyze2" "Run EDA on Experiment 2 exact-deduplicated training data."
	@printf "  %-22s %s\n" "make analyze3" "Run EDA on Experiment 3 weighted-deduplicated training data."
	@printf "  %-22s %s\n" "make train1" "Run model selection for Experiment 1."
	@printf "  %-22s %s\n" "make train2" "Run model selection for Experiment 2."
	@printf "  %-22s %s\n" "make train3" "Run model selection for Experiment 3."
	@printf "  %-22s %s\n" "make evaluate1" "Evaluate Experiment 1 on its held-out test set."
	@printf "  %-22s %s\n" "make evaluate2" "Evaluate Experiment 2 on its held-out test set."
	@printf "  %-22s %s\n" "make evaluate3" "Evaluate Experiment 3 on its held-out test set."
	@printf "  %-22s %s\n" "make explain" "Run explainability for Experiment 3."
	@printf "  %-22s %s\n" "make clean" "Remove datasets and generated outputs; preserve the venv."
	@echo ""


# -----------------------------------------------------------------------------
# Environment setup
# -----------------------------------------------------------------------------

setup:
	@echo "Creating/updating virtual environment in ./$(VENV)..."
	@if [ ! -x "$(PYTHON)" ]; then \
		python3 -m venv "$(VENV)"; \
	fi
	@echo "Installing dependencies from $(REQUIREMENTS)..."
	@"$(PYTHON)" -m pip install --upgrade pip
	@"$(PYTHON)" -m pip install -r "$(REQUIREMENTS)"
	@echo "------------------------------------------------------------"
	@echo "Setup complete."
	@echo "To activate the environment manually, run:"
	@echo "  source $(VENV)/bin/activate"
	@echo "------------------------------------------------------------"


dependencies: check-venv
	@echo "Updating dependencies inside ./$(VENV)..."
	@"$(PYTHON)" -m pip install --upgrade pip
	@"$(PYTHON)" -m pip install -r "$(REQUIREMENTS)"


# -----------------------------------------------------------------------------
# Prerequisite checks
# -----------------------------------------------------------------------------

check-venv:
	@if [ ! -x "$(PYTHON)" ]; then \
		echo "ERROR: virtual environment not found."; \
		echo "Run 'make setup' first."; \
		exit 1; \
	fi


check-experiment:
	@case "$(EXPERIMENT)" in \
		1|2|3) ;; \
		*) \
			echo "ERROR: invalid internal experiment identifier '$(EXPERIMENT)'."; \
			exit 1; \
			;; \
	esac


check-selected-train-data: check-experiment
	@case "$(EXPERIMENT)" in \
		1) TRAIN_FILE="$(EXP1_TRAIN)" ;; \
		2) TRAIN_FILE="$(EXP2_TRAIN)" ;; \
		3) TRAIN_FILE="$(EXP3_TRAIN)" ;; \
	esac; \
	if [ ! -f "$$TRAIN_FILE" ]; then \
		echo "ERROR: training dataset for Experiment $(EXPERIMENT) was not found."; \
		echo "Expected:"; \
		echo "  $$TRAIN_FILE"; \
		echo ""; \
		echo "Run first:"; \
		echo "  make install$(EXPERIMENT)"; \
		exit 1; \
	fi


check-selected-data: check-experiment
	@case "$(EXPERIMENT)" in \
		1) \
			TRAIN_FILE="$(EXP1_TRAIN)"; \
			TEST_FILE="$(EXP1_TEST)"; \
			;; \
		2) \
			TRAIN_FILE="$(EXP2_TRAIN)"; \
			TEST_FILE="$(EXP2_TEST)"; \
			;; \
		3) \
			TRAIN_FILE="$(EXP3_TRAIN)"; \
			TEST_FILE="$(EXP3_TEST)"; \
			;; \
	esac; \
	if [ ! -f "$$TRAIN_FILE" ] || [ ! -f "$$TEST_FILE" ]; then \
		echo "ERROR: dataset files for Experiment $(EXPERIMENT) were not found."; \
		echo "Expected:"; \
		echo "  $$TRAIN_FILE"; \
		echo "  $$TEST_FILE"; \
		echo ""; \
		echo "Run first:"; \
		echo "  make install$(EXPERIMENT)"; \
		exit 1; \
	fi


check-exp3-data:
	@if [ ! -f "$(EXP3_TRAIN)" ] || [ ! -f "$(EXP3_TEST)" ]; then \
		echo "ERROR: Experiment 3 cleaned train/test data not found."; \
		echo "Run first:"; \
		echo "  make install3"; \
		exit 1; \
	fi


# -----------------------------------------------------------------------------
# Dataset download and extraction
# -----------------------------------------------------------------------------

$(ZIP_FILE):
	@mkdir -p "$(DATA_DIR)"
	@echo "Downloading Phishing Websites dataset from UCI..."
	@curl -fL -sS "$(URL)" -o "$(ZIP_FILE)"


# Keep the ZIP after extraction so the stamp remains newer than its dependency
# and repeated install commands do not trigger a new download/extraction.
$(STAMP_FILE): $(ZIP_FILE) | check-venv
	@echo "Extracting dataset to $(DATA_DIR)/..."
	@"$(PYTHON)" -m zipfile -e "$(ZIP_FILE)" "$(DATA_DIR)"
	@touch "$(STAMP_FILE)"


# -----------------------------------------------------------------------------
# Dataset preparation
# -----------------------------------------------------------------------------

install1:
	@$(MAKE) --no-print-directory _install EXPERIMENT=1

install2:
	@$(MAKE) --no-print-directory _install EXPERIMENT=2

install3:
	@$(MAKE) --no-print-directory _install EXPERIMENT=3


_install: check-venv check-experiment $(STAMP_FILE)
	@echo "------------------------------------------------------------"
	@echo "Preparing Experiment $(EXPERIMENT)..."
	@echo "------------------------------------------------------------"
	@"$(PYTHON)" scripts/clean_and_split.py "$(EXPERIMENT)"
	@$(MAKE) --no-print-directory check-selected-data EXPERIMENT="$(EXPERIMENT)"
	@echo "------------------------------------------------------------"
	@echo "Experiment $(EXPERIMENT) dataset ready."
	@echo "------------------------------------------------------------"


# -----------------------------------------------------------------------------
# Exploratory Data Analysis
#
# EDA always uses the training/development split of the selected experiment.
# Outputs are saved by data_preprocessing.main in a separate directory for
# each experiment.
# -----------------------------------------------------------------------------

analyze1:
	@$(MAKE) --no-print-directory _analyze EXPERIMENT=1

analyze2:
	@$(MAKE) --no-print-directory _analyze EXPERIMENT=2

analyze3:
	@$(MAKE) --no-print-directory _analyze EXPERIMENT=3


_analyze: check-venv check-selected-train-data
	@echo "------------------------------------------------------------"
	@echo "Running EDA - Experiment $(EXPERIMENT)"
	@echo "------------------------------------------------------------"
	@"$(PYTHON)" -m data_preprocessing.main "$(EXPERIMENT)"


# -----------------------------------------------------------------------------
# Model selection
# -----------------------------------------------------------------------------

train1:
	@$(MAKE) --no-print-directory _train EXPERIMENT=1

train2:
	@$(MAKE) --no-print-directory _train EXPERIMENT=2

train3:
	@$(MAKE) --no-print-directory _train EXPERIMENT=3


_train: check-venv check-selected-train-data
	@echo "------------------------------------------------------------"
	@echo "Running Model Selection - Experiment $(EXPERIMENT)"
	@echo "------------------------------------------------------------"
	@"$(PYTHON)" -m model_selection.main "$(EXPERIMENT)"


# -----------------------------------------------------------------------------
# Final held-out evaluation
# -----------------------------------------------------------------------------

evaluate1:
	@$(MAKE) --no-print-directory _evaluate EXPERIMENT=1

evaluate2:
	@$(MAKE) --no-print-directory _evaluate EXPERIMENT=2

evaluate3:
	@$(MAKE) --no-print-directory _evaluate EXPERIMENT=3


_evaluate: check-venv check-selected-data
	@echo "------------------------------------------------------------"
	@echo "Running Final Evaluation - Experiment $(EXPERIMENT)"
	@echo "------------------------------------------------------------"
	@"$(PYTHON)" -m final_evaluation.main "$(EXPERIMENT)"


# -----------------------------------------------------------------------------
# Explainability
#
# Explainability currently refers to Experiment 3 only.
# -----------------------------------------------------------------------------

explain: check-venv check-exp3-data
	@echo "------------------------------------------------------------"
	@echo "Running Explainability pipeline for Experiment 3..."
	@echo "Using SELECTED_RUN_NAME from code/shared/config.py"
	@echo "------------------------------------------------------------"
	@"$(PYTHON)" -m explainability.main


# -----------------------------------------------------------------------------
# Cleanup
# -----------------------------------------------------------------------------

clean:
	@echo "Removing dataset files, stamps, and generated outputs..."
	@rm -rf "$(DATA_DIR)"/*
	@rm -f "$(STAMP_FILE)"
	@rm -rf "$(PREPROCESS_OUTPUT_DIR)"
	@rm -rf "$(MODEL_OUTPUT_DIR)"
	@rm -rf "$(EXPLAIN_OUTPUT_DIR)"
	@rm -rf "$(FINAL_EVAL_OUTPUT_DIR)"
	@echo "Clean complete. Virtual environment preserved."
