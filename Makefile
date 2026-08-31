# -----------------------------------------------------------------------------
# Project configuration
# -----------------------------------------------------------------------------
DATA_DIR = data
ZIP_FILE = $(DATA_DIR)/phishing_websites.zip
URL = https://archive.ics.uci.edu/static/public/327/phishing+websites.zip
STAMP_FILE = $(DATA_DIR)/.extracted
CLEANED_TRAIN = $(DATA_DIR)/train_cleaned.csv
CLEANED_TEST = $(DATA_DIR)/test_cleaned.csv

REQUIREMENTS = requirements.txt
VENV = venv
PYTHON = $(VENV)/bin/python

# Generated outputs
PREPROCESS_OUTPUT_DIR = code/data_preprocessing/outputs
MODEL_OUTPUT_DIR = code/model_selection/outputs
EXPLAIN_OUTPUT_DIR = code/explainability/outputs
FINAL_EVAL_OUTPUT_DIR = code/final_evaluation/outputs

export PYTHONPATH := code

.PHONY: all help setup dependencies install analyze train evaluate explain clean \
        check-venv check-data

# Default target: show available commands.
all: help

help:
	@echo "Available commands:"
	@echo "  make setup        - Create the virtual environment and install dependencies"
	@echo "  make dependencies - Update dependencies inside the existing virtual environment"
	@echo "  make install      - Download, extract, clean, and split the dataset"
	@echo "  make analyze      - Run preprocessing validation, statistics, and EDA"
	@echo "  make train        - Run nested CV, model selection, and final hyperparameter search"
	@echo "  make evaluate     - Evaluate the run selected in code/shared/config.py on the test set"
	@echo "  make explain      - Explain the same selected run with global importance and local SHAP"
	@echo "  make clean        - Remove datasets and generated outputs (keeps the virtual environment)"

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
# These targets validate state only; they never run another project phase.
# -----------------------------------------------------------------------------
check-venv:
	@if [ ! -x "$(PYTHON)" ]; then \
		echo "ERROR: virtual environment not found. Run 'make setup' first."; \
		exit 1; \
	fi

check-data:
	@if [ ! -f "$(CLEANED_TRAIN)" ] || [ ! -f "$(CLEANED_TEST)" ]; then \
		echo "ERROR: cleaned train/test data not found. Run 'make install' first."; \
		exit 1; \
	fi

# -----------------------------------------------------------------------------
# Dataset installation
# -----------------------------------------------------------------------------
install: check-venv $(STAMP_FILE)
	@if [ ! -f "$(CLEANED_TRAIN)" ] || [ ! -f "$(CLEANED_TEST)" ]; then \
		echo "Running data cleaning and train/test split..."; \
		"$(PYTHON)" scripts/clean_and_split.py; \
	fi
	@if [ ! -f "$(CLEANED_TRAIN)" ] || [ ! -f "$(CLEANED_TEST)" ]; then \
		echo "ERROR: cleaning completed without producing both cleaned datasets."; \
		exit 1; \
	fi
	@echo "Dataset ready."

$(ZIP_FILE):
	@mkdir -p "$(DATA_DIR)"
	@echo "Downloading Phishing Websites dataset from UCI..."
	@curl -L -sS "$(URL)" -o "$(ZIP_FILE)"

$(STAMP_FILE): $(ZIP_FILE)
	@echo "Extracting dataset to $(DATA_DIR)/..."
	@"$(PYTHON)" -m zipfile -e "$(ZIP_FILE)" "$(DATA_DIR)"
	@rm -f "$(ZIP_FILE)"
	@touch "$(STAMP_FILE)"

# -----------------------------------------------------------------------------
# Project phases
# Each phase is deliberately separate. No target below runs a previous phase.
# -----------------------------------------------------------------------------
analyze: check-venv check-data
	@echo "Running Exploratory Data Analysis and Preprocessing pipeline..."
	@"$(PYTHON)" -m data_preprocessing.main

train: check-venv check-data
	@echo "Running Model Selection and Cross-Validation pipeline..."
	@"$(PYTHON)" -m model_selection.main

evaluate: check-venv check-data
	@echo "Running Final Evaluation pipeline..."
	@echo "Using SELECTED_RUN_NAME from code/shared/config.py"
	@"$(PYTHON)" -m final_evaluation.main

explain: check-venv check-data
	@echo "Running Explainability pipeline..."
	@echo "Using SELECTED_RUN_NAME from code/shared/config.py"
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
