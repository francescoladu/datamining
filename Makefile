# Variables
DATA_DIR = data
ZIP_FILE = $(DATA_DIR)/phishing_websites.zip
URL = https://archive.ics.uci.edu/static/public/327/phishing+websites.zip
STAMP_FILE = $(DATA_DIR)/.extracted
CLEANED_TRAIN = $(DATA_DIR)/train_cleaned.csv

# Output locations to clean up
PREPROCESS_OUTPUT_DIR = code/data_preprocessing/outputs
MODEL_OUTPUT_DIR = code/model_selection/outputs
EXPLAIN_OUTPUT_DIR = code/explainability/outputs

# Detect if local venv exists and use its interpreter; otherwise, fall back to global python3
PYTHON = $(shell [ -f venv/bin/python ] && echo "venv/bin/python" || echo "python3")
VENV = venv

PIP = $(shell [ -f venv/bin/pip ] && echo "venv/bin/pip" || echo "pip3")
REQUIREMENTS = requirements.txt

.PHONY: all clean install analyze train explain setup dependencies help

export PYTHONPATH := code

# Default target displays the help menu
all: help

help:
	@echo "Available commands:"
	@echo "  make setup     - Create the local virtual environment"
	@echo "  make install   - Download, extract, and clean the phishing dataset"
	@echo "  make analyze   - Run validation, compute statistics, and generate EDA plots"
	@echo "  make train     - Run nested CV, model family selection, and final test evaluation"
	@echo "  make explain   - Run global and local explainability analysis (SHAP & Permutation Importance)"
	@echo "  make clean     - Remove datasets, stamps, and all generated outputs"
	@echo "  make demo      - Run the entire pipeline from setup to explainability"

# Environment Setup
setup: $(VENV)/bin/activate

$(VENV)/bin/activate: $(REQUIREMENTS)
	@echo "Creating virtual environment in ./$(VENV)..."
	python3 -m venv $(VENV)
	@echo "Installing dependencies from $(REQUIREMENTS)..."
	$(PIP) install --upgrade pip
	$(PIP) install -r $(REQUIREMENTS)
	@touch $(VENV)/bin/activate
	@echo "------------------------------------------------------------"
	@echo "Setup complete. Virtual environment created and updated."
	@echo "To activate it, run:"
	@echo "  source $(VENV)/bin/activate"
	@echo "------------------------------------------------------------"

# Target to manually force-update or install dependencies
dependencies: $(REQUIREMENTS)
	@echo "Updating dependencies..."
	$(PIP) install -r $(REQUIREMENTS)

install: $(CLEANED_TRAIN)

$(ZIP_FILE):
	@mkdir -p $(DATA_DIR)
	@echo "Downloading Phishing Websites dataset from UCI..."
	curl -L -sS "$(URL)" -o "$(ZIP_FILE)"

$(STAMP_FILE): $(ZIP_FILE)
	@echo "Extracting dataset to $(DATA_DIR)/..."
	@$(PYTHON) -m zipfile -e "$(ZIP_FILE)" $(DATA_DIR)
	@rm -f "$(ZIP_FILE)"
	@touch "$(STAMP_FILE)"

# Automatically run the cleaning script once the raw data is extracted
$(CLEANED_TRAIN): $(STAMP_FILE)
	@echo "Running data cleaning and split script..."
	$(PYTHON) scripts/clean_and_split.py

# Automatically run the EDA and plotting pipeline
analyze: $(CLEANED_TRAIN)
	@echo "Running Exploratory Data Analysis and Preprocessing pipeline..."
	$(PYTHON) code/data_preprocessing/main.py

# Automatically run the model selection, training, and testing pipeline
train: $(CLEANED_TRAIN)
	@echo "Running Model Selection and Cross-Validation pipeline..."
	$(PYTHON) code/model_selection/main.py

# Automatically run the global and local explainability pipeline
explain: $(CLEANED_TRAIN)
	@echo "Running Explainability pipeline..."
	$(PYTHON) code/explainability/main.py

demo: setup
	@echo "Starting the end-to-end demonstration..."
	$(MAKE) install
	$(MAKE) analyze
	$(MAKE) train
	$(MAKE) explain
	@echo "------------------------------------------------------------"
	@echo "Demo complete! All pipeline stages executed successfully."
	@echo "------------------------------------------------------------"

clean:
	@echo "Removing dataset files, stamps, and all generated outputs..."
	rm -rf $(DATA_DIR)/*
	rm -f $(STAMP_FILE)
	rm -rf $(PREPROCESS_OUTPUT_DIR)
	rm -rf $(MODEL_OUTPUT_DIR)
	rm -rf $(EXPLAIN_OUTPUT_DIR)