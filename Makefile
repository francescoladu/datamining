# Variables
DATA_DIR = data
ZIP_FILE = $(DATA_DIR)/phishing_websites.zip
URL = https://archive.ics.uci.edu/static/public/327/phishing+websites.zip
STAMP_FILE = $(DATA_DIR)/.extracted
CLEANED_TRAIN = $(DATA_DIR)/train_cleaned.csv

# Output locations to clean up
PREPROCESS_OUTPUT_DIR = code/data_preprocessing/outputs
MODEL_OUTPUT_DIR = code/model_selection/outputs

# Detect if local venv exists and use its interpreter; otherwise, fall back to global python3
PYTHON = $(shell [ -f venv/bin/python ] && echo "venv/bin/python" || echo "python3")
VENV = venv

.PHONY: all clean install analyze train setup help

# Default target displays the help menu
all: help

help:
	@echo "Available commands:"
	@echo "  make setup     - Create the local virtual environment"
	@echo "  make install   - Download, extract, and clean the phishing dataset"
	@echo "  make analyze   - Run validation, compute statistics, and generate EDA plots"
	@echo "  make train     - Run nested CV, model family selection, and final test evaluation"
	@echo "  make clean     - Remove datasets, stamps, and all generated outputs"

# Environment Setup
setup:
	@echo "Creating virtual environment in ./$(VENV)..."
	python3 -m venv $(VENV)
	@echo "------------------------------------------------------------"
	@echo "Virtual environment created successfully."
	@echo "To activate it, run:"
	@echo "  source $(VENV)/bin/activate"
	@echo "Then install the dependencies:"
	@echo "  pip install pandas scipy scikit-learn seaborn matplotlib"
	@echo "------------------------------------------------------------"

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

clean:
	@echo "Removing dataset files, stamps, and all generated outputs..."
	rm -rf $(DATA_DIR)/*
	rm -f $(STAMP_FILE)
	rm -rf $(PREPROCESS_OUTPUT_DIR)
	rm -rf $(MODEL_OUTPUT_DIR)