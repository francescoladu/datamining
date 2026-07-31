# Variables
DATA_DIR = data
ZIP_FILE = $(DATA_DIR)/phishing_websites.zip
URL = https://archive.ics.uci.edu/static/public/327/phishing+websites.zip
STAMP_FILE = $(DATA_DIR)/.extracted
CLEANED_TRAIN = $(DATA_DIR)/train_cleaned.csv
PYTHON = python3

.PHONY: all clean install

all: install

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

clean:
	@echo "Removing dataset files..."
	rm -rf $(DATA_DIR)/*
	rm -f $(STAMP_FILE)