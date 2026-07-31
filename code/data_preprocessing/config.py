from pathlib import Path


# ============================================================
# 1. PATHS
# ============================================================

# This file lives at code/data_preprocessing/config.py, so the
# repository root is two levels up.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

# Both dataset splits are supported. The analysis can be run on
# either split, or both, by iterating over this dict.
DATASET_PATHS = {
    "train": DATA_DIR / "train_cleaned.csv",
    # "test": DATA_DIR / "test_cleaned.csv",
}
# ============================================================
# 2. DATASET SETTINGS
# ============================================================

TARGET_COLUMN = "Result"

# Columns that may result from saving the DataFrame (e.g. an
# extra index column written out by a previous export step).
INDEX_COLUMNS = [
    "index",
    "Unnamed: 0",
]

PHISHING_LABEL = -1
LEGITIMATE_LABEL = 1

RANDOM_STATE = 42

# Number of features to show in the phishing rate heatmap.
TOP_FEATURES_NUMBER = 8

# Number of strongest correlations to show.
TOP_CORRELATIONS_NUMBER = 15


# ============================================================
# 3. OUTPUT FILES
# ============================================================

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def correlation_matrix_path(dataset_name: str) -> Path:
    """CSV path for the Pearson correlation matrix of a given dataset split."""
    return OUTPUT_DIR / f"pearson_correlation_matrix_{dataset_name}.csv"


def correlation_heatmap_pdf_path(dataset_name: str) -> Path:
    """PDF path for the Pearson correlation heatmap of a given dataset split."""
    return OUTPUT_DIR / f"pearson_correlation_heatmap_{dataset_name}.pdf"


def correlation_heatmap_png_path(dataset_name: str) -> Path:
    """PNG path for the Pearson correlation heatmap of a given dataset split."""
    return OUTPUT_DIR / f"pearson_correlation_heatmap_{dataset_name}.png"