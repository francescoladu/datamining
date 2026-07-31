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

# Number of Mutual-Information-ranked features included in the
# Pearson correlation heatmap.
TOP_FEATURES_NUMBER = 15

# Number of strongest correlations to export in CSV format.
TOP_CORRELATIONS_NUMBER = 15


# ============================================================
# 3. OUTPUT FILES
# ============================================================

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def feature_histograms_dir(dataset_name: str) -> Path:
    """Directory containing feature histograms for a dataset split."""
    return OUTPUT_DIR / f"feature_histograms_{dataset_name}"


def correlation_matrix_path(dataset_name: str) -> Path:
    """CSV path for the Pearson matrix of the most relevant features."""
    return OUTPUT_DIR / f"pearson_correlation_matrix_top_features_{dataset_name}.csv"


def correlation_heatmap_pdf_path(dataset_name: str) -> Path:
    """PDF path for the Pearson heatmap of the most relevant features."""
    return OUTPUT_DIR / f"pearson_correlation_heatmap_top_features_{dataset_name}.pdf"


def correlation_heatmap_png_path(dataset_name: str) -> Path:
    """PNG path for the Pearson heatmap of the most relevant features."""
    return OUTPUT_DIR / f"pearson_correlation_heatmap_top_features_{dataset_name}.png"