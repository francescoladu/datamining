from pathlib import Path

# Import global configurations from shared module
from shared.config import (
    DATA_DIR,
    TARGET_COLUMN,
    PHISHING_LABEL,
    LEGITIMATE_LABEL,
    RANDOM_STATE
)

# ============================================================
# 1. DATASET SETTINGS
# ============================================================

# Both dataset splits are supported. The analysis can be run on
# either split, or both, by iterating over this dict.
DATASET_PATHS = {
    "train": DATA_DIR / "train_cleaned.csv",
    # "test": DATA_DIR / "test_cleaned.csv",
}

# Columns that may result from saving the DataFrame (e.g. an
# extra index column written out by a previous export step).
INDEX_COLUMNS = [
    "index",
    "Unnamed: 0",
]

# Number of Mutual-Information-ranked features included in the
# Spearman correlation heatmap.
TOP_FEATURES_NUMBER = 15

# Number of strongest correlations to export in CSV format.
TOP_CORRELATIONS_NUMBER = 15

# ============================================================
# 2. OUTPUT FILES
# ============================================================

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def feature_histograms_dir(dataset_name: str) -> Path:
    """Directory containing feature histograms for a dataset split."""
    return OUTPUT_DIR / f"feature_histograms_{dataset_name}"


def correlation_matrix_path(dataset_name: str) -> Path:
    """CSV path for the Spearman matrix of the most relevant features."""
    return OUTPUT_DIR / f"spearman_correlation_matrix_top_features_{dataset_name}.csv"


def correlation_heatmap_pdf_path(dataset_name: str) -> Path:
    """PDF path for the Spearman heatmap of the most relevant features."""
    return OUTPUT_DIR / f"spearman_correlation_heatmap_top_features_{dataset_name}.pdf"


def correlation_heatmap_png_path(dataset_name: str) -> Path:
    """PNG path for the Spearman heatmap of the most relevant features."""
    return OUTPUT_DIR / f"spearman_correlation_heatmap_top_features_{dataset_name}.png"