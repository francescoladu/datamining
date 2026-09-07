from pathlib import Path

# Import global configurations from shared module
from shared.config import (
    DATA_DIR,
    TARGET_COLUMN,
    SAMPLE_WEIGHT_COLUMN,
    PHISHING_LABEL,
    LEGITIMATE_LABEL,
    RANDOM_STATE,
)

DATASET_PATHS = {
    "train": DATA_DIR / "train_cleaned.csv",
    # "test": DATA_DIR / "test_cleaned.csv",
}

INDEX_COLUMNS = [
    "index",
    "Unnamed: 0",
]

TOP_FEATURES_NUMBER = 30
TOP_CORRELATIONS_NUMBER = 15

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