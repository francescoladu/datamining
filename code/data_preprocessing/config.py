from pathlib import Path

from shared.config import (
    DATA_DIR,
    TARGET_COLUMN,
    SAMPLE_WEIGHT_COLUMN,
    PHISHING_LABEL,
    LEGITIMATE_LABEL,
    RANDOM_STATE,
)


# ===========================================================================
# EXPERIMENT DATASETS
# ===========================================================================

EXPERIMENT_DATASET_PATHS = {
    "1": DATA_DIR / "experiment_1_raw_train.csv",
    "2": DATA_DIR / "experiment_2_standard_dedup_train.csv",
    "3": DATA_DIR / "train_cleaned.csv",
}

EXPERIMENT_NAMES = {
    "1": "Experiment 1 - Raw Data",
    "2": "Experiment 2 - Exact Deduplication",
    "3": "Experiment 3 - Weighted Deduplication",
}


# ===========================================================================
# OUTPUT DIRECTORIES
# ===========================================================================

OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"

EXPERIMENT_OUTPUT_DIRS = {
    "1": OUTPUT_ROOT / "experiment_1_raw",
    "2": OUTPUT_ROOT / "experiment_2_standard_dedup",
    "3": OUTPUT_ROOT / "experiment_3_weighted_dedup",
}


# ===========================================================================
# EDA SETTINGS
# ===========================================================================

INDEX_COLUMNS = [
    "index",
    "Unnamed: 0",
]

TOP_FEATURES_NUMBER = 30
TOP_CORRELATIONS_NUMBER = 15


# ===========================================================================
# OUTPUT PATH HELPERS
# ===========================================================================

def feature_histograms_dir(
    output_dir: Path,
    dataset_name: str,
) -> Path:
    """Directory containing feature histograms for one dataset split."""
    return output_dir / f"feature_histograms_{dataset_name}"


def correlation_matrix_path(
    output_dir: Path,
    dataset_name: str,
) -> Path:
    """CSV path for the Spearman correlation matrix."""
    return (
        output_dir
        / f"spearman_correlation_matrix_top_features_{dataset_name}.csv"
    )


def correlation_heatmap_pdf_path(
    output_dir: Path,
    dataset_name: str,
) -> Path:
    """PDF path for the Spearman correlation heatmap."""
    return (
        output_dir
        / f"spearman_correlation_heatmap_top_features_{dataset_name}.pdf"
    )


def correlation_heatmap_png_path(
    output_dir: Path,
    dataset_name: str,
) -> Path:
    """PNG path for the Spearman correlation heatmap."""
    return (
        output_dir
        / f"spearman_correlation_heatmap_top_features_{dataset_name}.png"
    )