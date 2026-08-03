import os
from pathlib import Path
from typing import Final


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
CODE_DIR: Final[Path] = PROJECT_ROOT / "code"

MODEL_SELECTION_DIR: Final[Path] = CODE_DIR / "model_selection"
EXPLAINABILITY_DIR: Final[Path] = CODE_DIR / "explainability"

# The Makefile passes this value. The default matches the current
# model-selection configuration: FEATURE_SELECTION_K_VALUES = ["all"].
RUN_NAME: Final[str] = os.getenv("RUN_NAME", "k_all")

if Path(RUN_NAME).name != RUN_NAME:
    raise ValueError("RUN_NAME must be a directory name, not a path.")

MODEL_SELECTION_OUTPUT_DIR: Final[Path] = (
    MODEL_SELECTION_DIR / "outputs" / RUN_NAME
)

OUTPUT_DIR: Final[Path] = (
    EXPLAINABILITY_DIR / "outputs" / RUN_NAME
)

TRAIN_DATA_PATH: Final[Path] = (
    DATA_DIR / "train_cleaned.csv"
)

TEST_DATA_PATH: Final[Path] = (
    DATA_DIR / "test_cleaned.csv"
)

TARGET_COLUMN: Final[str] = "Result"

FINAL_BEST_PARAMETERS_PATH: Final[Path] = (
    MODEL_SELECTION_OUTPUT_DIR
    / "final_best_parameters.csv"
)

PERMUTATION_SUMMARY_SOURCE_PATH: Final[Path] = (
    MODEL_SELECTION_OUTPUT_DIR
    / "nested_cv_permutation_importance_summary.csv"
)

RANDOM_STATE: Final[int] = 42

PHISHING_LABEL: Final[int] = -1
LEGITIMATE_LABEL: Final[int] = 1

EXPECTED_LABELS: Final[set[int]] = {
    PHISHING_LABEL,
    LEGITIMATE_LABEL,
}

PLOT_DPI: Final[int] = 300

SAVE_PNG: Final[bool] = True
SAVE_PDF: Final[bool] = True

PERMUTATION_MAX_DISPLAY: Final[int | None] = None

PERMUTATION_RESULTS_PATH: Final[Path] = (
    OUTPUT_DIR
    / "nested_cv_permutation_importance_selected_model.csv"
)

PERMUTATION_PNG_PATH: Final[Path] = (
    OUTPUT_DIR
    / "nested_cv_permutation_importance.png"
)

PERMUTATION_PDF_PATH: Final[Path] = (
    OUTPUT_DIR
    / "nested_cv_permutation_importance.pdf"
)

SAMPLE_POSITION: Final[int] = int(
    os.getenv("SHAP_SAMPLE_POSITION", "0")
)

CLASS_TO_EXPLAIN: Final[int] = PHISHING_LABEL

SHAP_BACKGROUND_SIZE: Final[int] = 500


def create_output_directory() -> None:
    """Create the output directory when it does not exist."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )