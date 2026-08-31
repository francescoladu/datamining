import os
from pathlib import Path
from typing import Final

from shared.config import (
    CODE_DIR,
    DATA_DIR,
    EXPECTED_LABELS,
    LEGITIMATE_LABEL,
    PHISHING_LABEL,
    RANDOM_STATE,
    SELECTED_RUN_NAME,
    TARGET_COLUMN,
)

MODEL_SELECTION_DIR: Final[Path] = CODE_DIR / "model_selection"
EXPLAINABILITY_DIR: Final[Path] = CODE_DIR / "explainability"

RUN_NAME: Final[str] = SELECTED_RUN_NAME
if Path(RUN_NAME).name != RUN_NAME:
    raise ValueError("SELECTED_RUN_NAME must be a directory name, not a path.")

MODEL_SELECTION_OUTPUT_DIR: Final[Path] = (
    MODEL_SELECTION_DIR / "outputs" / RUN_NAME
)
OUTPUT_DIR: Final[Path] = EXPLAINABILITY_DIR / "outputs" / RUN_NAME
GLOBAL_OUTPUT_DIR: Final[Path] = OUTPUT_DIR / "global"
LOCAL_OUTPUT_DIR: Final[Path] = OUTPUT_DIR / "local"

TRAIN_DATA_PATH: Final[Path] = DATA_DIR / "train_cleaned.csv"
TEST_DATA_PATH: Final[Path] = DATA_DIR / "test_cleaned.csv"

FINAL_BEST_PARAMETERS_PATH: Final[Path] = (
    MODEL_SELECTION_OUTPUT_DIR
    / "hyperparameter_search"
    / "final_best_parameters.csv"
)
PERMUTATION_SUMMARY_SOURCE_PATH: Final[Path] = (
    MODEL_SELECTION_OUTPUT_DIR
    / "explainability"
    / "permutation_importance_summary.csv"
)

PLOT_DPI: Final[int] = 300
SAVE_PNG: Final[bool] = False
SAVE_PDF: Final[bool] = True
PERMUTATION_MAX_DISPLAY: Final[int | None] = None
PERMUTATION_PDF_PATH: Final[Path] = (
    GLOBAL_OUTPUT_DIR / "permutation_importance.pdf"
)

SAMPLE_POSITION: Final[int] = int(os.getenv("SHAP_SAMPLE_POSITION", "0"))
CLASS_TO_EXPLAIN: Final[int] = PHISHING_LABEL
SHAP_BACKGROUND_SIZE: Final[int] = 500


def create_output_directories() -> None:
    """Create the global/local explainability output hierarchy."""
    for directory in (GLOBAL_OUTPUT_DIR, LOCAL_OUTPUT_DIR):
        directory.mkdir(parents=True, exist_ok=True)
