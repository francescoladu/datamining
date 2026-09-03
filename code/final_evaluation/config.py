from pathlib import Path
from typing import Final

from shared.config import CODE_DIR, DATA_DIR, SELECTED_RUN_NAME


RUN_NAME: Final[str] = SELECTED_RUN_NAME

if Path(RUN_NAME).name != RUN_NAME:
    raise ValueError("SELECTED_RUN_NAME must be a directory name, not a path.")

MODEL_SELECTION_OUTPUT_DIR: Final[Path] = (
    CODE_DIR / "model_selection" / "outputs" / RUN_NAME
)

FINAL_BEST_PARAMETERS_PATH: Final[Path] = (
    MODEL_SELECTION_OUTPUT_DIR
    / "hyperparameter_search"
    / "final_best_parameters.csv"
)

DEVELOPMENT_DATA_PATH: Final[Path] = DATA_DIR / "train_cleaned.csv"
TEST_DATA_PATH: Final[Path] = DATA_DIR / "test_cleaned.csv"

OUTPUT_DIR: Final[Path] = CODE_DIR / "final_evaluation" / "outputs" / RUN_NAME
METRICS_DIR: Final[Path] = OUTPUT_DIR / "metrics"
DIAGNOSTICS_DIR: Final[Path] = OUTPUT_DIR / "diagnostics"
FIGURES_DIR: Final[Path] = OUTPUT_DIR / "figures"


def create_output_directories() -> None:
    """Create the final-evaluation output hierarchy."""
    for directory in (METRICS_DIR, DIAGNOSTICS_DIR, FIGURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)
