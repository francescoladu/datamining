import os
from pathlib import Path

from shared.config import (
    CODE_DIR,
    DATA_DIR,
    PHISHING_LABEL,
    RANDOM_STATE,
    SELECTED_RUN_NAME,
)


MODEL_SELECTION_DIR = CODE_DIR / "model_selection"
EXPLAINABILITY_DIR = CODE_DIR / "explainability"

RUN_NAME = SELECTED_RUN_NAME

if Path(RUN_NAME).name != RUN_NAME:
    raise ValueError(
        "SELECTED_RUN_NAME must be a directory name, not a path."
    )


# ===========================================================================
# EXPLAINABILITY SETTINGS
# ===========================================================================

PLOT_DPI = 300
SAVE_PNG = False
SAVE_PDF = True

PERMUTATION_MAX_DISPLAY = 10

SAMPLE_POSITION = int(
    os.getenv(
        "SHAP_SAMPLE_POSITION",
        "0",
    )
)

CLASS_TO_EXPLAIN = PHISHING_LABEL
SHAP_BACKGROUND_SIZE = 500


# ===========================================================================
# PATHS
#
# These are configured at runtime by configure_experiment().
# Experiment 3 is used as the default only so the module always has
# valid path variables before configuration.
# ===========================================================================

TRAIN_DATA_PATH = DATA_DIR / "train_cleaned.csv"
TEST_DATA_PATH = DATA_DIR / "test_cleaned.csv"

MODEL_SELECTION_OUTPUT_DIR = (
    MODEL_SELECTION_DIR
    / "outputs"
    / RUN_NAME
)

OUTPUT_DIR = (
    EXPLAINABILITY_DIR
    / "outputs"
    / RUN_NAME
)

GLOBAL_OUTPUT_DIR = (
    OUTPUT_DIR
    / "global"
)

LOCAL_OUTPUT_DIR = (
    OUTPUT_DIR
    / "local"
)

FINAL_BEST_PARAMETERS_PATH = (
    MODEL_SELECTION_OUTPUT_DIR
    / "hyperparameter_search"
    / "final_best_parameters.csv"
)

PERMUTATION_SUMMARY_SOURCE_PATH = (
    MODEL_SELECTION_OUTPUT_DIR
    / "explainability"
    / "permutation_importance_summary.csv"
)

PERMUTATION_PDF_PATH = (
    GLOBAL_OUTPUT_DIR
    / "permutation_importance.pdf"
)


def configure_experiment(
    experiment: str,
) -> None:
    """
    Configure explainability paths for one experiment.
    """

    global TRAIN_DATA_PATH
    global TEST_DATA_PATH
    global MODEL_SELECTION_OUTPUT_DIR
    global OUTPUT_DIR
    global GLOBAL_OUTPUT_DIR
    global LOCAL_OUTPUT_DIR
    global FINAL_BEST_PARAMETERS_PATH
    global PERMUTATION_SUMMARY_SOURCE_PATH
    global PERMUTATION_PDF_PATH

    if experiment == "1":

        TRAIN_DATA_PATH = (
            DATA_DIR
            / "experiment_1_raw_train.csv"
        )

        TEST_DATA_PATH = (
            DATA_DIR
            / "experiment_1_raw_test.csv"
        )

        MODEL_SELECTION_OUTPUT_DIR = (
            MODEL_SELECTION_DIR
            / "outputs"
            / "experiment_1_raw"
            / RUN_NAME
        )

        OUTPUT_DIR = (
            EXPLAINABILITY_DIR
            / "outputs"
            / "experiment_1_raw"
            / RUN_NAME
        )

    elif experiment == "2":

        TRAIN_DATA_PATH = (
            DATA_DIR
            / "experiment_2_standard_dedup_train.csv"
        )

        TEST_DATA_PATH = (
            DATA_DIR
            / "experiment_2_standard_dedup_test.csv"
        )

        MODEL_SELECTION_OUTPUT_DIR = (
            MODEL_SELECTION_DIR
            / "outputs"
            / "experiment_2_standard_dedup"
            / RUN_NAME
        )

        OUTPUT_DIR = (
            EXPLAINABILITY_DIR
            / "outputs"
            / "experiment_2_standard_dedup"
            / RUN_NAME
        )

    elif experiment == "3":

        TRAIN_DATA_PATH = (
            DATA_DIR
            / "train_cleaned.csv"
        )

        TEST_DATA_PATH = (
            DATA_DIR
            / "test_cleaned.csv"
        )

        MODEL_SELECTION_OUTPUT_DIR = (
            MODEL_SELECTION_DIR
            / "outputs"
            / RUN_NAME
        )

        OUTPUT_DIR = (
            EXPLAINABILITY_DIR
            / "outputs"
            / RUN_NAME
        )

    else:
        raise ValueError(
            "Experiment must be '1', '2', or '3'."
        )

    GLOBAL_OUTPUT_DIR = (
        OUTPUT_DIR
        / "global"
    )

    LOCAL_OUTPUT_DIR = (
        OUTPUT_DIR
        / "local"
    )

    FINAL_BEST_PARAMETERS_PATH = (
        MODEL_SELECTION_OUTPUT_DIR
        / "hyperparameter_search"
        / "final_best_parameters.csv"
    )

    PERMUTATION_SUMMARY_SOURCE_PATH = (
        MODEL_SELECTION_OUTPUT_DIR
        / "explainability"
        / "permutation_importance_summary.csv"
    )

    PERMUTATION_PDF_PATH = (
        GLOBAL_OUTPUT_DIR
        / "permutation_importance.pdf"
    )


def create_output_directories() -> None:
    """Create explainability output directories."""

    for directory in (
        GLOBAL_OUTPUT_DIR,
        LOCAL_OUTPUT_DIR,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )