from pathlib import Path
from typing import Final, Literal


# ============================================================
# PROJECT PATHS
# ============================================================

# Absolute path to the project root directory
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

# Main project directories
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
CODE_DIR: Final[Path] = PROJECT_ROOT / "code"

MODEL_SELECTION_DIR: Final[Path] = CODE_DIR / "model_selection"
EXPLAINABILITY_DIR: Final[Path] = CODE_DIR / "explainability"


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

# Model-selection experiment to explain.
# Accepted values currently available in the project:
# - "k_25"
# - "k_all"
RUN_NAME: Final[Literal["k_25", "k_all"]] = "k_25"


# Directory containing the model-selection results
MODEL_SELECTION_OUTPUT_DIR: Final[Path] = (
    MODEL_SELECTION_DIR / "outputs" / RUN_NAME
)

# Directory where explainability results will be saved
OUTPUT_DIR: Final[Path] = (
    EXPLAINABILITY_DIR / "outputs" / RUN_NAME
)


# ============================================================
# DATASET PATHS
# ============================================================

TRAIN_DATA_PATH: Final[Path] = DATA_DIR / "train_cleaned.csv"
TEST_DATA_PATH: Final[Path] = DATA_DIR / "test_cleaned.csv"

# Name of the target column in the cleaned datasets
TARGET_COLUMN: Final[str] = "Result"


# ============================================================
# MODEL-SELECTION FILES
# ============================================================

# The final trained estimator must be saved with this name
MODEL_PATH: Final[Path] = (
    MODEL_SELECTION_OUTPUT_DIR / "final_model.joblib"
)

# CSV containing the features selected during model selection
SELECTED_FEATURES_PATH: Final[Path] = (
    MODEL_SELECTION_OUTPUT_DIR / "final_selected_features.csv"
)


# ============================================================
# GENERAL SETTINGS
# ============================================================

RANDOM_STATE: Final[int] = 42

# Image resolution used when saving plots
PLOT_DPI: Final[int] = 300

# Save plots in both formats
SAVE_PNG: Final[bool] = True
SAVE_PDF: Final[bool] = True


# ============================================================
# GLOBAL EXPLAINABILITY: PERMUTATION IMPORTANCE
# ============================================================

# Metric whose decrease is measured after shuffling a feature
PERMUTATION_SCORING: Final[str] = "f1_macro"

# Number of independent permutations for each feature
PERMUTATION_N_REPEATS: Final[int] = 40

# Use all available CPU cores
PERMUTATION_N_JOBS: Final[int] = -1

# Maximum number of features displayed in the plot.
# Use None to display all selected features.
PERMUTATION_MAX_DISPLAY: Final[int | None] = None


# ============================================================
# LOCAL EXPLAINABILITY: SHAP FORCE PLOT
# ============================================================

# Position inside the test set of the observation to explain
SAMPLE_POSITION: Final[int] = 0

# Explain the probability output rather than raw model scores
SHAP_MODEL_OUTPUT: Final[str] = "probability"

# Use the interventional approach adopted in the lecture code
SHAP_FEATURE_PERTURBATION: Final[str] = "interventional"

# The professor's code disables the additional additivity check
SHAP_CHECK_ADDITIVITY: Final[bool] = False

# Use the complete training set as SHAP background data.
# Set this to False to use a random sample instead.
SHAP_USE_FULL_TRAINING_BACKGROUND: Final[bool] = True

# Number of training observations used when the complete
# training set is not used as background data
SHAP_BACKGROUND_SIZE: Final[int] = 500


# ============================================================
# OUTPUT FILES
# ============================================================

PERMUTATION_RESULTS_PATH: Final[Path] = (
    OUTPUT_DIR / "final_permutation_importance.csv"
)

PERMUTATION_PNG_PATH: Final[Path] = (
    OUTPUT_DIR / "final_permutation_importance.png"
)

PERMUTATION_PDF_PATH: Final[Path] = (
    OUTPUT_DIR / "final_permutation_importance.pdf"
)

SHAP_CONTRIBUTIONS_PATH: Final[Path] = (
    OUTPUT_DIR / f"shap_local_contributions_sample_{SAMPLE_POSITION:04d}.csv"
)

SHAP_FORCE_PNG_PATH: Final[Path] = (
    OUTPUT_DIR / f"shap_force_sample_{SAMPLE_POSITION:04d}.png"
)

SHAP_FORCE_PDF_PATH: Final[Path] = (
    OUTPUT_DIR / f"shap_force_sample_{SAMPLE_POSITION:04d}.pdf"
)

SHAP_PREDICTION_SUMMARY_PATH: Final[Path] = (
    OUTPUT_DIR / f"shap_prediction_summary_sample_{SAMPLE_POSITION:04d}.csv"
)


# ============================================================
# DIRECTORY INITIALIZATION
# ============================================================

def create_output_directory() -> None:
    """Create the explainability output directory if necessary."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )