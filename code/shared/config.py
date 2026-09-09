from pathlib import Path
from typing import Any, Final

# ============================================================
# 1. GLOBAL PATHS
# ============================================================
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
CODE_DIR: Final[Path] = PROJECT_ROOT / "code"

# ============================================================
# 2. GLOBAL DATASET SETTINGS
# ============================================================
TARGET_COLUMN: Final[str] = "Result"
SAMPLE_WEIGHT_COLUMN: Final[str] = "sample_weight"

PHISHING_LABEL: Final[int] = -1
LEGITIMATE_LABEL: Final[int] = 1

EXPECTED_LABELS: Final[set[int]] = {
    PHISHING_LABEL,
    LEGITIMATE_LABEL,
}

# Shared random seed for reproducibility across all modules
RANDOM_STATE: Final[int] = 42

# ============================================================
# 3. FEATURE SELECTION & RUN CONFIGURATION (SINGLE SOURCE OF TRUTH)
# ============================================================

# Fix 1: Define FEATURE_SELECTION_K_VALUES and build_run_tag in shared/config.py so all modules use the same dynamic run name

FEATURE_SELECTION_K_VALUES: list[Any] = [5, 10, 15, 20, 25, "all"]


def build_run_tag(k_values: list[Any]) -> str:
    """Build a filesystem-safe feature-selection run name."""
    labels = [str(value).lower() for value in k_values]
    if len(labels) == 1:
        return f"k_{labels[0]}"
    return "k_search_" + "-".join(labels)


SELECTED_RUN_NAME: Final[str] = build_run_tag(FEATURE_SELECTION_K_VALUES)
