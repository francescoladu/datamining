from pathlib import Path
from typing import Final

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
# 3. SELECTED MODEL-SELECTION RUN
# ============================================================
SELECTED_RUN_NAME: Final[str] = "k_search_5-10-15-20-25-all"