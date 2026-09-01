# Import global configurations from shared module
from shared.config import RANDOM_STATE

# ===========================================================================
# 1. GENERAL SETTINGS
# ===========================================================================

PRIMARY_SCORING = "f1_macro"

# Compute permutation importance on every untouched outer validation fold.
COMPUTE_PERMUTATION_IMPORTANCE = True
PERMUTATION_N_REPEATS = 20

# Threshold used to flag confidently wrong predictions in the error analysis.
HIGH_CONFIDENCE_THRESHOLD = 0.80

# Cross Validation Split settings (To be passed into StratifiedKFold in engine.py)
N_OUTER_SPLITS = 10
N_INNER_SPLITS = 5

# ---------------------------------------------------------------------------
# FEATURE-SELECTION EXPERIMENT
# ---------------------------------------------------------------------------
# Treat the number of selected features as an inner-CV hyperparameter.
# Including "all" lets CV choose no feature reduction when that generalizes best.

#FEATURE_SELECTION_K_VALUES = ["all"]
FEATURE_SELECTION_K_VALUES = [5, 10, 15, 20, 25, "all"]


# Number of configurations sampled by RandomizedSearchCV for Random Forest.
N_RANDOM_ITERATIONS = 30 * len(FEATURE_SELECTION_K_VALUES)


# ===========================================================================
# 2. DECISION TREE SEARCH SPACE
# ===========================================================================

decision_tree_param_grid = {
    "feature_selection__k": FEATURE_SELECTION_K_VALUES,
    "classifier__criterion": ["gini", "entropy"],
    "classifier__max_depth": [None, 3, 5, 7, 9, 12],
    "classifier__min_samples_split": [2, 4, 6, 8],
    "classifier__min_samples_leaf": [1, 2, 4],
}

# ===========================================================================
# 3. RANDOM FOREST SEARCH SPACE
# ===========================================================================

random_forest_param_distributions = {
    "feature_selection__k": FEATURE_SELECTION_K_VALUES,
    "classifier__n_estimators": [100, 200, 400, 600, 800],
    "classifier__criterion": ["gini", "entropy"],
    "classifier__max_depth": [None, 5, 10, 15, 20, 30],
    "classifier__min_samples_split": [2, 4, 6, 10],
    "classifier__min_samples_leaf": [1, 2, 4],
    "classifier__max_features": ["sqrt", "log2", 0.5, None],
    "classifier__bootstrap": [True, False],
}