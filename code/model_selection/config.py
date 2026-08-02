from functools import partial

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

# ===========================================================================
# 1. GENERAL SETTINGS
# ===========================================================================

RANDOM_STATE = 42
PRIMARY_SCORING = "f1_macro"

# Number of configurations sampled by RandomizedSearchCV for Random Forest.
N_RANDOM_ITERATIONS = 40

# Compute permutation importance on every untouched outer validation fold.
COMPUTE_PERMUTATION_IMPORTANCE = True
PERMUTATION_N_REPEATS = 20

# Threshold used to flag confidently wrong predictions in the error analysis.
HIGH_CONFIDENCE_THRESHOLD = 0.80

# IMPORTANT:
# Keep this False while comparing different values of k. The held-out test set
# must be evaluated only once, after the final experimental setup is chosen.
EVALUATE_FINAL_TEST = False

# ---------------------------------------------------------------------------
# FEATURE-SELECTION EXPERIMENT
# ---------------------------------------------------------------------------
# Activate ONE line only. The same setting is applied to Decision Tree and
# Random Forest, allowing fair comparisons on exactly the same outer folds.

# FEATURE_SELECTION_K_VALUES = [5, 10, 15, 20, 25]  # Joint k search
FEATURE_SELECTION_K_VALUES = ["all"]                # All 30 features

# Outer CV evaluates the complete model-selection procedure.
outer_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE,
)

# Inner CV used for the final search on the complete development set.
final_inner_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE,
)


# ===========================================================================
# 2. FEATURE-SELECTION FUNCTION
# ===========================================================================

# All predictors are discrete, so Mutual Information treats every feature as
# discrete rather than as a continuous measurement.
mutual_information = partial(
    mutual_info_classif,
    discrete_features=True,
    random_state=RANDOM_STATE,
)


# ===========================================================================
# 3. DECISION TREE PIPELINE AND SEARCH SPACE
# ===========================================================================

decision_tree_pipeline = Pipeline(
    steps=[
        (
            "feature_selection",
            SelectKBest(score_func=mutual_information),
        ),
        (
            "classifier",
            DecisionTreeClassifier(random_state=RANDOM_STATE),
        ),
    ]
)

decision_tree_param_grid = {
    "feature_selection__k": FEATURE_SELECTION_K_VALUES,
    "classifier__criterion": ["gini", "entropy"],
    "classifier__max_depth": [None, 3, 5, 7, 9, 12],
    "classifier__min_samples_split": [2, 4, 6, 8],
    "classifier__min_samples_leaf": [1, 2, 4],
}


# ===========================================================================
# 4. RANDOM FOREST PIPELINE AND SEARCH SPACE
# ===========================================================================

random_forest_pipeline = Pipeline(
    steps=[
        (
            "feature_selection",
            SelectKBest(score_func=mutual_information),
        ),
        (
            "classifier",
            RandomForestClassifier(
                random_state=RANDOM_STATE,
                # The parent SearchCV handles parallel computation.
                n_jobs=1,
            ),
        ),
    ]
)

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