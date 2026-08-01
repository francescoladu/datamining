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

# Main metric used for feature and hyperparameter selection.
PRIMARY_SCORING = "f1_macro"

# Number of iterations sampled by RandomizedSearchCV for Random Forest
N_RANDOM_ITERATIONS = 5 

# Outer CV evaluates the complete model-selection procedure.
outer_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE,
)

# Inner CV configuration used for the final search on the complete dataset.
final_inner_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE,
)


# ===========================================================================
# 2. FEATURE-SELECTION FUNCTION
# ===========================================================================

# All predictors are discrete, so mutual information is computed by explicitly
# treating every input feature as discrete.
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
            DecisionTreeClassifier(
                random_state=RANDOM_STATE,
            ),
        ),
    ]
)

# Small and structured search space:
# GridSearchCV evaluates every combination.
decision_tree_param_grid = {
    # "all" represents the model without feature removal.
    "feature_selection__k": [10, 15, 20, 25, "all"],

    # Hyperparameters used in the Decision Tree workflow.
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

                # Setting n_jobs=1 here avoids nested parallelism.
                # Parallel computation is handled by the parent SearchCV instead.
                n_jobs=1,
            ),
        ),
    ]
)

# The Random Forest has a larger hyperparameter space.
# RandomizedSearchCV samples a fixed number of configurations.
random_forest_param_distributions = {
    "feature_selection__k": [10, 15, 20, 25, "all"],
    "classifier__n_estimators": [100, 200, 400, 600, 800],
    "classifier__criterion": ["gini", "entropy"],
    "classifier__max_depth": [None, 5, 10, 15, 20, 30],
    "classifier__min_samples_split": [2, 4, 6, 10],
    "classifier__min_samples_leaf": [1, 2, 4],
    "classifier__max_features": ["sqrt", "log2", 0.5, None],
    "classifier__bootstrap": [True, False],
}