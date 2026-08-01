from __future__ import annotations

from functools import partial
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier


# ---------------------------------------------------------------------------
# 1. GENERAL SETTINGS
# ---------------------------------------------------------------------------

RANDOM_STATE = 42

# Main metric used for feature and hyperparameter selection.
PRIMARY_SCORING = "f1_macro"

# Outer CV evaluates the complete model-selection procedure.
outer_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE,
)

# The same outer folds will be used for both model families.
# This makes the comparison between Decision Tree and Random Forest paired
# and more consistent.
outer_splits = list(outer_cv.split(X_dev, y_dev))


# ---------------------------------------------------------------------------
# 2. FEATURE-SELECTION FUNCTION
# ---------------------------------------------------------------------------

# All predictors are discrete, so mutual information is computed by explicitly
# treating every input feature as discrete.
mutual_information = partial(
    mutual_info_classif,
    discrete_features=True,
    random_state=RANDOM_STATE,
)


# ---------------------------------------------------------------------------
# 3. DECISION TREE PIPELINE AND SEARCH SPACE
# ---------------------------------------------------------------------------

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

    # Hyperparameters used in the professor's Decision Tree workflow.
    "classifier__criterion": ["gini", "entropy"],
    "classifier__max_depth": [None, 3, 5, 7, 9, 12],
    "classifier__min_samples_split": [2, 4, 6, 8],
    "classifier__min_samples_leaf": [1, 2, 4],
}


# ---------------------------------------------------------------------------
# 4. RANDOM FOREST PIPELINE AND SEARCH SPACE
# ---------------------------------------------------------------------------

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

                # The outer search performs parallel computation.
                # Setting n_jobs=1 here avoids nested parallelism.
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


# ---------------------------------------------------------------------------
# 5. INDEXING UTILITY
# ---------------------------------------------------------------------------

def select_rows(data: Any, indices: np.ndarray) -> Any:
    """
    Select rows from a pandas object or a NumPy array.

    Parameters
    ----------
    data:
        DataFrame, Series, or NumPy array.
    indices:
        Integer row positions.

    Returns
    -------
    Selected rows in the same format as the input object.
    """
    if hasattr(data, "iloc"):
        return data.iloc[indices]

    return data[indices]


# ---------------------------------------------------------------------------
# 6. METRIC COMPUTATION
# ---------------------------------------------------------------------------

def compute_classification_metrics(
    fitted_pipeline: Pipeline,
    X_validation: Any,
    y_validation: Any,
) -> dict[str, float]:
    """
    Compute the evaluation metrics on one outer validation fold.

    Phishing is encoded as -1 and is treated as the positive class for
    precision, recall, and ROC-AUC.
    """
    y_pred = fitted_pipeline.predict(X_validation)

    classifier = fitted_pipeline.named_steps["classifier"]

    # Find the probability column corresponding to the phishing class (-1).
    phishing_class_index = int(
        np.flatnonzero(classifier.classes_ == -1)[0]
    )

    phishing_probability = fitted_pipeline.predict_proba(
        X_validation
    )[:, phishing_class_index]

    # ROC-AUC expects a binary indicator in which the positive class is 1.
    y_phishing_binary = (
        np.asarray(y_validation) == -1
    ).astype(int)

    return {
        "macro_f1": f1_score(
            y_validation,
            y_pred,
            average="macro",
        ),
        "phishing_precision": precision_score(
            y_validation,
            y_pred,
            pos_label=-1,
            zero_division=0,
        ),
        "phishing_recall": recall_score(
            y_validation,
            y_pred,
            pos_label=-1,
            zero_division=0,
        ),
        "accuracy": accuracy_score(
            y_validation,
            y_pred,
        ),
        "roc_auc": roc_auc_score(
            y_phishing_binary,
            phishing_probability,
        ),
    }


# ---------------------------------------------------------------------------
# 7. NESTED CROSS-VALIDATION FUNCTION
# ---------------------------------------------------------------------------

def nested_cross_validation(
    *,
    model_name: str,
    pipeline: Pipeline,
    search_space: dict[str, list[Any]],
    search_method: str,
    X: Any,
    y: Any,
    n_random_iterations: int = 40,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """
    Run nested stratified cross-validation for one model family.

    Inner CV:
        Selects the number of features and classifier hyperparameters.

    Outer CV:
        Evaluates the selected pipeline on data that were not used for
        feature or hyperparameter selection.
    """
    fold_results: list[dict[str, Any]] = []
    selected_configurations: list[dict[str, Any]] = []

    for outer_fold, (outer_train_idx, outer_validation_idx) in enumerate(
        outer_splits,
        start=1,
    ):
        print(
            f"{model_name} - outer fold "
            f"{outer_fold}/{len(outer_splits)}"
        )

        X_outer_train = select_rows(X, outer_train_idx)
        y_outer_train = select_rows(y, outer_train_idx)

        X_outer_validation = select_rows(
            X,
            outer_validation_idx,
        )
        y_outer_validation = select_rows(
            y,
            outer_validation_idx,
        )

        # The inner CV is recreated inside every outer fold.
        inner_cv = StratifiedKFold(
            n_splits=4,
            shuffle=True,
            random_state=RANDOM_STATE + outer_fold,
        )

        if search_method == "grid":
            inner_search = GridSearchCV(
                estimator=pipeline,
                param_grid=search_space,
                scoring=PRIMARY_SCORING,
                cv=inner_cv,
                refit=True,
                n_jobs=-1,
                return_train_score=False,
                error_score="raise",
            )

        elif search_method == "random":
            inner_search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=search_space,
                n_iter=n_random_iterations,
                scoring=PRIMARY_SCORING,
                cv=inner_cv,
                refit=True,
                random_state=RANDOM_STATE + outer_fold,
                n_jobs=-1,
                return_train_score=False,
                error_score="raise",
            )

        else:
            raise ValueError(
                "search_method must be either 'grid' or 'random'."
            )

        # The pipeline is fitted only on the outer-training portion.
        # Feature selection is therefore recalculated inside the inner folds.
        inner_search.fit(
            X_outer_train,
            y_outer_train,
        )

        best_pipeline = inner_search.best_estimator_

        # Evaluate the selected configuration on the untouched outer fold.
        metrics = compute_classification_metrics(
            fitted_pipeline=best_pipeline,
            X_validation=X_outer_validation,
            y_validation=y_outer_validation,
        )

        fold_results.append(
            {
                "model": model_name,
                "outer_fold": outer_fold,
                "inner_best_macro_f1": inner_search.best_score_,
                **metrics,
            }
        )

        selected_configurations.append(
            {
                "model": model_name,
                "outer_fold": outer_fold,
                "best_inner_score": inner_search.best_score_,
                "best_parameters": inner_search.best_params_,
            }
        )

        print(
            f"  Inner best macro F1: "
            f"{inner_search.best_score_:.4f}"
        )
        print(
            f"  Outer macro F1: "
            f"{metrics['macro_f1']:.4f}"
        )
        print(
            f"  Best parameters: "
            f"{inner_search.best_params_}"
        )
        print()

    return (
        pd.DataFrame(fold_results),
        selected_configurations,
    )


# ---------------------------------------------------------------------------
# 8. RUN NESTED CV FOR DECISION TREE
# ---------------------------------------------------------------------------

decision_tree_scores, decision_tree_best_params = (
    nested_cross_validation(
        model_name="Decision Tree",
        pipeline=decision_tree_pipeline,
        search_space=decision_tree_param_grid,
        search_method="grid",
        X=X_dev,
        y=y_dev,
    )
)


# ---------------------------------------------------------------------------
# 9. RUN NESTED CV FOR RANDOM FOREST
# ---------------------------------------------------------------------------

random_forest_scores, random_forest_best_params = (
    nested_cross_validation(
        model_name="Random Forest",
        pipeline=random_forest_pipeline,
        search_space=random_forest_param_distributions,
        search_method="random",
        X=X_dev,
        y=y_dev,
        n_random_iterations=40,
    )
)


# ---------------------------------------------------------------------------
# 10. COMBINE AND SUMMARIZE OUTER SCORES
# ---------------------------------------------------------------------------

nested_scores = pd.concat(
    [
        decision_tree_scores,
        random_forest_scores,
    ],
    ignore_index=True,
)

summary = (
    nested_scores
    .groupby("model")
    .agg(
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
        phishing_precision_mean=("phishing_precision", "mean"),
        phishing_recall_mean=("phishing_recall", "mean"),
        accuracy_mean=("accuracy", "mean"),
        roc_auc_mean=("roc_auc", "mean"),
    )
    .sort_values(
        by="macro_f1_mean",
        ascending=False,
    )
)

print("\nNested cross-validation summary")
print(summary.round(4))


# Save the fold-level results for the report.
nested_scores.to_csv(
    "nested_cv_fold_scores.csv",
    index=False,
)

summary.to_csv(
    "nested_cv_summary.csv",
)


# ---------------------------------------------------------------------------
# 11. FIGURE: OUTER MACRO F1 DISTRIBUTION
# ---------------------------------------------------------------------------

model_order = [
    "Decision Tree",
    "Random Forest",
]

boxplot_values = [
    nested_scores.loc[
        nested_scores["model"] == model_name,
        "macro_f1",
    ].to_numpy()
    for model_name in model_order
]

plt.figure(figsize=(7, 5))

plt.boxplot(
    boxplot_values,
    tick_labels=model_order,
    showmeans=True,
)

# Show the individual outer-fold scores.
for position, values in enumerate(
    boxplot_values,
    start=1,
):
    x_positions = np.full(
        shape=len(values),
        fill_value=position,
        dtype=float,
    )

    plt.scatter(
        x_positions,
        values,
        zorder=3,
    )

plt.ylabel("Outer-fold macro F1-score")
plt.xlabel("Model family")
plt.title("Nested cross-validation scores")
plt.tight_layout()

plt.savefig(
    "nested_cv_model_comparison.pdf",
    bbox_inches="tight",
)

plt.show()


# ---------------------------------------------------------------------------
# 12. SELECT THE BEST MODEL FAMILY
# ---------------------------------------------------------------------------

best_model_family = summary.index[0]

print(
    "\nSelected model family:",
    best_model_family,
)


# ---------------------------------------------------------------------------
# 13. FINAL HYPERPARAMETER SEARCH ON THE COMPLETE DEVELOPMENT SET
# ---------------------------------------------------------------------------

# Nested CV evaluates the selection procedure.
# We now repeat the inner search on the complete development set to obtain
# the configuration used for final training.

final_inner_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE,
)

if best_model_family == "Decision Tree":
    final_search = GridSearchCV(
        estimator=decision_tree_pipeline,
        param_grid=decision_tree_param_grid,
        scoring=PRIMARY_SCORING,
        cv=final_inner_cv,
        refit=True,
        n_jobs=-1,
        return_train_score=False,
        error_score="raise",
    )

elif best_model_family == "Random Forest":
    final_search = RandomizedSearchCV(
        estimator=random_forest_pipeline,
        param_distributions=random_forest_param_distributions,
        n_iter=40,
        scoring=PRIMARY_SCORING,
        cv=final_inner_cv,
        refit=True,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        return_train_score=False,
        error_score="raise",
    )

else:
    raise RuntimeError(
        f"Unknown selected model family: {best_model_family}"
    )


# Fit the selected pipeline on the complete development set.
final_search.fit(
    X_dev,
    y_dev,
)

final_model = final_search.best_estimator_

print(
    "\nFinal best development CV score:",
    round(final_search.best_score_, 4),
)

print(
    "Final best hyperparameters:",
    final_search.best_params_,
)


# ---------------------------------------------------------------------------
# 14. FINAL TEST EVALUATION
# ---------------------------------------------------------------------------

# The test set is accessed only here, after model-family selection,
# feature selection, and hyperparameter optimization have been completed.
final_test_metrics = compute_classification_metrics(
    fitted_pipeline=final_model,
    X_validation=X_test,
    y_validation=y_test,
)

print("\nFinal test metrics")

for metric_name, metric_value in final_test_metrics.items():
    print(
        f"{metric_name}: {metric_value:.4f}"
    )