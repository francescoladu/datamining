from typing import Any
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline

import config
from utils import select_rows, compute_classification_metrics


def nested_cross_validation(
    *,
    model_name: str,
    pipeline: Pipeline,
    search_space: dict[str, list[Any]],
    search_method: str,
    X: Any,
    y: Any,
    outer_splits: list[tuple[np.ndarray, np.ndarray]],
    n_random_iterations: int = config.N_RANDOM_ITERATIONS,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """
    Run nested stratified cross-validation for a specific model family.

    Inner CV:
        Iteratively selects the optimal number of features and classifier 
        hyperparameters using Grid Search or Randomized Search.

    Outer CV:
        Evaluates the best selected pipeline configuration on untouched test folds 
        that were excluded from both feature and parameter selection.
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

        # Slice training and validation sets for this outer fold
        X_outer_train = select_rows(X, outer_train_idx)
        y_outer_train = select_rows(y, outer_train_idx)

        X_outer_validation = select_rows(X, outer_validation_idx)
        y_outer_validation = select_rows(y, outer_validation_idx)

        # Re-initialize the inner CV split on each outer fold to vary fold alignments
        inner_cv = StratifiedKFold(
            n_splits=4,
            shuffle=True,
            random_state=config.RANDOM_STATE + outer_fold,
        )

        # Build search objects dynamically based on the requested method
        if search_method == "grid":
            inner_search = GridSearchCV(
                estimator=pipeline,
                param_grid=search_space,
                scoring=config.PRIMARY_SCORING,
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
                scoring=config.PRIMARY_SCORING,
                cv=inner_cv,
                refit=True,
                random_state=config.RANDOM_STATE + outer_fold,
                n_jobs=-1,
                return_train_score=False,
                error_score="raise",
            )

        else:
            raise ValueError(
                "search_method must be either 'grid' or 'random'."
            )

        # Train the hyperparameter search wrapper on the outer training fold
        # (This automatically computes feature selection on training splits only)
        inner_search.fit(
            X_outer_train,
            y_outer_train,
        )

        best_pipeline = inner_search.best_estimator_

        # Evaluate the optimized pipeline on the untouched validation fold
        metrics = compute_classification_metrics(
            fitted_pipeline=best_pipeline,
            X_validation=X_outer_validation,
            y_validation=y_outer_validation,
        )

        # Record metrics and hyperparameter targets for auditing
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