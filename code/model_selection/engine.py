from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

# Consistent project imports
from model_selection import config
from model_selection.utils import (
    compute_classification_metrics,
    predict_with_phishing_probability,
    select_rows,
)
from shared.modeling import build_pipeline

# Use config.RANDOM_STATE consistently below
outer_cv = StratifiedKFold(
    n_splits=config.N_OUTER_SPLITS, 
    shuffle=True, 
    random_state=config.RANDOM_STATE
)

final_inner_cv = StratifiedKFold(
    n_splits=config.N_INNER_SPLITS, 
    shuffle=True, 
    random_state=config.RANDOM_STATE
)

decision_tree_pipeline = build_pipeline("Decision Tree")
random_forest_pipeline = build_pipeline("Random Forest")

def _extract_prediction_rows(
    *,
    model_name: str,
    outer_fold: int,
    fitted_pipeline: Pipeline,
    X_outer_validation: Any,
    y_outer_validation: Any,
    outer_validation_idx: np.ndarray,
) -> list[dict[str, Any]]:
    """Create one out-of-fold prediction record per validation observation."""
    y_pred, phishing_probability = predict_with_phishing_probability(
        fitted_pipeline,
        X_outer_validation,
    )
    y_true = np.asarray(y_outer_validation)

    # With two classes, P(legitimate) = 1 - P(phishing).
    predicted_confidence = np.where(
        y_pred == -1,
        phishing_probability,
        1.0 - phishing_probability,
    )

    original_indices = (
        np.asarray(X_outer_validation.index)
        if hasattr(X_outer_validation, "index")
        else np.asarray(outer_validation_idx)
    )

    prediction_rows: list[dict[str, Any]] = []
    for position, original_index, truth, prediction, probability, confidence in zip(
        outer_validation_idx,
        original_indices,
        y_true,
        y_pred,
        phishing_probability,
        predicted_confidence,
    ):
        correct = bool(truth == prediction)

        if truth == -1 and prediction == -1:
            error_type = "true_positive_phishing"
        elif truth != -1 and prediction != -1:
            error_type = "true_negative_legitimate"
        elif truth == -1 and prediction != -1:
            error_type = "false_negative"
        else:
            error_type = "false_positive"

        prediction_rows.append(
            {
                "model": model_name,
                "outer_fold": outer_fold,
                "sample_position": int(position),
                "sample_index": original_index,
                "y_true": truth,
                "y_pred": prediction,
                "phishing_probability": float(probability),
                "predicted_confidence": float(confidence),
                "correct": correct,
                "error_type": error_type,
                "high_confidence_error": bool(
                    (not correct)
                    and confidence >= config.HIGH_CONFIDENCE_THRESHOLD
                ),
            }
        )

    return prediction_rows


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
) -> dict[str, pd.DataFrame]:
    """
    Run nested stratified cross-validation and return analysis-ready tables.

    Returned tables include fold-level metrics, out-of-fold predictions, and
    permutation importance measured on untouched outer validation folds.
    """
    if not hasattr(X, "columns"):
        raise TypeError(
            "X must be a pandas DataFrame so feature names can be exported."
        )

    feature_names = list(X.columns)

    fold_results: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    permutation_rows: list[dict[str, Any]] = []

    for outer_fold, (outer_train_idx, outer_validation_idx) in enumerate(
        outer_splits,
        start=1,
    ):
        print(f"{model_name} - outer fold {outer_fold}/{len(outer_splits)}")

        X_outer_train = select_rows(X, outer_train_idx)
        y_outer_train = select_rows(y, outer_train_idx)
        X_outer_validation = select_rows(X, outer_validation_idx)
        y_outer_validation = select_rows(y, outer_validation_idx)

        inner_cv = StratifiedKFold(
            n_splits=config.N_INNER_SPLITS,
            shuffle=True,
            random_state=config.RANDOM_STATE + outer_fold,
        )

        if search_method == "grid":
            inner_search: GridSearchCV | RandomizedSearchCV = GridSearchCV(
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
            raise ValueError("search_method must be either 'grid' or 'random'.")

        inner_search.fit(X_outer_train, y_outer_train)
        best_pipeline = inner_search.best_estimator_

        metrics = compute_classification_metrics(
            fitted_pipeline=best_pipeline,
            X_validation=X_outer_validation,
            y_validation=y_outer_validation,
        )

        selector = best_pipeline.named_steps["feature_selection"]
        selected_k = selector.k
        selected_feature_count = int(np.asarray(selector.get_support()).sum())

        fold_results.append(
            {
                "model": model_name,
                "outer_fold": outer_fold,
                "selected_k": selected_k,
                "selected_feature_count": selected_feature_count,
                "inner_best_macro_f1": float(inner_search.best_score_),
                **metrics,
            }
        )

        prediction_rows.extend(
            _extract_prediction_rows(
                model_name=model_name,
                outer_fold=outer_fold,
                fitted_pipeline=best_pipeline,
                X_outer_validation=X_outer_validation,
                y_outer_validation=y_outer_validation,
                outer_validation_idx=outer_validation_idx,
            )
        )

        if config.COMPUTE_PERMUTATION_IMPORTANCE:
            permutation_result = permutation_importance(
                best_pipeline,
                X_outer_validation,
                y_outer_validation,
                scoring=config.PRIMARY_SCORING,
                n_repeats=config.PERMUTATION_N_REPEATS,
                random_state=config.RANDOM_STATE + outer_fold,
                n_jobs=-1,
            )
            selected_mask = np.asarray(selector.get_support(), dtype=bool)

            for feature_name, is_selected, mean_value, std_value in zip(
                feature_names,
                selected_mask,
                permutation_result.importances_mean,
                permutation_result.importances_std,
            ):
                permutation_rows.append(
                    {
                        "model": model_name,
                        "outer_fold": outer_fold,
                        "feature": feature_name,
                        "selected": bool(is_selected),
                        "importance_mean": float(mean_value),
                        "importance_std": float(std_value),
                    }
                )

        print(f"  Inner best macro F1: {inner_search.best_score_:.4f}")
        print(f"  Outer macro F1: {metrics['macro_f1']:.4f}")
        print(f"  Selected k: {selected_k}")
        print(f"  Best parameters: {inner_search.best_params_}")
        print()

    return {
        "fold_scores": pd.DataFrame(fold_results),
        "oof_predictions": pd.DataFrame(prediction_rows),
        "permutation_importance": pd.DataFrame(permutation_rows),
    }
