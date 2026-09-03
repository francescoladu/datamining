from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from model_selection import config


def select_rows(data: Any, indices: np.ndarray) -> Any:
    """Select rows by integer position from pandas or NumPy objects."""
    if hasattr(data, "iloc"):
        return data.iloc[indices]
    return data[indices]


def predict_with_phishing_probability(
    fitted_pipeline: Pipeline,
    X_validation: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return class predictions and the probability assigned to phishing.

    Phishing is encoded as -1.
    """
    y_pred = np.asarray(fitted_pipeline.predict(X_validation))
    classifier = fitted_pipeline.named_steps["classifier"]

    phishing_positions = np.flatnonzero(classifier.classes_ == -1)
    if phishing_positions.size != 1:
        raise ValueError(
            "The fitted classifier must contain the phishing class encoded as -1."
        )

    phishing_class_index = int(phishing_positions[0])
    phishing_probability = np.asarray(
        fitted_pipeline.predict_proba(X_validation)[:, phishing_class_index]
    )

    return y_pred, phishing_probability


def compute_classification_metrics(
    fitted_pipeline: Pipeline,
    X_validation: Any,
    y_validation: Any,
) -> dict[str, float]:
    """Compute the classification metrics used by the project."""
    y_pred, phishing_probability = predict_with_phishing_probability(
        fitted_pipeline,
        X_validation,
    )

    y_validation_array = np.asarray(y_validation)
    y_phishing_binary = (y_validation_array == -1).astype(int)

    return {
        "macro_f1": f1_score(
            y_validation_array,
            y_pred,
            average="macro",
        ),
        "phishing_precision": precision_score(
            y_validation_array,
            y_pred,
            pos_label=-1,
            zero_division=0,
        ),
        "phishing_recall": recall_score(
            y_validation_array,
            y_pred,
            pos_label=-1,
            zero_division=0,
        ),
        "accuracy": accuracy_score(
            y_validation_array,
            y_pred,
        ),
        "roc_auc": roc_auc_score(
            y_phishing_binary,
            phishing_probability,
        ),
    }


def select_by_one_se_rule(cv_results: dict[str, Any]) -> int:
    """
    Select the simplest model within one standard error of the best score.

    Simplicity priority:
      1. Smallest number of features (feature_selection__k)
      2. Smallest tree depth (classifier__max_depth, if present)
      3. Highest validation score as tie-breaker
    """
    mean_scores = np.asarray(cv_results["mean_test_score"], dtype=float)
    std_scores = np.asarray(cv_results["std_test_score"], dtype=float)

    # Infer the number of CV folds directly from cv_results_.
    n_splits = len(
        [
            column
            for column in cv_results
            if column.startswith("split")
            and column.endswith("_test_score")
        ]
    )
    if n_splits == 0:
        n_splits = config.N_INNER_SPLITS

    # Candidate with the highest mean validation score.
    best_idx = int(np.argmax(mean_scores))
    best_score = float(mean_scores[best_idx])
    best_std = float(std_scores[best_idx])

    # Standard error of the best candidate across the inner folds.
    best_se = best_std / np.sqrt(n_splits)
    threshold = best_score - best_se

    # Candidates whose mean CV score lies within one standard error
    # of the maximum mean CV score.
    candidate_indices = np.where(mean_scores >= threshold)[0]

    def complexity_key(idx: int) -> tuple[int, int, float]:
        """Return a lexicographic simplicity key for one candidate."""
        params = cv_results["params"][idx]

        # First priority: fewer selected features.
        k_value = params.get("feature_selection__k", 999)
        k_value = 999 if k_value == "all" else int(k_value)

        # Second priority: shallower trees.
        depth = params.get("classifier__max_depth", 999)
        depth = 999 if depth is None else int(depth)

        # Third priority: higher mean validation score.
        negative_score = -float(mean_scores[idx])

        return (k_value, depth, negative_score)

    selected_idx = min(candidate_indices, key=complexity_key)
    return int(selected_idx)
