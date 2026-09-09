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
from shared.config import PHISHING_LABEL


def select_rows(data: Any, indices: np.ndarray) -> Any:
    """Select rows by integer position from pandas or NumPy objects."""
    if hasattr(data, "iloc"):
        return data.iloc[indices]
    return data[indices]


def predict_with_phishing_probability(
    fitted_pipeline: Pipeline,
    X_validation: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Return class predictions and phishing probability (class PHISHING_LABEL)."""
    y_pred = np.asarray(fitted_pipeline.predict(X_validation))
    classifier = fitted_pipeline.named_steps["classifier"]

    phishing_positions = np.flatnonzero(classifier.classes_ == PHISHING_LABEL)
    if phishing_positions.size != 1:
        raise ValueError(
            f"The fitted classifier must contain the phishing class encoded as {PHISHING_LABEL}."
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
    sample_weight: Any | None = None,
) -> dict[str, float]:
    """Compute classification metrics using the provided sample weights."""
    y_pred, phishing_probability = predict_with_phishing_probability(
        fitted_pipeline,
        X_validation,
    )

    y_val_array = np.asarray(y_validation)
    y_phishing_binary = (y_val_array == PHISHING_LABEL).astype(int)
    sw = np.asarray(sample_weight) if sample_weight is not None else None

    return {
        "macro_f1": f1_score(
            y_val_array,
            y_pred,
            average="macro",
            sample_weight=sw,
        ),
        "phishing_precision": precision_score(
            y_val_array,
            y_pred,
            pos_label=PHISHING_LABEL,
            zero_division=0,
            sample_weight=sw,
        ),
        "phishing_recall": recall_score(
            y_val_array,
            y_pred,
            pos_label=PHISHING_LABEL,
            zero_division=0,
            sample_weight=sw,
        ),
        "accuracy": accuracy_score(
            y_val_array,
            y_pred,
            sample_weight=sw,
        ),
        "roc_auc": roc_auc_score(
            y_phishing_binary,
            phishing_probability,
            sample_weight=sw,
        ),
    }


def select_by_one_se_rule(cv_results: dict[str, Any]) -> int:
    """
    Select the simplest model within one standard error of the best score.

    Simplicity hierarchy:
      1. Smallest number of features (feature_selection__k)
      2. Smallest tree depth (classifier__max_depth)
      3. Highest validation score
    """
    mean_scores = np.asarray(cv_results["mean_test_score"], dtype=float)
    std_scores = np.asarray(cv_results["std_test_score"], dtype=float)

    n_splits = len(
        [
            col
            for col in cv_results
            if col.startswith("split") and col.endswith("_test_score")
        ]
    )
    if n_splits == 0:
        n_splits = config.N_INNER_SPLITS

    best_idx = int(np.argmax(mean_scores))
    best_score = float(mean_scores[best_idx])
    best_std = float(std_scores[best_idx])

    best_se = best_std / np.sqrt(n_splits)
    threshold = best_score - best_se

    candidate_indices = np.where(mean_scores >= threshold)[0]

    def complexity_key(idx: int) -> tuple[int, int, float]:
        params = cv_results["params"][idx]
        k_value = params.get("feature_selection__k", 999)
        k_value = 999 if k_value == "all" else int(k_value)
        depth = params.get("classifier__max_depth", 999)
        depth = 999 if depth is None else int(depth)
        negative_score = -float(mean_scores[idx])
        return (k_value, depth, negative_score)

    selected_idx = min(candidate_indices, key=complexity_key)
    return int(selected_idx)
