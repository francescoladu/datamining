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