from typing import Any
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def select_rows(data: Any, indices: np.ndarray) -> Any:
    """
    Select rows from a pandas DataFrame/Series or a NumPy array based on integer indices.

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


def compute_classification_metrics(
    fitted_pipeline: Pipeline,
    X_validation: Any,
    y_validation: Any,
) -> dict[str, float]:
    """
    Compute key classification evaluation metrics on a validation fold.

    Phishing is encoded as -1 and is treated as the positive class for
    precision, recall, and ROC-AUC.
    """
    y_pred = fitted_pipeline.predict(X_validation)

    classifier = fitted_pipeline.named_steps["classifier"]

    # Locate the probability column corresponding specifically to the phishing class (-1).
    phishing_class_index = int(
        np.flatnonzero(classifier.classes_ == -1)[0]
    )

    phishing_probability = fitted_pipeline.predict_proba(
        X_validation
    )[:, phishing_class_index]

    # ROC-AUC expects a binary indicator (0 or 1) where the target class (-1) maps to 1.
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