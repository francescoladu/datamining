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
from config import N_INNER_SPLITS

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
    Select the simplest model within 1 standard error of the best score.
    Simplicity priority:
      1. Smallest number of features (feature_selection__k)
      2. Smallest tree depth (classifier__max_depth, if present)
      3. Highest validation score as tie-breaker
    """
    mean_scores = np.asarray(cv_results["mean_test_score"])
    std_scores = np.asarray(cv_results["std_test_score"])

    # Count how many CV folds were evaluated
    n_splits = len([col for col in cv_results if col.startswith("split") and col.endswith("_test_score")])
    if n_splits == 0:
        n_splits = N_INNER_SPLITS  # fallback to default inner splits

    # 1. Best performing model index and threshold
    best_idx = int(np.argmax(mean_scores))
    best_score = mean_scores[best_idx]
    best_std = std_scores[best_idx]
    
    # Standard Error = std / sqrt(n_splits)
    best_se = best_std / np.sqrt(n_splits)
    threshold = best_score - best_se

    # 2. Find all candidate configurations within [best_score - 1*SE, best_score]
    candidate_indices = np.where(mean_scores >= threshold)[0]

    # 3. Sort candidates by parsimony (simplest first)
    def complexity_key(idx: int) -> tuple[int, int, float]:
        params = cv_results["params"][idx]
        
        # 1st: Feature count (k)
        k_val = params.get("feature_selection__k", 999)
        k_val = 999 if k_val == "all" else int(k_val)
        
        # 2nd: Tree depth (None means unrestricted, so treat as large)
        depth = params.get("classifier__max_depth", 999)
        depth = 999 if depth is None else int(depth)
        
        # 3rd: Negative mean score (so highest score among equal complexity comes first)
        neg_score = -float(mean_scores[idx])
        
        return (k_val, depth, neg_score)

    # Pick the candidate index with the lowest complexity
    selected_idx = min(candidate_indices, key=complexity_key)
    return int(selected_idx)