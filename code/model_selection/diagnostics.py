from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from model_selection import config


def build_error_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    """Create confusion-matrix counts and error rates from predictions."""
    rows: list[dict[str, Any]] = []

    for model_name, frame in predictions.groupby("model"):
        y_true = frame["y_true"].to_numpy()
        y_pred = frame["y_pred"].to_numpy()

        true_positives = int(((y_true == -1) & (y_pred == -1)).sum())
        false_negatives = int(((y_true == -1) & (y_pred != -1)).sum())
        false_positives = int(((y_true != -1) & (y_pred == -1)).sum())
        true_negatives = int(((y_true != -1) & (y_pred != -1)).sum())

        positive_total = true_positives + false_negatives
        negative_total = true_negatives + false_positives

        rows.append(
            {
                "model": model_name,
                "observations": len(frame),
                "true_positive_phishing": true_positives,
                "false_negative": false_negatives,
                "false_positive": false_positives,
                "true_negative_legitimate": true_negatives,
                "false_negative_rate": (
                    false_negatives / positive_total if positive_total else np.nan
                ),
                "false_positive_rate": (
                    false_positives / negative_total if negative_total else np.nan
                ),
                "total_errors": int((~frame["correct"]).sum()),
                "error_rate": float((~frame["correct"]).mean()),
                "high_confidence_errors": int(
                    frame["high_confidence_error"].sum()
                ),
                "high_confidence_threshold": config.HIGH_CONFIDENCE_THRESHOLD,
            }
        )

    return pd.DataFrame(rows)


def build_error_by_feature_value(
    predictions: pd.DataFrame,
    X_dev: pd.DataFrame,
) -> pd.DataFrame:
    """Measure OOF error rates for every feature value and model."""
    feature_table = X_dev.reset_index(drop=False).rename(
        columns={"index": "original_dataframe_index"}
    )
    feature_table.insert(0, "sample_position", np.arange(len(feature_table)))

    rows: list[dict[str, Any]] = []
    for model_name, model_predictions in predictions.groupby("model"):
        merged = model_predictions.merge(
            feature_table,
            on="sample_position",
            how="left",
            validate="one_to_one",
        )

        for feature in X_dev.columns:
            for feature_value, group in merged.groupby(feature, dropna=False):
                observations = len(group)
                errors = int((~group["correct"]).sum())
                rows.append(
                    {
                        "model": model_name,
                        "feature": feature,
                        "feature_value": feature_value,
                        "observations": observations,
                        "errors": errors,
                        "error_rate": errors / observations,
                        "false_negatives": int(
                            (group["error_type"] == "false_negative").sum()
                        ),
                        "false_positives": int(
                            (group["error_type"] == "false_positive").sum()
                        ),
                        "high_confidence_errors": int(
                            group["high_confidence_error"].sum()
                        ),
                        "mean_phishing_probability": float(
                            group["phishing_probability"].mean()
                        ),
                    }
                )

    return pd.DataFrame(rows)
