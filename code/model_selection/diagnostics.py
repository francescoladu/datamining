from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from model_selection import config
from shared.config import PHISHING_LABEL


def build_error_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    """Create error summary and rates accounting for sample weights."""
    rows: list[dict[str, Any]] = []

    for model_name, frame in predictions.groupby("model"):
        y_true = frame["y_true"].to_numpy()
        y_pred = frame["y_pred"].to_numpy()
        weights = (
            frame["sample_weight"].to_numpy()
            if "sample_weight" in frame.columns
            else np.ones(len(frame))
        )

        tp_mask = (y_true == PHISHING_LABEL) & (y_pred == PHISHING_LABEL)
        fn_mask = (y_true == PHISHING_LABEL) & (y_pred != PHISHING_LABEL)
        fp_mask = (y_true != PHISHING_LABEL) & (y_pred == PHISHING_LABEL)
        tn_mask = (y_true != PHISHING_LABEL) & (y_pred != PHISHING_LABEL)
        err_mask = ~frame["correct"].to_numpy()

        weighted_pos = weights[y_true == PHISHING_LABEL].sum()
        weighted_neg = weights[y_true != PHISHING_LABEL].sum()
        weighted_total = weights.sum()

        fn_weighted = weights[fn_mask].sum()
        fp_weighted = weights[fp_mask].sum()
        err_weighted = weights[err_mask].sum()

        rows.append(
            {
                "model": model_name,
                "unique_profiles": len(frame),
                "retained_weighted_instances": float(weighted_total),
                "true_positive_phishing_mass": float(weights[tp_mask].sum()),
                "false_negative_mass": float(fn_weighted),
                "false_positive_mass": float(fp_weighted),
                "true_negative_legitimate_mass": float(weights[tn_mask].sum()),
                "weighted_false_negative_rate": (
                    fn_weighted / weighted_pos if weighted_pos else np.nan
                ),
                "weighted_false_positive_rate": (
                    fp_weighted / weighted_neg if weighted_neg else np.nan
                ),
                "weighted_error_rate": (
                    err_weighted / weighted_total if weighted_total else np.nan
                ),
                "high_confidence_errors": int(frame["high_confidence_error"].sum()),
                "high_confidence_threshold": config.HIGH_CONFIDENCE_THRESHOLD,
            }
        )

    return pd.DataFrame(rows)


def build_error_by_feature_value(
    predictions: pd.DataFrame,
    X_dev: pd.DataFrame,
) -> pd.DataFrame:
    """Measure error rates broken down by feature value."""
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
                weights = (
                    group["sample_weight"].to_numpy()
                    if "sample_weight" in group.columns
                    else np.ones(len(group))
                )
                err_mask = (~group["correct"]).to_numpy()
                err_weighted = weights[err_mask].sum()
                tot_weighted = weights.sum()

                rows.append(
                    {
                        "model": model_name,
                        "feature": feature,
                        "feature_value": feature_value,
                        "unique_profiles": len(group),
                        "weighted_instances": float(tot_weighted),
                        "weighted_errors": float(err_weighted),
                        "weighted_error_rate": (
                            err_weighted / tot_weighted if tot_weighted else 0.0
                        ),
                        "false_negatives": int((group["error_type"] == "false_negative").sum()),
                        "false_positives": int((group["error_type"] == "false_positive").sum()),
                        "high_confidence_errors": int(group["high_confidence_error"].sum()),
                        "mean_phishing_probability": float(group["phishing_probability"].mean()),
                    }
                )

    return pd.DataFrame(rows)
