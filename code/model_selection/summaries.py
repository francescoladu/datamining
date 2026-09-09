from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def summarize_permutation_importance(
    permutation_scores: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate outer-fold permutation importance for each feature."""
    if permutation_scores.empty:
        return pd.DataFrame()

    return (
        permutation_scores
        .groupby(["model", "feature"], as_index=False)
        .agg(
            selected_in_folds=("selected", "sum"),
            mean_importance=("importance_mean", "mean"),
            std_importance_across_folds=("importance_mean", "std"),
            mean_within_fold_std=("importance_std", "mean"),
        )
        .sort_values(["model", "mean_importance"], ascending=[True, False])
    )


def compute_statistical_tests(
    nested_scores: pd.DataFrame,
) -> pd.DataFrame:
    """Compute the paired Wilcoxon signed-rank test."""
    rows: list[dict[str, Any]] = []
    fold_pivot = nested_scores.pivot(
        index="outer_fold",
        columns="model",
        values="accuracy",
    )

    if {"Decision Tree", "Random Forest"}.issubset(fold_pivot.columns):
        paired = fold_pivot[["Decision Tree", "Random Forest"]].dropna()
        differences = paired["Random Forest"] - paired["Decision Tree"]

        if np.allclose(differences.to_numpy(), 0.0):
            statistic = 0.0
            p_value = 1.0
        else:
            try:
                result = wilcoxon(
                    paired["Decision Tree"],
                    paired["Random Forest"],
                    alternative="two-sided",
                    method="exact",
                )
            except ValueError:
                result = wilcoxon(
                    paired["Decision Tree"],
                    paired["Random Forest"],
                    alternative="two-sided",
                    method="auto",
                )
            statistic = float(result.statistic)
            p_value = float(result.pvalue)

        rows.append(
            {
                "test": "Wilcoxon signed-rank",
                "statistic": statistic,
                "p_value": p_value,
                "sample_size": len(paired),
                "mean_paired_difference_rf_minus_dt": float(
                    differences.mean()
                ),
            }
        )

    return pd.DataFrame(rows)
