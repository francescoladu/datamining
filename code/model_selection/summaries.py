from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def summarize_feature_frequency(
    selected_features: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize how often each feature is selected across outer folds."""
    return (
        selected_features
        .groupby(["model", "feature"], as_index=False)
        .agg(
            selected_in_folds=("selected", "sum"),
            selection_frequency=("selected", "mean"),
            mean_mutual_information=("mutual_information_score", "mean"),
            std_mutual_information=("mutual_information_score", "std"),
            mean_mutual_information_rank=("mutual_information_rank", "mean"),
        )
        .sort_values(
            ["model", "selected_in_folds", "mean_mutual_information"],
            ascending=[True, False, False],
        )
    )


def compute_feature_stability(
    selected_features: pd.DataFrame,
) -> pd.DataFrame:
    """Compute pairwise Jaccard similarity between selected feature subsets."""
    rows: list[dict[str, Any]] = []

    for model_name, model_frame in selected_features.groupby("model"):
        feature_sets = {
            int(outer_fold): set(
                fold_frame.loc[fold_frame["selected"], "feature"]
            )
            for outer_fold, fold_frame in model_frame.groupby("outer_fold")
        }

        for (fold_a, features_a), (fold_b, features_b) in combinations(
            sorted(feature_sets.items()),
            2,
        ):
            union = features_a | features_b
            intersection = features_a & features_b
            jaccard = len(intersection) / len(union) if union else 1.0
            rows.append(
                {
                    "model": model_name,
                    "outer_fold_a": fold_a,
                    "outer_fold_b": fold_b,
                    "features_in_a": len(features_a),
                    "features_in_b": len(features_b),
                    "intersection_size": len(intersection),
                    "union_size": len(union),
                    "jaccard_similarity": jaccard,
                }
            )

    return pd.DataFrame(rows)


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
        values="macro_f1",
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
