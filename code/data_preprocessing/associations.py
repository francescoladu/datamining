from collections.abc import Sequence

import numpy as np
import pandas as pd

from sklearn.feature_selection import mutual_info_classif


def calculate_mutual_information(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Calcola la mutual information tra ogni feature
    e la variabile target.
    """
    mutual_information_values = mutual_info_classif(
        X,
        y,
        discrete_features=True,
        random_state=random_state,
    )

    results = pd.DataFrame(
        {
            "Feature": X.columns,
            "Mutual Information": mutual_information_values,
        }
    )

    results = (
        results
        .sort_values(
            by="Mutual Information",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    results.insert(
        0,
        "MI Rank",
        range(1, len(results) + 1),
    )

    return results


def calculate_pearson_correlation_matrix(
    X: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calcola la matrice di correlazione di Pearson
    tra tutte le feature predittive.

    La variabile target deve essere già stata rimossa da X.
    """
    return X.corr(
        method="pearson"
    )


def find_strongest_correlations(
    correlation_matrix: pd.DataFrame,
    top_n: int = 15,
) -> pd.DataFrame:
    """
    Restituisce le coppie di feature con la correlazione
    assoluta più elevata.

    Le coppie duplicate e la diagonale vengono escluse.
    """
    feature_names = correlation_matrix.columns.tolist()

    correlations = []

    for first_index in range(len(feature_names)):
        for second_index in range(
            first_index + 1,
            len(feature_names),
        ):
            first_feature = feature_names[first_index]
            second_feature = feature_names[second_index]

            correlation = float(
                correlation_matrix.iloc[
                    first_index,
                    second_index,
                ]
            )

            correlations.append(
                {
                    "Feature 1": first_feature,
                    "Feature 2": second_feature,
                    "Pearson correlation": correlation,
                    "Absolute correlation": abs(correlation),
                }
            )

    results = pd.DataFrame(correlations)

    results = (
        results
        .sort_values(
            by="Absolute correlation",
            ascending=False,
        )
        .head(top_n)
        .reset_index(drop=True)
    )

    return results


def calculate_class_rate_by_feature_value(
    data: pd.DataFrame,
    features: Sequence[str],
    target_column: str,
    class_label: int,
    possible_values: Sequence[int] = (-1, 0, 1),
) -> pd.DataFrame:
    """
    Calcola la percentuale di osservazioni appartenenti
    a una determinata classe per ogni valore delle feature.
    """
    rate_table = pd.DataFrame(
        index=list(features),
        columns=list(possible_values),
        dtype=float,
    )

    for feature in features:
        for feature_value in possible_values:
            selected_rows = data.loc[
                data[feature] == feature_value,
                target_column,
            ]

            if not selected_rows.empty:
                rate_table.loc[
                    feature,
                    feature_value,
                ] = 100 * (
                    selected_rows == class_label
                ).mean()

    rate_table.index.name = "Feature"
    rate_table.columns.name = "Feature value"

    return rate_table