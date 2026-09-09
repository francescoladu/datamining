from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from data_preprocessing import config


def load_clean_dataset(
    file_path: str | Path,
    index_columns: Sequence[str] = tuple(config.INDEX_COLUMNS),
) -> pd.DataFrame:
    """Load the deduplicated dataset from a CSV file."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"The file '{file_path}' was not found.")

    data = pd.read_csv(file_path)
    data.drop(
        columns=list(index_columns),
        inplace=True,
        errors="ignore",
    )
    return data


def validate_dataset(
    data: pd.DataFrame,
    target_column: str,
    weight_column: str = config.SAMPLE_WEIGHT_COLUMN,
) -> None:
    """Check that the dataset can be safely used in the analyses that follow."""
    if data.empty:
        raise ValueError("The dataset is empty.")

    if target_column not in data.columns:
        raise ValueError(
            f"The target column '{target_column}' is not present."
        )

    allowed_labels = {
        config.PHISHING_LABEL,
        config.LEGITIMATE_LABEL,
    }

    actual_labels = set(data[target_column].dropna().unique())
    invalid_labels = actual_labels - allowed_labels

    if invalid_labels:
        raise ValueError(
            f"The target column '{target_column}' contains invalid labels: "
            f"{sorted(invalid_labels)}. Expected only {sorted(allowed_labels)}."
        )

    duplicated_columns = data.columns[data.columns.duplicated()].tolist()
    if duplicated_columns:
        raise ValueError(f"Duplicated columns are present: {duplicated_columns}")

    missing_values = int(data.isnull().sum().sum())
    if missing_values > 0:
        raise ValueError(f"The dataset contains {missing_values} missing values.")

    feature_columns = [
        column
        for column in data.columns
        if column not in (target_column, weight_column)
    ]

    if not feature_columns:
        raise ValueError("The dataset does not contain any predictive features.")

    non_numeric_features = [
        column
        for column in feature_columns
        if not pd.api.types.is_numeric_dtype(data[column])
    ]

    if non_numeric_features:
        raise TypeError(
            "Features must be numeric. Non-numeric columns found: "
            f"{non_numeric_features}"
        )


def split_features_target(
    data: pd.DataFrame,
    target_column: str = config.TARGET_COLUMN,
    weight_column: str = config.SAMPLE_WEIGHT_COLUMN,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Separate the predictive features from the target variable and sample weights."""
    drop_cols = [target_column]
    if weight_column in data.columns:
        drop_cols.append(weight_column)
        sample_weight = data[weight_column].astype(float).copy()
    else:
        sample_weight = pd.Series(1.0, index=data.index, name=weight_column)

    X = data.drop(columns=drop_cols).copy()
    y = data[target_column].copy()

    return X, y, sample_weight


def load_and_prepare_dataset(
    file_path: str | Path,
    target_column: str = config.TARGET_COLUMN,
    weight_column: str = config.SAMPLE_WEIGHT_COLUMN,
    index_columns: Sequence[str] = tuple(config.INDEX_COLUMNS),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load, validate, and split one configured training dataset."""

    data = load_clean_dataset(
        file_path=file_path,
        index_columns=index_columns,
    )

    validate_dataset(
        data=data,
        target_column=target_column,
        weight_column=weight_column,
    )

    X, y, sample_weight = split_features_target(
        data=data,
        target_column=target_column,
        weight_column=weight_column,
    )

    return data, X, y, sample_weight


def calculate_dataset_statistics(
    data: pd.DataFrame,
    target_column: str,
    weight_column: str,
    phishing_label: int,
    legitimate_label: int,
) -> pd.DataFrame:
    """
    Calculate descriptive statistics that are comparable
    across all three experiments.
    """

    feature_columns = [
        column
        for column in data.columns
        if column not in (target_column, weight_column)
    ]

    has_weights = weight_column in data.columns

    weights = (
        data[weight_column].astype(float)
        if has_weights
        else pd.Series(1.0, index=data.index)
    )

    number_of_rows = len(data)

    number_of_unique_profiles = len(
        data[feature_columns].drop_duplicates()
    )

    total_instance_mass = float(
        weights.sum()
    )

    number_of_features = len(
        feature_columns
    )

    phishing_mask = (
        data[target_column] == phishing_label
    )

    legitimate_mask = (
        data[target_column] == legitimate_label
    )

    phishing_rows = int(
        phishing_mask.sum()
    )

    legitimate_rows = int(
        legitimate_mask.sum()
    )

    phishing_weighted = float(
        weights[phishing_mask].sum()
    )

    legitimate_weighted = float(
        weights[legitimate_mask].sum()
    )

    phishing_pct = (
        100 * phishing_weighted / total_instance_mass
        if total_instance_mass > 0
        else 0.0
    )

    legitimate_pct = (
        100 * legitimate_weighted / total_instance_mass
        if total_instance_mass > 0
        else 0.0
    )

    minimum_class_mass = min(
        phishing_weighted,
        legitimate_weighted,
    )

    imbalance_ratio = (
        max(
            phishing_weighted,
            legitimate_weighted,
        )
        / minimum_class_mass
        if minimum_class_mass > 0
        else np.nan
    )

    statistics = pd.DataFrame(
        [
            (
                "Rows",
                number_of_rows,
            ),
            (
                "Unique feature profiles",
                number_of_unique_profiles,
            ),
            (
                "Total observation mass",
                total_instance_mass,
            ),
            (
                "Predictive features",
                number_of_features,
            ),
            (
                "Missing values",
                int(data.isnull().sum().sum()),
            ),
            (
                "Phishing rows (unweighted)",
                phishing_rows,
            ),
            (
                "Legitimate rows (unweighted)",
                legitimate_rows,
            ),
            (
                "Phishing observation mass",
                phishing_weighted,
            ),
            (
                "Phishing (%)",
                round(phishing_pct, 2),
            ),
            (
                "Legitimate observation mass",
                legitimate_weighted,
            ),
            (
                "Legitimate (%)",
                round(legitimate_pct, 2),
            ),
            (
                "Imbalance ratio",
                (
                    round(float(imbalance_ratio), 3)
                    if not np.isnan(imbalance_ratio)
                    else np.nan
                ),
            ),
        ],
        columns=[
            "Statistic",
            "Value",
        ],
    )

    return statistics


def find_conflicting_profiles(
    data: pd.DataFrame,
    target_column: str,
    weight_column: str = config.SAMPLE_WEIGHT_COLUMN,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Look for identical predictor profiles with different labels."""
    feature_columns = [
        column
        for column in data.columns
        if column not in (target_column, weight_column)
    ]

    profile_summary = (
        data.groupby(feature_columns, dropna=False)[target_column]
        .agg(
            number_of_labels="nunique",
            number_of_rows="count",
        )
        .reset_index()
    )

    conflicting_profiles = profile_summary[
        profile_summary["number_of_labels"] > 1
    ].copy()

    labels_per_profile = data.groupby(feature_columns, dropna=False)[
        target_column
    ].transform("nunique")

    conflicting_rows = data[labels_per_profile > 1].copy()
    conflicting_percentage = (
        100 * len(conflicting_rows) / len(data) if len(data) > 0 else 0.0
    )

    statistics = pd.DataFrame(
        [
            (
                "Conflicting predictor profiles",
                len(conflicting_profiles),
            ),
            (
                "Rows in conflicting profiles",
                len(conflicting_rows),
            ),
            (
                "Conflicting rows (%)",
                round(conflicting_percentage, 2),
            ),
        ],
        columns=["Statistic", "Value"],
    )

    return statistics, conflicting_profiles, conflicting_rows


def calculate_mutual_information(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = config.RANDOM_STATE,
) -> pd.DataFrame:
    """Calculate mutual information between discrete features and the target."""
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

    results = results.sort_values(
        by="Mutual Information",
        ascending=False,
    ).reset_index(drop=True)

    results.insert(
        0,
        "MI Rank",
        range(1, len(results) + 1),
    )

    return results


def calculate_spearman_correlation_matrix(
    X: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate the Spearman rank correlation matrix between predictive features."""
    return X.corr(method="spearman")


def find_strongest_correlations(
    correlation_matrix: pd.DataFrame,
    top_n: int = config.TOP_CORRELATIONS_NUMBER,
) -> pd.DataFrame:
    """Return the feature pairs with the highest absolute Spearman correlation."""
    feature_names = correlation_matrix.columns.tolist()
    correlations = []

    for first_index in range(len(feature_names)):
        for second_index in range(
            first_index + 1,
            len(feature_names),
        ):
            first_feature = feature_names[first_index]
            second_feature = feature_names[second_index]

            correlation = correlation_matrix.iloc[
                first_index,
                second_index,
            ]

            if pd.isna(correlation):
                continue

            correlation = float(correlation)

            correlations.append(
                {
                    "Feature 1": first_feature,
                    "Feature 2": second_feature,
                    "Spearman correlation": correlation,
                    "Absolute correlation": abs(correlation),
                }
            )

    if not correlations:
        return pd.DataFrame(
            columns=[
                "Feature 1",
                "Feature 2",
                "Spearman correlation",
                "Absolute correlation",
            ]
        )

    results = pd.DataFrame(correlations)
    results = results.sort_values(
        by="Absolute correlation",
        ascending=False,
    ).head(top_n).reset_index(drop=True)

    return results
