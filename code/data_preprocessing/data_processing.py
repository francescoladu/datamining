from pathlib import Path
from collections.abc import Sequence

import numpy as np
import pandas as pd

from sklearn.feature_selection import mutual_info_classif

import config


# ============================================================
# 1. DATASET LOADING
# ============================================================

def load_clean_dataset(
    file_path: str | Path,
    index_columns: Sequence[str] = tuple(config.INDEX_COLUMNS),
) -> pd.DataFrame:
    """
    Load the deduplicated dataset from a CSV file.

    Drops any index columns that are not real features of the
    dataset (e.g. leftovers from a previous CSV export).
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"The file '{file_path}' was not found."
        )

    data = pd.read_csv(file_path)

    data.drop(
        columns=list(index_columns),
        inplace=True,
        errors="ignore",
    )

    return data


# ============================================================
# 2. DATASET VALIDATION
# ============================================================

def validate_dataset(
    data: pd.DataFrame,
    target_column: str,
) -> None:
    """
    Check that the dataset can be safely used in the analyses
    that follow.
    """
    if data.empty:
        raise ValueError("The dataset is empty.")

    if target_column not in data.columns:
        raise ValueError(
            f"The target column '{target_column}' is not present."
        )

    duplicated_columns = data.columns[
        data.columns.duplicated()
    ].tolist()

    if duplicated_columns:
        raise ValueError(
            "Duplicated columns are present: "
            f"{duplicated_columns}"
        )

    missing_values = int(
        data.isnull().sum().sum()
    )

    if missing_values > 0:
        raise ValueError(
            f"The dataset contains {missing_values} missing values."
        )

    feature_columns = [
        column
        for column in data.columns
        if column != target_column
    ]

    if not feature_columns:
        raise ValueError(
            "The dataset does not contain any predictive features."
        )

    non_numeric_features = [
        column
        for column in feature_columns
        if not pd.api.types.is_numeric_dtype(data[column])
    ]

    if non_numeric_features:
        raise TypeError(
            "Pearson correlation requires numeric features. "
            "Non-numeric columns found: "
            f"{non_numeric_features}"
        )


# ============================================================
# 3. FEATURE/TARGET SPLIT
# ============================================================

def split_features_target(
    data: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate the predictive features from the target variable.

    The target column will not be included in the feature
    correlation matrix.
    """
    X = data.drop(
        columns=[target_column]
    ).copy()

    y = data[target_column].copy()

    return X, y


# ============================================================
# 4. LOAD + VALIDATE + SPLIT 
# ============================================================

def load_and_prepare_dataset(
    dataset_name: str,
    target_column: str = config.TARGET_COLUMN,
    index_columns: Sequence[str] = tuple(config.INDEX_COLUMNS),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Load, validate, and split one of the configured dataset
    splits (e.g. "train" or "test").

    Returns the full DataFrame together with the separated
    features (X) and target (y), so callers can run both the
    dataset-level and feature-level analyses without repeating
    the loading logic.
    """
    if dataset_name not in config.DATASET_PATHS:
        raise KeyError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available options: {list(config.DATASET_PATHS)}"
        )

    file_path = config.DATASET_PATHS[dataset_name]

    data = load_clean_dataset(
        file_path=file_path,
        index_columns=index_columns,
    )

    validate_dataset(
        data=data,
        target_column=target_column,
    )

    X, y = split_features_target(
        data=data,
        target_column=target_column,
    )

    return data, X, y


# ============================================================
# 5. GENERAL DATASET STATISTICS
# ============================================================

def calculate_dataset_statistics(
    data: pd.DataFrame,
    target_column: str,
    phishing_label: int,
    legitimate_label: int,
) -> pd.DataFrame:
    """
    Calculate the main descriptive statistics of the clean
    dataset.
    """
    number_of_rows = len(data)
    number_of_features = len(data.columns) - 1

    class_counts = data[
        target_column
    ].value_counts()

    phishing_count = int(
        class_counts.get(phishing_label, 0)
    )

    legitimate_count = int(
        class_counts.get(legitimate_label, 0)
    )

    phishing_percentage = (
        100 * phishing_count / number_of_rows
        if number_of_rows > 0
        else 0.0
    )

    legitimate_percentage = (
        100 * legitimate_count / number_of_rows
        if number_of_rows > 0
        else 0.0
    )

    nonzero_counts = class_counts[
        class_counts > 0
    ]

    if len(nonzero_counts) >= 2:
        imbalance_ratio = (
            nonzero_counts.max()
            / nonzero_counts.min()
        )
    else:
        imbalance_ratio = np.nan

    statistics = pd.DataFrame(
        [
            ("Observations", number_of_rows),
            ("Predictive features", number_of_features),
            (
                "Missing values",
                int(data.isnull().sum().sum()),
            ),
            (
                "Duplicated rows",
                int(data.duplicated().sum()),
            ),
            (
                "Phishing observations",
                phishing_count,
            ),
            (
                "Phishing (%)",
                round(phishing_percentage, 2),
            ),
            (
                "Legitimate observations",
                legitimate_count,
            ),
            (
                "Legitimate (%)",
                round(legitimate_percentage, 2),
            ),
            (
                "Imbalance ratio",
                round(float(imbalance_ratio), 3)
                if not np.isnan(imbalance_ratio)
                else np.nan,
            ),
        ],
        columns=["Statistic", "Value"],
    )

    return statistics


# ============================================================
# 6. CONFLICTING LABEL PROFILES
# ============================================================

def find_conflicting_profiles(
    data: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Look for identical feature vectors that are associated with
    different target labels.
    """
    feature_columns = [
        column
        for column in data.columns
        if column != target_column
    ]

    profile_summary = (
        data
        .groupby(
            feature_columns,
            dropna=False,
        )[target_column]
        .agg(
            number_of_labels="nunique",
            number_of_rows="count",
        )
        .reset_index()
    )

    conflicting_profiles = profile_summary[
        profile_summary["number_of_labels"] > 1
    ].copy()

    labels_per_profile = (
        data
        .groupby(
            feature_columns,
            dropna=False,
        )[target_column]
        .transform("nunique")
    )

    conflicting_rows = data[
        labels_per_profile > 1
    ].copy()

    conflicting_percentage = (
        100 * len(conflicting_rows) / len(data)
        if len(data) > 0
        else 0.0
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

    return (
        statistics,
        conflicting_profiles,
        conflicting_rows,
    )


# ============================================================
# 7. MUTUAL INFORMATION
# ============================================================

def calculate_mutual_information(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = config.RANDOM_STATE,
) -> pd.DataFrame:
    """
    Calculate the mutual information between each feature and
    the target variable.
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


# ============================================================
# 8. PEARSON CORRELATION MATRIX
# ============================================================

def calculate_pearson_correlation_matrix(
    X: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate the Pearson correlation matrix between all
    predictive features.

    The target variable must already have been excluded from X.
    """
    correlation_matrix = X.corr(
        method="pearson"
    )

    return correlation_matrix


# ============================================================
# 9. STRONGEST CORRELATIONS
# ============================================================

def find_strongest_correlations(
    correlation_matrix: pd.DataFrame,
    top_n: int = config.TOP_CORRELATIONS_NUMBER,
) -> pd.DataFrame:
    """
    Return the feature pairs with the highest absolute
    correlation.

    Excludes:
    - the diagonal;
    - duplicated pairs;
    - undefined correlations.
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
                    "Pearson correlation": correlation,
                    "Absolute correlation": abs(correlation),
                }
            )

    if not correlations:
        return pd.DataFrame(
            columns=[
                "Feature 1",
                "Feature 2",
                "Pearson correlation",
                "Absolute correlation",
            ]
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


# ============================================================
# 10. PHISHING RATE BY FEATURE VALUE
# ============================================================

def calculate_class_rate_by_feature_value(
    data: pd.DataFrame,
    features: Sequence[str],
    target_column: str,
    class_label: int,
    possible_values: Sequence[int] = (-1, 0, 1),
) -> pd.DataFrame:
    """
    Calculate the percentage of observations belonging to a
    class for each value taken by the given features.
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
                class_rate = 100 * (
                    selected_rows == class_label
                ).mean()

                rate_table.loc[
                    feature,
                    feature_value,
                ] = class_rate

    rate_table.index.name = "Feature"
    rate_table.columns.name = "Feature value"

    return rate_table