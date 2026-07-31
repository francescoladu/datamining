import pandas as pd


def calculate_dataset_statistics(
    data: pd.DataFrame,
    target_column: str,
    phishing_label: int,
    legitimate_label: int,
) -> pd.DataFrame:
    """
    Calcola le statistiche principali del dataset pulito.
    """
    number_of_rows = len(data)
    number_of_features = len(data.columns) - 1

    class_counts = data[target_column].value_counts()

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

    nonzero_counts = class_counts[class_counts > 0]

    if len(nonzero_counts) >= 2:
        imbalance_ratio = (
            nonzero_counts.max()
            / nonzero_counts.min()
        )
    else:
        imbalance_ratio = float("nan")

    return pd.DataFrame(
        [
            ("Observations", number_of_rows),
            ("Predictive features", number_of_features),
            ("Missing values", int(data.isnull().sum().sum())),
            ("Duplicated rows", int(data.duplicated().sum())),
            ("Phishing observations", phishing_count),
            ("Phishing (%)", round(phishing_percentage, 2)),
            ("Legitimate observations", legitimate_count),
            ("Legitimate (%)", round(legitimate_percentage, 2)),
            ("Imbalance ratio", round(imbalance_ratio, 3)),
        ],
        columns=["Statistic", "Value"],
    )


def find_conflicting_profiles(
    data: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Cerca vettori di feature identici associati
    a etichette target differenti.
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