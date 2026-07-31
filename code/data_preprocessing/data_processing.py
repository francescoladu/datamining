from pathlib import Path
from collections.abc import Sequence

import pandas as pd


def load_clean_dataset(
    file_path: str | Path,
    index_columns: Sequence[str] = ("index", "Unnamed: 0"),
) -> pd.DataFrame:
    """
    Carica il dataset deduplicato da un file CSV.

    Rimuove eventuali colonne indice che non rappresentano
    vere feature del dataset.
    """
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
) -> None:
    """
    Controlla che il dataset possa essere utilizzato
    nelle analisi statistiche.
    """
    if data.empty:
        raise ValueError("Il dataset è vuoto.")

    if target_column not in data.columns:
        raise ValueError(
            f"La colonna target '{target_column}' non è presente."
        )

    duplicated_columns = data.columns[
        data.columns.duplicated()
    ].tolist()

    if duplicated_columns:
        raise ValueError(
            "Sono presenti colonne duplicate: "
            f"{duplicated_columns}"
        )

    missing_values = int(data.isnull().sum().sum())

    if missing_values > 0:
        raise ValueError(
            f"Il dataset contiene {missing_values} valori mancanti."
        )

    feature_columns = [
        column
        for column in data.columns
        if column != target_column
    ]

    non_numeric_features = [
        column
        for column in feature_columns
        if not pd.api.types.is_numeric_dtype(data[column])
    ]

    if non_numeric_features:
        raise TypeError(
            "La correlazione di Pearson richiede feature numeriche. "
            "Colonne non numeriche trovate: "
            f"{non_numeric_features}"
        )


def split_features_target(
    data: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separa le feature predittive dalla variabile target.

    Result non viene quindi inclusa nella matrice
    di correlazione tra feature.
    """
    X = data.drop(
        columns=[target_column]
    ).copy()

    y = data[target_column].copy()

    return X, y