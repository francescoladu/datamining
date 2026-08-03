from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from config import (
    MODEL_PATH,
    RUN_NAME,
    SAMPLE_POSITION,
    SELECTED_FEATURES_PATH,
    TARGET_COLUMN,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
    create_output_directory,
)
from permutation_importance import run_permutation_importance
from plots import run_permutation_importance_plot
from shap_force import run_shap_force


def _require_file(
    file_path: Path,
    description: str,
) -> None:
    """Check that a required input file exists."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"{description} was not found:\n{file_path}"
        )

    if not file_path.is_file():
        raise FileNotFoundError(
            f"{description} is not a valid file:\n{file_path}"
        )


def load_model(
    model_path: Path = MODEL_PATH,
) -> Any:
    """Load the final trained model or pipeline."""

    _require_file(
        file_path=model_path,
        description="Final trained model",
    )

    try:
        model = joblib.load(model_path)

    except Exception as error:
        raise RuntimeError(
            "The final model could not be loaded from:\n"
            f"{model_path}"
        ) from error

    if not hasattr(model, "predict"):
        raise TypeError(
            "The loaded object does not provide a predict() method."
        )

    if not hasattr(model, "predict_proba"):
        raise TypeError(
            "The loaded object does not provide a "
            "predict_proba() method."
        )

    return model


def load_dataset(
    dataset_path: Path,
    dataset_name: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load a cleaned dataset and separate features from the target.

    Parameters
    ----------
    dataset_path:
        Path of the cleaned CSV dataset.

    dataset_name:
        Human-readable dataset name used in error messages.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.Series]
        Feature matrix and target vector.
    """

    _require_file(
        file_path=dataset_path,
        description=dataset_name,
    )

    dataset = pd.read_csv(dataset_path)

    if dataset.empty:
        raise ValueError(
            f"{dataset_name} is empty."
        )

    if dataset.columns.has_duplicates:
        duplicated_columns = dataset.columns[
            dataset.columns.duplicated()
        ].tolist()

        raise ValueError(
            f"{dataset_name} contains duplicated columns: "
            f"{duplicated_columns}"
        )

    if TARGET_COLUMN not in dataset.columns:
        raise ValueError(
            f"Target column {TARGET_COLUMN!r} was not found in "
            f"{dataset_name}.\n"
            f"Available columns: {dataset.columns.tolist()}"
        )

    X = dataset.drop(
        columns=[TARGET_COLUMN]
    )

    y = dataset[TARGET_COLUMN].copy()

    if X.empty:
        raise ValueError(
            f"{dataset_name} does not contain any feature."
        )

    if y.isna().any():
        raise ValueError(
            f"{dataset_name} contains missing target values."
        )

    return X, y


def _convert_to_boolean_mask(
    values: pd.Series,
) -> pd.Series:
    """Convert a CSV selection column into a Boolean mask."""

    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)

    if pd.api.types.is_numeric_dtype(values):
        return values.fillna(0).astype(float) != 0

    normalized_values = (
        values
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return normalized_values.isin({
        "true",
        "1",
        "yes",
        "y",
        "selected",
    })


def load_selected_features(
    selected_features_path: Path = SELECTED_FEATURES_PATH,
) -> list[str]:
    """
    Load the feature names selected during model selection.

    The function supports common column names such as:
    - feature
    - feature_name
    - selected_feature

    If a Boolean column called selected or is_selected exists,
    only rows marked as selected are retained.
    """

    _require_file(
        file_path=selected_features_path,
        description="Selected-features file",
    )

    selected_features_df = pd.read_csv(
        selected_features_path
    )

    if selected_features_df.empty:
        raise ValueError(
            "The selected-features file is empty."
        )

    normalized_columns = {
        str(column).strip().lower(): column
        for column in selected_features_df.columns
    }

    selection_column = None

    for candidate in (
        "selected",
        "is_selected",
        "keep",
        "included",
    ):
        if candidate in normalized_columns:
            selection_column = normalized_columns[candidate]
            break

    if selection_column is not None:
        selection_mask = _convert_to_boolean_mask(
            selected_features_df[selection_column]
        )

        selected_features_df = selected_features_df.loc[
            selection_mask
        ]

    feature_column = None

    for candidate in (
        "feature",
        "feature_name",
        "selected_feature",
        "variable",
        "column",
    ):
        if candidate in normalized_columns:
            feature_column = normalized_columns[candidate]
            break

    if feature_column is None:
        if selected_features_df.shape[1] == 1:
            feature_column = selected_features_df.columns[0]

        else:
            raise ValueError(
                "The selected-features file must contain a column "
                "named 'feature', 'feature_name' or "
                "'selected_feature'.\n"
                "Available columns: "
                f"{selected_features_df.columns.tolist()}"
            )

    selected_features = (
        selected_features_df[feature_column]
        .dropna()
        .astype(str)
        .str.strip()
    )

    selected_features = [
        feature
        for feature in selected_features
        if feature
    ]

    # Preserve the original order while removing duplicates.
    selected_features = list(
        dict.fromkeys(selected_features)
    )

    if not selected_features:
        raise ValueError(
            "No selected feature was found in the "
            "selected-features file."
        )

    return selected_features


def get_model_feature_names(
    model: Any,
) -> list[str] | None:
    """
    Read the input feature names stored by scikit-learn.

    A fitted estimator or pipeline usually provides
    feature_names_in_ when it was trained using a DataFrame.
    """

    feature_names = getattr(
        model,
        "feature_names_in_",
        None,
    )

    if feature_names is None:
        return None

    return [
        str(feature)
        for feature in np.asarray(feature_names)
    ]


def resolve_feature_names(
    model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> list[str]:
    """
    Determine the exact feature set expected by the final model.

    Model feature names are preferred because they preserve the
    exact columns and order used during fitting. When the model
    does not store feature names, the selected-features CSV is
    used instead.
    """

    model_feature_names = get_model_feature_names(
        model=model
    )

    if model_feature_names is not None:
        feature_names = model_feature_names

        print(
            "\nFeature names obtained from the trained model."
        )

    else:
        feature_names = load_selected_features()

        print(
            "\nFeature names obtained from:"
            f"\n{SELECTED_FEATURES_PATH}"
        )

    missing_from_train = [
        feature
        for feature in feature_names
        if feature not in X_train.columns
    ]

    missing_from_test = [
        feature
        for feature in feature_names
        if feature not in X_test.columns
    ]

    if missing_from_train:
        raise ValueError(
            "The following model features are missing from the "
            "training dataset:\n"
            f"{missing_from_train}"
        )

    if missing_from_test:
        raise ValueError(
            "The following model features are missing from the "
            "test dataset:\n"
            f"{missing_from_test}"
        )

    return feature_names


def prepare_explainability_data(
    model: Any,
) -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
]:
    """
    Load and prepare the training and test datasets.

    The training features are used as SHAP background data.
    The test features and labels are used for permutation
    importance and local SHAP explanations.
    """

    X_train_all, y_train = load_dataset(
        dataset_path=TRAIN_DATA_PATH,
        dataset_name="Cleaned training dataset",
    )

    X_test_all, y_test = load_dataset(
        dataset_path=TEST_DATA_PATH,
        dataset_name="Cleaned test dataset",
    )

    feature_names = resolve_feature_names(
        model=model,
        X_train=X_train_all,
        X_test=X_test_all,
    )

    X_train = X_train_all.loc[
        :,
        feature_names,
    ].copy()

    X_test = X_test_all.loc[
        :,
        feature_names,
    ].copy()

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError(
            "Training and test features are not aligned."
        )

    if X_train.isna().any().any():
        missing_columns = X_train.columns[
            X_train.isna().any()
        ].tolist()

        raise ValueError(
            "The training features contain missing values in: "
            f"{missing_columns}"
        )

    if X_test.isna().any().any():
        missing_columns = X_test.columns[
            X_test.isna().any()
        ].tolist()

        raise ValueError(
            "The test features contain missing values in: "
            f"{missing_columns}"
        )

    return X_train, y_train, X_test, y_test


def print_run_information(
    model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """Print the main configuration of the explainability run."""

    print("\n" + "=" * 70)
    print("EXPLAINABILITY MODULE")
    print("=" * 70)

    print(f"Experiment: {RUN_NAME}")
    print(f"Model type: {type(model).__name__}")
    print(f"Training observations: {len(X_train)}")
    print(f"Test observations: {len(X_test)}")
    print(f"Number of features: {X_test.shape[1]}")
    print(f"Target column: {TARGET_COLUMN}")
    print(f"SHAP sample position: {SAMPLE_POSITION}")

    print(
        "Test target values: "
        f"{sorted(pd.unique(y_test).tolist())}"
    )

    print("\nFeatures used by the final model:")

    for position, feature in enumerate(
        X_test.columns,
        start=1,
    ):
        print(f"{position:02d}. {feature}")


def main() -> None:
    """
    Execute global and local explainability analyses.

    Workflow
    --------
    1. Load the final trained model.
    2. Load and align the training and test datasets.
    3. Compute global permutation importance.
    4. Save the permutation importance CSV and plots.
    5. Compute a local SHAP force explanation.
    6. Save the SHAP tables and force plot.
    """

    create_output_directory()

    model = load_model()

    (
        X_train,
        _,
        X_test,
        y_test,
    ) = prepare_explainability_data(
        model=model
    )

    print_run_information(
        model=model,
        X_train=X_train,
        X_test=X_test,
        y_test=y_test,
    )

    # ========================================================
    # GLOBAL EXPLANATION: PERMUTATION IMPORTANCE
    # ========================================================

    permutation_df = run_permutation_importance(
        model=model,
        X_test=X_test,
        y_test=y_test,
    )

    run_permutation_importance_plot(
        importance_df=permutation_df,
        show=False,
    )

    # ========================================================
    # LOCAL EXPLANATION: SHAP FORCE PLOT
    # ========================================================

    run_shap_force(
        model=model,
        X_background=X_train,
        X_test=X_test,
        sample_position=SAMPLE_POSITION,
    )

    print("\n" + "=" * 70)
    print("EXPLAINABILITY ANALYSIS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()