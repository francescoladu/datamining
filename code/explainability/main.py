import ast

from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    SelectKBest,
    mutual_info_classif,
)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from config import (
    CLASS_TO_EXPLAIN,
    EXPECTED_LABELS,
    FINAL_BEST_PARAMETERS_PATH,
    RANDOM_STATE,
    RUN_NAME,
    SAMPLE_POSITION,
    TARGET_COLUMN,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
    create_output_directory,
)
from permutation_importance import (
    run_permutation_importance,
)
from plots import (
    run_permutation_importance_plot,
)
from shap_force import (
    run_shap_force,
)


def load_dataset(
    path: Path,
    name: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load one cleaned dataset and validate its target."""

    if not path.is_file():
        raise FileNotFoundError(
            f"{name} not found:\n{path}"
        )

    dataset = pd.read_csv(path)

    if dataset.empty:
        raise ValueError(
            f"{name} is empty."
        )

    if TARGET_COLUMN not in dataset.columns:
        raise ValueError(
            f"Target column {TARGET_COLUMN!r} "
            f"missing from {name}."
        )

    X = dataset.drop(
        columns=[TARGET_COLUMN]
    )

    y = dataset[
        TARGET_COLUMN
    ].copy()

    if set(y.unique()) != EXPECTED_LABELS:
        raise ValueError(
            f"Unexpected labels in {name}: "
            f"{sorted(y.unique())}"
        )

    if X.empty:
        raise ValueError(
            f"{name} does not contain features."
        )

    if X.columns.has_duplicates:
        duplicated_columns = X.columns[
            X.columns.duplicated()
        ].tolist()

        raise ValueError(
            f"{name} contains duplicated columns: "
            f"{duplicated_columns}"
        )

    if X.isna().any().any():
        missing_columns = X.columns[
            X.isna().any()
        ].tolist()

        raise ValueError(
            f"{name} contains missing values in: "
            f"{missing_columns}"
        )

    return X, y


def load_data() -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
]:
    """Load training and test data with identical feature order."""

    X_train, y_train = load_dataset(
        TRAIN_DATA_PATH,
        "training dataset",
    )

    X_test, y_test = load_dataset(
        TEST_DATA_PATH,
        "test dataset",
    )

    if set(X_train.columns) != set(X_test.columns):
        missing_from_test = [
            feature
            for feature in X_train.columns
            if feature not in X_test.columns
        ]

        extra_in_test = [
            feature
            for feature in X_test.columns
            if feature not in X_train.columns
        ]

        raise ValueError(
            "Training and test datasets have different features. "
            f"Missing from test: {missing_from_test}. "
            f"Extra in test: {extra_in_test}."
        )

    # Preserve the exact training feature order.
    X_test = X_test.loc[
        :,
        X_train.columns,
    ].copy()

    return (
        X_train,
        y_train,
        X_test,
        y_test,
    )


def parse_parameter(
    value: str,
) -> Any:
    """Convert a CSV value to a Python parameter value."""

    value = value.strip()

    if value == "":
        return None

    try:
        return ast.literal_eval(value)

    except (ValueError, SyntaxError):
        # Values such as "sqrt", "entropy", and "all"
        # are valid strings but not Python literals.
        return value


def build_pipeline(
    model_name: str,
) -> Pipeline:
    """
    Create the same pipeline used during model selection.

    All project features are discrete, so Mutual Information
    receives discrete_features=True.
    """

    mutual_information = partial(
        mutual_info_classif,
        discrete_features=True,
        random_state=RANDOM_STATE,
    )

    if model_name == "Decision Tree":
        classifier = DecisionTreeClassifier(
            random_state=RANDOM_STATE,
        )

    elif model_name == "Random Forest":
        classifier = RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=1,
        )

    else:
        raise ValueError(
            f"Unsupported model family: {model_name!r}. "
            "Expected 'Decision Tree' or 'Random Forest'."
        )

    return Pipeline(
        steps=[
            (
                "feature_selection",
                SelectKBest(
                    score_func=mutual_information,
                ),
            ),
            (
                "classifier",
                classifier,
            ),
        ]
    )


def rebuild_final_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[
    Pipeline,
    str,
    dict[str, Any],
]:
    """
    Rebuild and fit the final model from the saved CSV.

    No joblib file is needed. The pipeline is reconstructed
    using the model family and parameters stored in
    final_best_parameters.csv.
    """

    if not FINAL_BEST_PARAMETERS_PATH.is_file():
        raise FileNotFoundError(
            "Final best-parameters CSV not found:\n"
            f"{FINAL_BEST_PARAMETERS_PATH}"
        )

    parameters_df = pd.read_csv(
        FINAL_BEST_PARAMETERS_PATH,
        dtype=str,
        keep_default_na=False,
    )

    if len(parameters_df) != 1:
        raise ValueError(
            "final_best_parameters.csv "
            "must contain exactly one row."
        )

    if "model" not in parameters_df.columns:
        raise ValueError(
            "final_best_parameters.csv "
            "does not contain the 'model' column."
        )

    row = parameters_df.iloc[0]

    model_name = row[
        "model"
    ].strip()

    model = build_pipeline(
        model_name
    )

    parameters: dict[str, Any] = {}

    for column, raw_value in row.items():
        if not column.startswith(
            (
                "feature_selection__",
                "classifier__",
            )
        ):
            continue

        value = parse_parameter(
            raw_value
        )

        if (
            column == "feature_selection__k"
            and value != "all"
        ):
            value = int(value)

        parameters[column] = value

    if not parameters:
        raise ValueError(
            "No pipeline parameters were found in "
            "final_best_parameters.csv."
        )

    model.set_params(
        **parameters
    )

    # Refit the final pipeline on the complete training set.
    model.fit(
        X_train,
        y_train,
    )

    return (
        model,
        model_name,
        parameters,
    )


def main() -> None:
    """Run global and local explainability."""

    create_output_directory()

    (
        X_train,
        y_train,
        X_test,
        _,
    ) = load_data()

    (
        model,
        model_name,
        parameters,
    ) = rebuild_final_model(
        X_train,
        y_train,
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "EXPLAINABILITY MODULE"
    )

    print(
        "=" * 70
    )

    print(
        f"Experiment: {RUN_NAME}"
    )

    print(
        f"Rebuilt model: {model_name}"
    )

    print(
        f"Training observations: {len(X_train)}"
    )

    print(
        f"Test observations: {len(X_test)}"
    )

    print(
        f"Input features: {X_train.shape[1]}"
    )

    print(
        f"SHAP sample position: {SAMPLE_POSITION}"
    )

    print(
        "\nReconstructed parameters:"
    )

    for name, value in sorted(
        parameters.items()
    ):
        print(
            f"- {name}: {value}"
        )

    # Global explanation loaded from the untouched
    # outer validation folds of nested cross-validation.
    importance_df = run_permutation_importance(
        model_name
    )

    run_permutation_importance_plot(
        importance_df
    )

    # Local explanation of the phishing probability.
    run_shap_force(
        model=model,
        X_background=X_train,
        X_test=X_test,
        sample_position=SAMPLE_POSITION,
        class_to_explain=CLASS_TO_EXPLAIN,
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "EXPLAINABILITY ANALYSIS COMPLETED"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()