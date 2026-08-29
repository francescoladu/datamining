from __future__ import annotations

import ast
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from shared.config import (
    EXPECTED_LABELS,
    RANDOM_STATE,
    TARGET_COLUMN,
)


def load_clean_dataset(
    path: Path,
    name: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load one cleaned dataset and validate the project invariants."""
    if not path.is_file():
        raise FileNotFoundError(f"{name} not found:\n{path}")

    dataset = pd.read_csv(path)
    if dataset.empty:
        raise ValueError(f"{name} is empty.")
    if TARGET_COLUMN not in dataset.columns:
        raise ValueError(
            f"Target column {TARGET_COLUMN!r} missing from {name}."
        )

    X = dataset.drop(columns=[TARGET_COLUMN])
    y = dataset[TARGET_COLUMN].copy()

    if set(y.unique()) != EXPECTED_LABELS:
        raise ValueError(
            f"Unexpected labels in {name}: {sorted(y.unique())}"
        )
    if X.empty:
        raise ValueError(f"{name} does not contain features.")
    if X.columns.has_duplicates:
        duplicated_columns = X.columns[X.columns.duplicated()].tolist()
        raise ValueError(
            f"{name} contains duplicated columns: {duplicated_columns}"
        )
    if X.isna().any().any():
        missing_columns = X.columns[X.isna().any()].tolist()
        raise ValueError(
            f"{name} contains missing values in: {missing_columns}"
        )

    return X, y


def load_development_and_test(
    development_path: Path,
    test_path: Path,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load development and test data with identical feature order."""
    X_dev, y_dev = load_clean_dataset(
        development_path,
        "development dataset",
    )
    X_test, y_test = load_clean_dataset(test_path, "test dataset")

    if set(X_dev.columns) != set(X_test.columns):
        missing_from_test = [
            feature for feature in X_dev.columns if feature not in X_test.columns
        ]
        extra_in_test = [
            feature for feature in X_test.columns if feature not in X_dev.columns
        ]
        raise ValueError(
            "Development and test datasets have different features. "
            f"Missing from test: {missing_from_test}. "
            f"Extra in test: {extra_in_test}."
        )

    X_test = X_test.loc[:, X_dev.columns].copy()
    return X_dev, y_dev, X_test, y_test


def parse_parameter(value: str) -> Any:
    """Convert a CSV cell back to the parameter value expected by sklearn."""
    value = value.strip()
    if value == "":
        return None

    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def build_pipeline(model_name: str) -> Pipeline:
    """Create the exact pipeline definition shared by all project modules."""
    mutual_information = partial(
        mutual_info_classif,
        discrete_features=True,
        random_state=RANDOM_STATE,
    )

    if model_name == "Decision Tree":
        classifier = DecisionTreeClassifier(random_state=RANDOM_STATE)
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
                SelectKBest(score_func=mutual_information),
            ),
            ("classifier", classifier),
        ]
    )


def load_final_model_configuration(
    parameters_path: Path,
) -> tuple[str, float, dict[str, Any]]:
    """Read the frozen model family, development score, and pipeline params."""
    if not parameters_path.is_file():
        raise FileNotFoundError(
            "Final best-parameters CSV not found:\n"
            f"{parameters_path}"
        )

    parameters_df = pd.read_csv(
        parameters_path,
        dtype=str,
        keep_default_na=False,
    )
    if len(parameters_df) != 1:
        raise ValueError(
            "final_best_parameters.csv must contain exactly one row."
        )
    if "model" not in parameters_df.columns:
        raise ValueError(
            "final_best_parameters.csv does not contain the 'model' column."
        )
    if "development_cv_score" not in parameters_df.columns:
        raise ValueError(
            "final_best_parameters.csv does not contain "
            "the 'development_cv_score' column."
        )

    row = parameters_df.iloc[0]
    model_name = row["model"].strip()
    development_cv_score = float(row["development_cv_score"])

    parameters: dict[str, Any] = {}
    for column, raw_value in row.items():
        if not column.startswith(("feature_selection__", "classifier__")):
            continue

        value = parse_parameter(raw_value)
        if column == "feature_selection__k" and value != "all":
            value = int(value)
        parameters[column] = value

    if not parameters:
        raise ValueError(
            "No pipeline parameters were found in final_best_parameters.csv."
        )

    return model_name, development_cv_score, parameters


def rebuild_final_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    parameters_path: Path,
) -> tuple[Pipeline, str, float, dict[str, Any]]:
    """Rebuild the frozen pipeline and refit it on the full development set."""
    model_name, development_cv_score, parameters = (
        load_final_model_configuration(parameters_path)
    )
    model = build_pipeline(model_name)
    model.set_params(**parameters)
    model.fit(X_train, y_train)
    return model, model_name, development_cv_score, parameters
