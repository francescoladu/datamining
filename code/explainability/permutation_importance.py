from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import (
    permutation_importance as sklearn_permutation_importance,
)

from config import (
    PERMUTATION_N_JOBS,
    PERMUTATION_N_REPEATS,
    PERMUTATION_RESULTS_PATH,
    PERMUTATION_SCORING,
    RANDOM_STATE,
    create_output_directory,
)


def _validate_inputs(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series | np.ndarray,
) -> None:
    """Validate the inputs used for permutation importance."""

    if not hasattr(model, "predict"):
        raise TypeError(
            "The model must provide a predict() method."
        )

    if not isinstance(X_test, pd.DataFrame):
        raise TypeError(
            "X_test must be a pandas DataFrame so that feature "
            "names can be preserved."
        )

    if X_test.empty:
        raise ValueError("X_test cannot be empty.")

    if X_test.columns.has_duplicates:
        duplicated_columns = X_test.columns[
            X_test.columns.duplicated()
        ].tolist()

        raise ValueError(
            "X_test contains duplicated feature names: "
            f"{duplicated_columns}"
        )

    if len(X_test) != len(y_test):
        raise ValueError(
            "X_test and y_test must contain the same number "
            "of observations."
        )


def compute_permutation_importance(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series | np.ndarray,
    *,
    scoring: str = PERMUTATION_SCORING,
    n_repeats: int = PERMUTATION_N_REPEATS,
    random_state: int = RANDOM_STATE,
    n_jobs: int = PERMUTATION_N_JOBS,
) -> pd.DataFrame:
    """
    Compute global permutation importance on unseen test data.

    For each feature, the function randomly shuffles its values
    while keeping all the other features unchanged. It then
    measures the decrease in the selected performance metric.

    A large positive importance means that the model strongly
    relies on the feature.

    A value close to zero means that shuffling the feature has
    little effect on model performance.

    A negative value means that the model performed slightly
    better after the feature was shuffled. This can occur because
    of random variation, redundant features, or noisy features.

    Parameters
    ----------
    model:
        Trained estimator or trained scikit-learn pipeline.

    X_test:
        Test features that were not used to train the model.

    y_test:
        True target values associated with X_test.

    scoring:
        Scikit-learn scoring metric used to measure the
        performance decrease.

    n_repeats:
        Number of independent shuffles performed for each
        feature.

    random_state:
        Random seed used to make the experiment reproducible.

    n_jobs:
        Number of CPU cores used by scikit-learn.
        A value of -1 uses all available cores.

    Returns
    -------
    pandas.DataFrame
        Table containing the mean permutation importance,
        standard deviation, minimum, maximum, and feature rank.
    """

    _validate_inputs(
        model=model,
        X_test=X_test,
        y_test=y_test,
    )

    if n_repeats <= 0:
        raise ValueError("n_repeats must be greater than zero.")

    result = sklearn_permutation_importance(
        estimator=model,
        X=X_test,
        y=y_test,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    # result.importances has shape:
    # number of features × number of repetitions
    importances = result.importances

    importance_df = pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
        "importance_min": importances.min(axis=1),
        "importance_max": importances.max(axis=1),
    })

    # Rank 1 corresponds to the feature with the largest
    # average decrease in model performance.
    importance_df["rank"] = (
        importance_df["importance_mean"]
        .rank(
            method="dense",
            ascending=False,
        )
        .astype(int)
    )

    importance_df["scoring"] = scoring
    importance_df["n_repeats"] = n_repeats

    importance_df = (
        importance_df
        .sort_values(
            by=[
                "importance_mean",
                "importance_std",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )

    return importance_df


def save_permutation_importance(
    importance_df: pd.DataFrame,
    output_path=PERMUTATION_RESULTS_PATH,
) -> None:
    """
    Save the permutation importance summary to a CSV file.

    Parameters
    ----------
    importance_df:
        DataFrame returned by compute_permutation_importance().

    output_path:
        Destination path of the CSV file.
    """

    if importance_df.empty:
        raise ValueError(
            "The permutation importance DataFrame cannot be empty."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    importance_df.to_csv(
        output_path,
        index=False,
    )


def print_permutation_importance(
    importance_df: pd.DataFrame,
) -> None:
    """Print a readable permutation importance summary."""

    columns_to_print = [
        "rank",
        "feature",
        "importance_mean",
        "importance_std",
    ]

    print("\nGLOBAL EXPLAINABILITY — PERMUTATION IMPORTANCE")
    print(
        importance_df[columns_to_print].to_string(
            index=False,
        )
    )


def run_permutation_importance(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series | np.ndarray,
) -> pd.DataFrame:
    """
    Execute the complete permutation importance workflow.

    The function:
    1. creates the output directory;
    2. computes permutation importance;
    3. saves the results to CSV;
    4. prints a summary;
    5. returns the resulting DataFrame.
    """

    create_output_directory()

    importance_df = compute_permutation_importance(
        model=model,
        X_test=X_test,
        y_test=y_test,
    )

    save_permutation_importance(
        importance_df=importance_df,
    )

    print_permutation_importance(
        importance_df=importance_df,
    )

    print(
        "\nPermutation importance results saved to:"
        f"\n{PERMUTATION_RESULTS_PATH}"
    )

    return importance_df