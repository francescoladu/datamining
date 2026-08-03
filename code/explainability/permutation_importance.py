import pandas as pd

from config import (
    PERMUTATION_RESULTS_PATH,
    PERMUTATION_SUMMARY_SOURCE_PATH,
    create_output_directory,
)


def run_permutation_importance(
    model_name: str,
) -> pd.DataFrame:
    """
    Load, filter, standardize, and save global importance values.

    The values are not recomputed on the final test set.
    They are loaded from the nested-CV permutation importance
    summary produced during model selection.
    """

    if not PERMUTATION_SUMMARY_SOURCE_PATH.is_file():
        raise FileNotFoundError(
            "Permutation importance summary not found:\n"
            f"{PERMUTATION_SUMMARY_SOURCE_PATH}"
        )

    importance_df = pd.read_csv(
        PERMUTATION_SUMMARY_SOURCE_PATH
    )

    required_columns = {
        "model",
        "feature",
        "selected_in_folds",
        "mean_importance",
        "std_importance_across_folds",
        "mean_within_fold_std",
    }

    missing_columns = required_columns.difference(
        importance_df.columns
    )

    if missing_columns:
        raise ValueError(
            "Missing columns in permutation importance CSV: "
            f"{sorted(missing_columns)}"
        )

    # Keep only the model selected during the final search.
    importance_df = importance_df.loc[
        importance_df["model"]
        .astype(str)
        .str.strip()
        == model_name.strip()
    ].copy()

    if importance_df.empty:
        raise ValueError(
            f"No permutation importance found for {model_name!r}."
        )

    # Rename the columns so that the plotting module can use
    # a simple and consistent format.
    importance_df = importance_df.rename(
        columns={
            "mean_importance": "importance_mean",
            "std_importance_across_folds": "importance_std",
        }
    )

    importance_df["importance_std"] = (
        importance_df["importance_std"].fillna(0.0)
    )

    importance_df["rank"] = (
        importance_df["importance_mean"]
        .rank(
            method="dense",
            ascending=False,
        )
        .astype(int)
    )

    importance_df = (
        importance_df[
            [
                "rank",
                "model",
                "feature",
                "selected_in_folds",
                "importance_mean",
                "importance_std",
                "mean_within_fold_std",
            ]
        ]
        .sort_values(
            "importance_mean",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    create_output_directory()

    importance_df.to_csv(
        PERMUTATION_RESULTS_PATH,
        index=False,
    )

    print(
        "\nGLOBAL EXPLAINABILITY — "
        "PERMUTATION IMPORTANCE"
    )

    print(
        importance_df[
            [
                "rank",
                "feature",
                "selected_in_folds",
                "importance_mean",
                "importance_std",
            ]
        ].to_string(
            index=False,
        )
    )

    print(
        "\nFiltered importance table saved to:"
        f"\n{PERMUTATION_RESULTS_PATH}"
    )

    return importance_df