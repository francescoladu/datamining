import pandas as pd

from explainability.config import PERMUTATION_SUMMARY_SOURCE_PATH


def run_permutation_importance(model_name: str) -> pd.DataFrame:
    """
    Load and filter nested-CV global permutation importance.

    Values are not recomputed on the final test set. The authoritative CSV is
    the nested-CV summary written by model selection
    """
    if not PERMUTATION_SUMMARY_SOURCE_PATH.is_file():
        raise FileNotFoundError(
            "Permutation importance summary not found:\n"
            f"{PERMUTATION_SUMMARY_SOURCE_PATH}"
        )

    importance_df = pd.read_csv(PERMUTATION_SUMMARY_SOURCE_PATH)
    required_columns = {
        "model",
        "feature",
        "selected_in_folds",
        "mean_importance",
        "std_importance_across_folds",
        "mean_within_fold_std",
    }
    missing_columns = required_columns.difference(importance_df.columns)
    if missing_columns:
        raise ValueError(
            "Missing columns in permutation importance CSV: "
            f"{sorted(missing_columns)}"
        )

    importance_df = importance_df.loc[
        importance_df["model"].astype(str).str.strip() == model_name.strip()
    ].copy()
    if importance_df.empty:
        raise ValueError(f"No permutation importance found for {model_name!r}.")

    importance_df = importance_df.rename(
        columns={
            "mean_importance": "importance_mean",
            "std_importance_across_folds": "importance_std",
        }
    )
    importance_df["importance_std"] = importance_df["importance_std"].fillna(0.0)
    importance_df["rank"] = (
        importance_df["importance_mean"]
        .rank(method="dense", ascending=False)
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
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )

    print("\nGLOBAL EXPLAINABILITY - PERMUTATION IMPORTANCE")
    print(
        importance_df[
            [
                "rank",
                "feature",
                "selected_in_folds",
                "importance_mean",
                "importance_std",
            ]
        ].to_string(index=False)
    )
    print(
        "\nSource table (authoritative nested-CV output):"
        f"\n{PERMUTATION_SUMMARY_SOURCE_PATH}"
    )
    return importance_df
