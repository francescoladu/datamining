from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

module_dir = Path(__file__).resolve().parent
code_dir = module_dir.parent
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

from model_selection import config
from model_selection.diagnostics import (
    build_error_by_feature_value,
    build_error_summary,
)
from model_selection.engine import (
    decision_tree_pipeline,
    final_inner_cv,
    nested_cross_validation,
    outer_cv,
    random_forest_pipeline,
)
from model_selection.paths import create_run_output_paths
from model_selection.plots import (
    plot_hyperparameter_optimization,
    plot_nested_cv_comparison,
    plot_selected_feature_ranking,
)
from model_selection.summaries import (
    compute_statistical_tests,
    summarize_permutation_importance,
)
from shared.config import DATA_DIR
from shared.modeling import load_clean_dataset


def experiment_tag(k_values: list[Any]) -> str:
    """Build a filesystem-safe experiment name from the active k setting."""
    labels = [str(value).lower() for value in k_values]
    if len(labels) == 1:
        return f"k_{labels[0]}"
    return "k_search_" + "-".join(labels)


def save_csv(
    dataframe: pd.DataFrame,
    path: Path,
    description: str,
    *,
    index: bool = False,
) -> None:
    """Save a DataFrame and print a consistent audit message."""
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=index)
    print(f"-> Saved {description}: {path.relative_to(path.parents[2])}")


def save_json(payload: dict[str, Any], path: Path, description: str) -> None:
    """Save JSON metadata with readable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"-> Saved {description}: {path.name}")


def combine_result_tables(
    model_results: list[dict[str, pd.DataFrame]],
    table_name: str,
) -> pd.DataFrame:
    """Concatenate the same result table returned by multiple model runs."""
    frames = [result[table_name] for result in model_results]
    non_empty_frames = [frame for frame in frames if not frame.empty]
    if not non_empty_frames:
        return pd.DataFrame()
    return pd.concat(non_empty_frames, ignore_index=True)


def compact_search_results(
    search: GridSearchCV | RandomizedSearchCV,
) -> pd.DataFrame:
    """Convert final SearchCV results into a CSV-friendly table."""
    results = pd.DataFrame(search.cv_results_).copy()
    results.insert(0, "candidate_id", np.arange(1, len(results) + 1))

    if "params" in results.columns:
        results["params"] = results["params"].map(
            lambda value: json.dumps(value, sort_keys=True, default=str)
        )

    preferred_columns = [
        "candidate_id",
        "rank_test_score",
        "mean_test_score",
        "std_test_score",
        "mean_fit_time",
        "std_fit_time",
        "mean_score_time",
        "std_score_time",
        "params",
    ]
    parameter_columns = sorted(
        column for column in results.columns if column.startswith("param_")
    )
    split_columns = sorted(
        column
        for column in results.columns
        if column.startswith("split") and column.endswith("_test_score")
    )

    return results[
        [
            column
            for column in preferred_columns + parameter_columns + split_columns
            if column in results.columns
        ]
    ]


def final_selected_feature_table(
    final_model: Any,
    feature_names: list[str],
) -> pd.DataFrame:
    """Export MI scores and selection status from the final fitted pipeline."""
    selector = final_model.named_steps["feature_selection"]
    selected_mask = np.asarray(selector.get_support(), dtype=bool)
    scores = np.asarray(selector.scores_, dtype=float)
    ranks = (
        pd.Series(scores)
        .rank(method="min", ascending=False, na_option="bottom")
        .astype(int)
        .to_numpy()
    )

    return pd.DataFrame(
        {
            "feature": feature_names,
            "mutual_information_score": scores,
            "mutual_information_rank": ranks,
            "selected": selected_mask,
        }
    ).sort_values(
        ["selected", "mutual_information_rank"],
        ascending=[False, True],
    )


def build_nested_summary(nested_scores: pd.DataFrame) -> pd.DataFrame:
    """Aggregate outer-fold metrics for model-family comparison."""
    return (
        nested_scores
        .groupby("model")
        .agg(
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            phishing_precision_mean=("phishing_precision", "mean"),
            phishing_precision_std=("phishing_precision", "std"),
            phishing_recall_mean=("phishing_recall", "mean"),
            phishing_recall_std=("phishing_recall", "std"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            mean_selected_features=("selected_feature_count", "mean"),
            std_selected_features=("selected_feature_count", "std"),
        )
        .sort_values(by="macro_f1_mean", ascending=False)
    )


def create_final_search(
    best_model_family: str,
) -> GridSearchCV | RandomizedSearchCV:
    """Create the development-set search used after model-family selection."""
    if best_model_family == "Decision Tree":
        return GridSearchCV(
            estimator=decision_tree_pipeline,
            param_grid=config.decision_tree_param_grid,
            scoring=config.PRIMARY_SCORING,
            cv=final_inner_cv,
            refit=True,
            n_jobs=-1,
            return_train_score=False,
            error_score="raise",
        )
    if best_model_family == "Random Forest":
        return RandomizedSearchCV(
            estimator=random_forest_pipeline,
            param_distributions=config.random_forest_param_distributions,
            n_iter=config.N_RANDOM_ITERATIONS,
            scoring=config.PRIMARY_SCORING,
            cv=final_inner_cv,
            refit=True,
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
            return_train_score=False,
            error_score="raise",
        )
    raise RuntimeError(f"Unknown model family: {best_model_family}")


def main() -> None:
    run_tag = experiment_tag(config.FEATURE_SELECTION_K_VALUES)
    paths = create_run_output_paths(module_dir / "outputs", run_tag)
    train_path = DATA_DIR / "train_cleaned.csv"

    print("=" * 80)
    print("STARTING MODEL SELECTION PIPELINE")
    print(f"Experiment: {run_tag}")
    print(f"k candidates: {config.FEATURE_SELECTION_K_VALUES}")
    print("Data scope: development set only")
    print("=" * 80)

    try:
        X_dev, y_dev = load_clean_dataset(
            train_path,
            "development dataset",
        )
    except (FileNotFoundError, ValueError) as error:
        print(f"Error loading development dataset: {error}")
        print("Please run your preprocessing pipeline first.")
        sys.exit(1)

    print(
        f"Development Set: {X_dev.shape[0]} samples with "
        f"{X_dev.shape[1]} features.\n"
    )

    run_configuration = {
        "experiment_tag": run_tag,
        "feature_selection_k_values": config.FEATURE_SELECTION_K_VALUES,
        "random_state": config.RANDOM_STATE,
        "primary_scoring": config.PRIMARY_SCORING,
        "random_forest_random_iterations": config.N_RANDOM_ITERATIONS,
        "outer_folds": outer_cv.n_splits,
        "final_inner_folds": final_inner_cv.n_splits,
        "compute_permutation_importance": (
            config.COMPUTE_PERMUTATION_IMPORTANCE
        ),
        "permutation_repeats": config.PERMUTATION_N_REPEATS,
        "high_confidence_threshold": config.HIGH_CONFIDENCE_THRESHOLD,
        "development_observations": len(X_dev),
        "input_features": X_dev.shape[1],
        "data_scope": "development_only",
    }
    save_json(
        run_configuration,
        paths.root / "run_config.json",
        "run configuration",
    )

    outer_splits = list(outer_cv.split(X_dev, y_dev))

    print("-" * 80)
    print("Executing Nested CV: Decision Tree")
    print("-" * 80)
    decision_tree_results = nested_cross_validation(
        model_name="Decision Tree",
        pipeline=decision_tree_pipeline,
        search_space=config.decision_tree_param_grid,
        search_method="grid",
        X=X_dev,
        y=y_dev,
        outer_splits=outer_splits,
    )

    print("-" * 80)
    print("Executing Nested CV: Random Forest")
    print("-" * 80)
    random_forest_results = nested_cross_validation(
        model_name="Random Forest",
        pipeline=random_forest_pipeline,
        search_space=config.random_forest_param_distributions,
        search_method="random",
        X=X_dev,
        y=y_dev,
        outer_splits=outer_splits,
        n_random_iterations=config.N_RANDOM_ITERATIONS,
    )

    model_results = [decision_tree_results, random_forest_results]
    nested_scores = combine_result_tables(model_results, "fold_scores")
    oof_predictions = combine_result_tables(model_results, "oof_predictions")
    permutation_scores = combine_result_tables(
        model_results,
        "permutation_importance",
    )

    nested_summary = build_nested_summary(nested_scores)
    permutation_summary = summarize_permutation_importance(permutation_scores)
    error_summary = build_error_summary(oof_predictions)
    error_by_feature_value = build_error_by_feature_value(
        oof_predictions,
        X_dev,
    )
    statistical_tests = compute_statistical_tests(nested_scores)

    print("-" * 80)
    print("Nested Cross-Validation Performance Summary")
    print("-" * 80)
    print(nested_summary.round(4))
    print()

    save_csv(
        nested_scores,
        paths.model_comparison / "fold_scores.csv",
        "outer-fold scores",
    )
    save_csv(
        nested_summary.reset_index(),
        paths.model_comparison / "model_summary.csv",
        "aggregated performance summary",
    )
    save_csv(
        statistical_tests,
        paths.model_comparison / "statistical_tests.csv",
        "paired statistical tests",
    )

    save_csv(
        oof_predictions,
        paths.diagnostics / "oof_predictions.csv",
        "out-of-fold predictions",
    )
    save_csv(
        error_summary,
        paths.diagnostics / "error_summary.csv",
        "out-of-fold error summary",
    )
    save_csv(
        error_by_feature_value,
        paths.diagnostics / "error_by_feature_value.csv",
        "error rates by feature value",
    )
    if not permutation_scores.empty:
        save_csv(
            permutation_summary,
            paths.explainability / "permutation_importance_summary.csv",
            "aggregated permutation importance",
        )

    model_comparison_pdf = paths.figures / "model_comparison.pdf"
    plot_nested_cv_comparison(
        nested_scores=nested_scores,
        output_pdf_path=model_comparison_pdf,
    )
    print(f"-> Generated model comparison chart: {model_comparison_pdf.name}\n")

    best_model_family = str(nested_summary.index[0])
    print(f"Selected Model Family for Final Deployment: {best_model_family}")
    print("-" * 80)
    print(f"Fitting final {best_model_family} on full Development Set...")
    print("-" * 80)

    final_search = create_final_search(best_model_family)
    final_search.fit(X_dev, y_dev)
    final_model = final_search.best_estimator_

    print(
        f"Final Development CV Score ({config.PRIMARY_SCORING}): "
        f"{final_search.best_score_:.4f}"
    )
    print(f"Final Optimized Hyperparameters: {final_search.best_params_}\n")

    final_search_table = compact_search_results(final_search)
    final_best_parameters = pd.DataFrame(
        [
            {
                "model": best_model_family,
                "development_cv_score": float(final_search.best_score_),
                **final_search.best_params_,
            }
        ]
    )
    final_features = final_selected_feature_table(
        final_model,
        list(X_dev.columns),
    )

    save_csv(
        final_search_table,
        paths.hyperparameter_search / "final_search_results.csv",
        "final development search candidates",
    )
    save_csv(
        final_best_parameters,
        paths.hyperparameter_search / "final_best_parameters.csv",
        "final best parameters",
    )
    save_csv(
        final_features,
        paths.feature_selection / "final_selected_features.csv",
        "final feature-selection results",
    )

    feature_ranking_pdf = paths.figures / "feature_selection_ranking.pdf"
    plot_selected_feature_ranking(
        selected_features=final_features,
        output_pdf_path=feature_ranking_pdf,
        max_display=15,
    )
    print(f"-> Generated feature-selection ranking: {feature_ranking_pdf.name}")

    hyperparameter_pdf = paths.figures / "hyperparameter_optimization.pdf"
    plot_hyperparameter_optimization(
        search_results=final_search_table,
        output_pdf_path=hyperparameter_pdf,
        model_name=best_model_family,
        max_candidates=15,
    )
    print(f"-> Generated hyperparameter-search chart: {hyperparameter_pdf.name}\n")

    selected_k = final_search.best_params_.get("feature_selection__k")
    wilcoxon_p_value = None
    if not statistical_tests.empty:
        wilcoxon_p_value = float(statistical_tests.iloc[0]["p_value"])

    results_summary = {
        "selected_model": best_model_family,
        "selected_k": selected_k,
        "development_cv_score": float(final_search.best_score_),
        "nested_cv_macro_f1_mean": float(
            nested_summary.loc[best_model_family, "macro_f1_mean"]
        ),
        "nested_cv_macro_f1_std": float(
            nested_summary.loc[best_model_family, "macro_f1_std"]
        ),
        "wilcoxon_p_value": wilcoxon_p_value,
        "final_best_parameters_path": (
            "hyperparameter_search/final_best_parameters.csv"
        ),
    }
    save_json(
        results_summary,
        paths.root / "results_summary.json",
        "run result summary",
    )

    print("=" * 80)
    print("MODEL SELECTION PIPELINE RUN COMPLETED SUCCESSFULLY")
    print("The held-out test set was not loaded or evaluated.")
    print("Run final_evaluation/main.py only after freezing this configuration.")
    print(f"Outputs directory: {paths.root}")
    print("=" * 80)


if __name__ == "__main__":
    main()
