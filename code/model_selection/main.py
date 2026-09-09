from __future__ import annotations

import argparse
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
    make_weighted_accuracy_scorer,
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
from model_selection.utils import select_by_one_se_rule
from shared.modeling import load_clean_dataset
from shared.config import SELECTED_RUN_NAME

def save_csv(
    dataframe: pd.DataFrame,
    path: Path,
    description: str,
    *,
    index: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=index)
    print(f"-> Saved {description}: {path.name}")


def save_json(payload: dict[str, Any], path: Path, description: str) -> None:
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
    frames = [result[table_name] for result in model_results]
    non_empty_frames = [frame for frame in frames if not frame.empty]
    if not non_empty_frames:
        return pd.DataFrame()
    return pd.concat(non_empty_frames, ignore_index=True)


def compact_search_results(
    search: GridSearchCV | RandomizedSearchCV,
) -> pd.DataFrame:
    results = pd.DataFrame(search.cv_results_).copy()
    results.insert(0, "candidate_id", np.arange(1, len(results) + 1))
    results["selected_by_one_se"] = False
    results.loc[search.best_index_, "selected_by_one_se"] = True

    if "params" in results.columns:
        results["params"] = results["params"].map(
            lambda val: json.dumps(val, sort_keys=True, default=str)
        )

    preferred_columns = [
        "candidate_id",
        "selected_by_one_se",
        "rank_test_score",
        "mean_test_score",
        "std_test_score",
        "mean_fit_time",
        "std_fit_time",
        "mean_score_time",
        "std_score_time",
        "params",
    ]
    param_cols = sorted(col for col in results.columns if col.startswith("param_"))
    split_cols = sorted(
        col for col in results.columns if col.startswith("split") and col.endswith("_test_score")
    )
    return results[
        [c for c in preferred_columns + param_cols + split_cols if c in results.columns]
    ]


def final_selected_feature_table(
    final_model: Any,
    feature_names: list[str],
) -> pd.DataFrame:
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
    ).sort_values(["selected", "mutual_information_rank"], ascending=[False, True])


def build_nested_summary(nested_scores: pd.DataFrame) -> pd.DataFrame:
    """Aggregate outer-fold metrics and rank model families by accuracy."""
    return (
        nested_scores.groupby("model")
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            phishing_precision_mean=("phishing_precision", "mean"),
            phishing_precision_std=("phishing_precision", "std"),
            phishing_recall_mean=("phishing_recall", "mean"),
            phishing_recall_std=("phishing_recall", "std"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            mean_selected_features=("selected_feature_count", "mean"),
            std_selected_features=("selected_feature_count", "std"),
        )
        .sort_values(by="accuracy_mean", ascending=False)
    )


def create_final_search(
    best_model_family: str,
    X_dev: pd.DataFrame,
    w_dev: pd.Series,
) -> GridSearchCV | RandomizedSearchCV:
    """
    Create the final hyperparameter search using the same
    weighted-accuracy criterion used during nested CV.
    """
    accuracy_scorer = make_weighted_accuracy_scorer(
        X_dev,
        w_dev,
    )

    if best_model_family == "Decision Tree":
        return GridSearchCV(
            estimator=decision_tree_pipeline,
            param_grid=config.decision_tree_param_grid,
            scoring=accuracy_scorer,
            cv=final_inner_cv,
            refit=select_by_one_se_rule,
            n_jobs=-1,
            return_train_score=False,
            error_score="raise",
        )

    if best_model_family == "Random Forest":
        return RandomizedSearchCV(
            estimator=random_forest_pipeline,
            param_distributions=config.random_forest_param_distributions,
            n_iter=config.N_RANDOM_ITERATIONS,
            scoring=accuracy_scorer,
            cv=final_inner_cv,
            refit=select_by_one_se_rule,
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
            return_train_score=False,
            error_score="raise",
        )

    raise RuntimeError(
        f"Unknown model family: {best_model_family}"
    )


def run_experiment(
    *,
    experiment_id: str,
    experiment_name: str,
    dataset_variant: str,
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    w_dev: pd.Series,
    output_base_dir: Path,
    weighted: bool,
) -> dict[str, Any]:
    """Run the existing nested-CV/model-selection workflow on one dataset variant."""
    run_tag = SELECTED_RUN_NAME
    paths = create_run_output_paths(output_base_dir, run_tag)

    scoring_label = ("weighted_accuracy" if weighted else "accuracy")
    print("=" * 80)
    print(f"{experiment_id}: {experiment_name}")
    print(f"Dataset variant: {dataset_variant}")
    print(f"Rows used by CV: {len(X_dev)}")
    print(f"Features: {X_dev.shape[1]}")
    print(f"Evaluation metric: {scoring_label}")
    print(f"Sample weighting: {'enabled' if weighted else 'unit weights (unweighted)'}")
    print(f"k candidates: {config.FEATURE_SELECTION_K_VALUES}")
    print("=" * 80)

    run_configuration = {
        "experiment_id": experiment_id,
        "experiment_name": experiment_name,
        "dataset_variant": dataset_variant,
        "experiment_tag": run_tag,
        "feature_selection_k_values": config.FEATURE_SELECTION_K_VALUES,
        "random_state": config.RANDOM_STATE,
        "primary_scoring": scoring_label,
        "random_forest_random_iterations": config.N_RANDOM_ITERATIONS,
        "outer_folds": outer_cv.n_splits,
        "final_inner_folds": final_inner_cv.n_splits,
        "compute_permutation_importance": config.COMPUTE_PERMUTATION_IMPORTANCE,
        "permutation_repeats": config.PERMUTATION_N_REPEATS,
        "high_confidence_threshold": config.HIGH_CONFIDENCE_THRESHOLD,
        "development_rows": len(X_dev),
        "development_weighted_instances": float(w_dev.sum()),
        "input_features": X_dev.shape[1],
        "sample_weighting_enabled": weighted,
        "selection_rule": "one_standard_error",
    }
    save_json(run_configuration, paths.root / "run_config.json", "run configuration")

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
        sample_weight=w_dev,
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
        sample_weight=w_dev,
        outer_splits=outer_splits,
        n_random_iterations=config.N_RANDOM_ITERATIONS,
    )

    model_results = [decision_tree_results, random_forest_results]
    nested_scores = combine_result_tables(model_results, "fold_scores")
    oof_predictions = combine_result_tables(model_results, "oof_predictions")
    permutation_scores = combine_result_tables(model_results, "permutation_importance")

    nested_summary = build_nested_summary(nested_scores)
    permutation_summary = summarize_permutation_importance(permutation_scores)
    error_summary = build_error_summary(oof_predictions)
    error_by_feature_value = build_error_by_feature_value(oof_predictions, X_dev)
    statistical_tests = compute_statistical_tests(nested_scores)

    print("-" * 80)
    summary_label = "Weighted" if weighted else "Unweighted"
    print(f"Nested Cross-Validation Accuracy Summary ({summary_label})")
    print("-" * 80)
    print(nested_summary.round(4))
    print()

    save_csv(nested_scores, paths.model_comparison / "fold_scores.csv", "outer-fold scores")
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
    print(f"-> Generated accuracy comparison chart: {model_comparison_pdf.name}\n")

    best_model_family = str(nested_summary.index[0])
    print(f"Selected Model Family by accuracy: {best_model_family}")
    print(f"Fitting final {best_model_family} search on the experiment dataset...")

    final_search = create_final_search(
        best_model_family,
        X_dev,
        w_dev,
)
    final_search.fit(
        X_dev,
        y_dev,
        classifier__sample_weight=np.asarray(w_dev),
    )
    final_model = final_search.best_estimator_

    selected_index = final_search.best_index_
    selected_development_cv_score = float(
        final_search.cv_results_["mean_test_score"][selected_index]
    )
    max_development_cv_score = float(
        np.max(final_search.cv_results_["mean_test_score"])
    )

    print(
        f"Selected Development CV Accuracy: "
        f"{selected_development_cv_score:.4f}"
    )
    print(f"Final Selected Hyperparameters: {final_search.best_params_}\n")

    final_search_table = compact_search_results(final_search)
    final_best_parameters = pd.DataFrame(
        [
            {
                "model": best_model_family,
                "development_cv_score": selected_development_cv_score,
                "development_cv_metric": scoring_label,
                "max_development_cv_score": max_development_cv_score,
                **final_search.best_params_,
            }
        ]
    )

    final_features = final_selected_feature_table(final_model, list(X_dev.columns))

    save_csv(
        final_search_table,
        paths.hyperparameter_search / "final_search_results.csv",
        "final search candidates",
    )
    save_csv(
        final_best_parameters,
        paths.hyperparameter_search / "final_best_parameters.csv",
        "final selected parameters",
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
    )
    print(f"-> Generated hyperparameter accuracy chart: {hyperparameter_pdf.name}\n")

    wilcoxon_p = (
        float(statistical_tests.iloc[0]["p_value"])
        if not statistical_tests.empty
        else None
    )
    results_summary = {
        "experiment_id": experiment_id,
        "experiment_name": experiment_name,
        "dataset_variant": dataset_variant,
        "selected_model": best_model_family,
        "selected_k": final_search.best_params_.get("feature_selection__k"),
        "development_cv_accuracy": selected_development_cv_score,
        "max_development_cv_accuracy": max_development_cv_score,
        "nested_cv_accuracy_mean": float(
            nested_summary.loc[best_model_family, "accuracy_mean"]
        ),
        "nested_cv_accuracy_std": float(
            nested_summary.loc[best_model_family, "accuracy_std"]
        ),
        "wilcoxon_p_value": wilcoxon_p,
        "rows": len(X_dev),
        "weighted_instances": float(w_dev.sum()),
        "sample_weighting_enabled": weighted,
    }
    save_json(results_summary, paths.root / "results_summary.json", "run result summary")

    print("=" * 80)
    print(f"{experiment_id} COMPLETED SUCCESSFULLY.")
    print("=" * 80)
    print()

    return results_summary


def parse_arguments() -> argparse.Namespace:
    """
    Select exactly one experiment to run.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run model selection for one Data Mining experiment."
        )
    )

    parser.add_argument(
        "experiment",
        choices=["1", "2", "3"],
        help=(
            "Experiment to run: "
            "1 = raw data, "
            "2 = exact deduplication, "
            "3 = weighted deduplication."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    print("=" * 80)
    print(
        f"STARTING MODEL SELECTION - EXPERIMENT {args.experiment}"
    )
    print("=" * 80)

    outputs_dir = module_dir / "outputs"

    # =========================================================
    # EXPERIMENT 1
    # =========================================================
    if args.experiment == "1":
        try:
            X_dev, y_dev, w_dev = load_clean_dataset(
                config.RAW_DEVELOPMENT_PATH,
                "Experiment 1 raw development dataset",
            )

        except (FileNotFoundError, ValueError) as error:
            print(
                f"Error loading Experiment 1 dataset: {error}"
            )
            print(
                "Run first:\n"
                "python scripts/clean_and_split.py 1"
            )
            sys.exit(1)

        result = run_experiment(
            experiment_id="EXPERIMENT 1",
            experiment_name="Baseline - Raw Data",
            dataset_variant=(
                "Original dataset without deduplication"
            ),
            X_dev=X_dev,
            y_dev=y_dev,
            w_dev=w_dev,
            output_base_dir=(
                outputs_dir
                / "experiment_1_raw"
            ),
            weighted=False,
        )

    # =========================================================
    # EXPERIMENT 2
    # =========================================================
    elif args.experiment == "2":
        try:
            X_dev, y_dev, w_dev = load_clean_dataset(
                config.STANDARD_DEDUP_DEVELOPMENT_PATH,
                "Experiment 2 exact-deduplicated development dataset",
            )

        except (FileNotFoundError, ValueError) as error:
            print(
                f"Error loading Experiment 2 dataset: {error}"
            )
            print(
                "Run first:\n"
                "python scripts/clean_and_split.py 2"
            )
            sys.exit(1)

        result = run_experiment(
            experiment_id="EXPERIMENT 2",
            experiment_name="Standard Deduplication",
            dataset_variant=(
                 "Dataset obtained by retaining one observation "
                 "for each identical feature-vector and target-label combination"
            ),
            X_dev=X_dev,
            y_dev=y_dev,
            w_dev=w_dev,
            output_base_dir=(
                outputs_dir
                / "experiment_2_standard_dedup"
            ),
            weighted=False,
        )

    # =========================================================
    # EXPERIMENT 3
    # =========================================================
    elif args.experiment == "3":
        try:
            X_dev, y_dev, w_dev = load_clean_dataset(
                config.WEIGHTED_DEVELOPMENT_PATH,
                "Experiment 3 weighted development dataset",
            )

        except (FileNotFoundError, ValueError) as error:
            print(
                f"Error loading Experiment 3 dataset: {error}"
            )
            print(
                "Run first:\n"
                "python scripts/clean_and_split.py 3"
            )
            sys.exit(1)

        result = run_experiment(
            experiment_id="EXPERIMENT 3",
            experiment_name="Weighted Deduplication",
            dataset_variant=(
                "Common development profile partition "
                "after majority-vote deduplication with "
                "retained-support sample weights"
            ),
            X_dev=X_dev,
            y_dev=y_dev,
            w_dev=w_dev,
            output_base_dir=outputs_dir,
            weighted=True,
        )

    else:
        raise RuntimeError(
            "Unexpected experiment identifier."
        )

    print("=" * 80)
    print(
        f"EXPERIMENT {args.experiment} "
        "MODEL SELECTION COMPLETED."
    )
    print("=" * 80)

    print(
        f"Selected model: "
        f"{result['selected_model']}"
    )

    print(
        f"Nested-CV accuracy: "
        f"{result['nested_cv_accuracy_mean']:.4f} "
        f"+/- {result['nested_cv_accuracy_std']:.4f}"
    )

    print(
        f"Development CV accuracy: "
        f"{result['development_cv_accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()

