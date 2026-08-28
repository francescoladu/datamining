from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Ensure the module directory is in sys.path so local siblings resolve correctly.
module_dir = Path(__file__).resolve().parent
if str(module_dir) not in sys.path:
    sys.path.insert(0, str(module_dir))

import config
from engine import (
    nested_cross_validation,
    decision_tree_pipeline,
    random_forest_pipeline,
    outer_cv,
    final_inner_cv
)
from plots import (
    plot_final_test_confusion_matrix,
    plot_final_test_roc_curve,
    plot_hyperparameter_optimization,
    plot_nested_cv_comparison,
    plot_selected_feature_ranking,
)
from utils import (
    compute_classification_metrics,
    predict_with_phishing_probability,
)


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
    dataframe.to_csv(path, index=index)
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


def summarize_feature_frequency(
    selected_features: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize how often each feature is selected across outer folds."""
    return (
        selected_features
        .groupby(["model", "feature"], as_index=False)
        .agg(
            selected_in_folds=("selected", "sum"),
            selection_frequency=("selected", "mean"),
            mean_mutual_information=("mutual_information_score", "mean"),
            std_mutual_information=("mutual_information_score", "std"),
            mean_mutual_information_rank=("mutual_information_rank", "mean"),
        )
        .sort_values(
            ["model", "selected_in_folds", "mean_mutual_information"],
            ascending=[True, False, False],
        )
    )


def compute_feature_stability(
    selected_features: pd.DataFrame,
) -> pd.DataFrame:
    """Compute pairwise Jaccard similarity between selected feature subsets."""
    rows: list[dict[str, Any]] = []

    for model_name, model_frame in selected_features.groupby("model"):
        feature_sets = {
            int(outer_fold): set(
                fold_frame.loc[fold_frame["selected"], "feature"]
            )
            for outer_fold, fold_frame in model_frame.groupby("outer_fold")
        }

        for (fold_a, features_a), (fold_b, features_b) in combinations(
            sorted(feature_sets.items()),
            2,
        ):
            union = features_a | features_b
            intersection = features_a & features_b
            jaccard = len(intersection) / len(union) if union else 1.0

            rows.append(
                {
                    "model": model_name,
                    "outer_fold_a": fold_a,
                    "outer_fold_b": fold_b,
                    "features_in_a": len(features_a),
                    "features_in_b": len(features_b),
                    "intersection_size": len(intersection),
                    "union_size": len(union),
                    "jaccard_similarity": jaccard,
                }
            )

    return pd.DataFrame(rows)


def summarize_permutation_importance(
    permutation_scores: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate outer-fold permutation importance for each feature."""
    if permutation_scores.empty:
        return pd.DataFrame()

    return (
        permutation_scores
        .groupby(["model", "feature"], as_index=False)
        .agg(
            selected_in_folds=("selected", "sum"),
            mean_importance=("importance_mean", "mean"),
            std_importance_across_folds=("importance_mean", "std"),
            mean_within_fold_std=("importance_std", "mean"),
        )
        .sort_values(
            ["model", "mean_importance"],
            ascending=[True, False],
        )
    )


def build_error_summary(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Create confusion-matrix counts and error rates from predictions."""
    rows: list[dict[str, Any]] = []

    for model_name, frame in predictions.groupby("model"):
        y_true = frame["y_true"].to_numpy()
        y_pred = frame["y_pred"].to_numpy()

        true_positives = int(((y_true == -1) & (y_pred == -1)).sum())
        false_negatives = int(((y_true == -1) & (y_pred != -1)).sum())
        false_positives = int(((y_true != -1) & (y_pred == -1)).sum())
        true_negatives = int(((y_true != -1) & (y_pred != -1)).sum())

        positive_total = true_positives + false_negatives
        negative_total = true_negatives + false_positives

        rows.append(
            {
                "model": model_name,
                "observations": len(frame),
                "true_positive_phishing": true_positives,
                "false_negative": false_negatives,
                "false_positive": false_positives,
                "true_negative_legitimate": true_negatives,
                "false_negative_rate": (
                    false_negatives / positive_total if positive_total else np.nan
                ),
                "false_positive_rate": (
                    false_positives / negative_total if negative_total else np.nan
                ),
                "total_errors": int((~frame["correct"]).sum()),
                "error_rate": float((~frame["correct"]).mean()),
                "high_confidence_errors": int(
                    frame["high_confidence_error"].sum()
                ),
                "high_confidence_threshold": config.HIGH_CONFIDENCE_THRESHOLD,
            }
        )

    return pd.DataFrame(rows)


def build_error_by_feature_value(
    predictions: pd.DataFrame,
    X_dev: pd.DataFrame,
) -> pd.DataFrame:
    """Measure OOF error rates for every feature value and model."""
    feature_table = X_dev.reset_index(drop=False).rename(
        columns={"index": "original_dataframe_index"}
    )
    feature_table.insert(0, "sample_position", np.arange(len(feature_table)))

    rows: list[dict[str, Any]] = []

    for model_name, model_predictions in predictions.groupby("model"):
        merged = model_predictions.merge(
            feature_table,
            on="sample_position",
            how="left",
            validate="one_to_one",
        )

        for feature in X_dev.columns:
            grouped = merged.groupby(feature, dropna=False)

            for feature_value, group in grouped:
                observations = len(group)
                errors = int((~group["correct"]).sum())

                rows.append(
                    {
                        "model": model_name,
                        "feature": feature,
                        "feature_value": feature_value,
                        "observations": observations,
                        "errors": errors,
                        "error_rate": errors / observations,
                        "false_negatives": int(
                            (group["error_type"] == "false_negative").sum()
                        ),
                        "false_positives": int(
                            (group["error_type"] == "false_positive").sum()
                        ),
                        "high_confidence_errors": int(
                            group["high_confidence_error"].sum()
                        ),
                        "mean_phishing_probability": float(
                            group["phishing_probability"].mean()
                        ),
                    }
                )

    return pd.DataFrame(rows)


def build_model_disagreements(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Compare Decision Tree and Random Forest on the same OOF observations."""
    required_models = {"Decision Tree", "Random Forest"}
    if not required_models.issubset(set(predictions["model"].unique())):
        return pd.DataFrame()

    columns = [
        "sample_position",
        "sample_index",
        "outer_fold",
        "y_true",
        "y_pred",
        "phishing_probability",
        "predicted_confidence",
        "correct",
        "error_type",
    ]

    decision_tree = (
        predictions.loc[predictions["model"] == "Decision Tree", columns]
        .rename(
            columns={
                "outer_fold": "decision_tree_outer_fold",
                "y_pred": "decision_tree_prediction",
                "phishing_probability": "decision_tree_phishing_probability",
                "predicted_confidence": "decision_tree_confidence",
                "correct": "decision_tree_correct",
                "error_type": "decision_tree_error_type",
            }
        )
    )

    random_forest = (
        predictions.loc[predictions["model"] == "Random Forest", columns]
        .drop(columns=["sample_index", "y_true"])
        .rename(
            columns={
                "outer_fold": "random_forest_outer_fold",
                "y_pred": "random_forest_prediction",
                "phishing_probability": "random_forest_phishing_probability",
                "predicted_confidence": "random_forest_confidence",
                "correct": "random_forest_correct",
                "error_type": "random_forest_error_type",
            }
        )
    )

    disagreements = decision_tree.merge(
        random_forest,
        on="sample_position",
        how="inner",
        validate="one_to_one",
    )

    conditions = [
        disagreements["decision_tree_correct"]
        & disagreements["random_forest_correct"],
        disagreements["decision_tree_correct"]
        & ~disagreements["random_forest_correct"],
        ~disagreements["decision_tree_correct"]
        & disagreements["random_forest_correct"],
    ]
    labels = [
        "both_correct",
        "decision_tree_only_correct",
        "random_forest_only_correct",
    ]
    disagreements["comparison_outcome"] = np.select(
        conditions,
        labels,
        default="both_wrong",
    )
    disagreements["predictions_disagree"] = (
        disagreements["decision_tree_prediction"]
        != disagreements["random_forest_prediction"]
    )

    return disagreements


def compute_statistical_tests(
    nested_scores: pd.DataFrame,
) -> pd.DataFrame:
    """Compute the paired Wilcoxon signed-rank test."""
    rows: list[dict[str, Any]] = []

    fold_pivot = nested_scores.pivot(
        index="outer_fold",
        columns="model",
        values="macro_f1",
    )

    if {"Decision Tree", "Random Forest"}.issubset(fold_pivot.columns):
        paired = fold_pivot[["Decision Tree", "Random Forest"]].dropna()
        differences = paired["Random Forest"] - paired["Decision Tree"]

        if np.allclose(differences.to_numpy(), 0.0):
            statistic = 0.0
            p_value = 1.0
        else:
            try:
                result = wilcoxon(
                    paired["Decision Tree"],
                    paired["Random Forest"],
                    alternative="two-sided",
                    method="exact",
                )
            except ValueError:
                result = wilcoxon(
                    paired["Decision Tree"],
                    paired["Random Forest"],
                    alternative="two-sided",
                    method="auto",
                )
            statistic = float(result.statistic)
            p_value = float(result.pvalue)

        rows.append(
            {
                "test": "Wilcoxon signed-rank",
                "statistic": statistic,
                "p_value": p_value,
                "sample_size": len(paired),
                "mean_paired_difference_rf_minus_dt": float(differences.mean()),
            }
        )

    return pd.DataFrame(rows)

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


def final_test_prediction_table(
    *,
    final_model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
) -> pd.DataFrame:
    """Create a row-level prediction table for the untouched final test set."""
    y_pred, phishing_probability = predict_with_phishing_probability(
        final_model,
        X_test,
    )
    y_true = y_test.to_numpy()
    confidence = np.where(
        y_pred == -1,
        phishing_probability,
        1.0 - phishing_probability,
    )
    correct = y_true == y_pred

    error_type = np.select(
        [
            (y_true == -1) & (y_pred == -1),
            (y_true != -1) & (y_pred != -1),
            (y_true == -1) & (y_pred != -1),
        ],
        [
            "true_positive_phishing",
            "true_negative_legitimate",
            "false_negative",
        ],
        default="false_positive",
    )

    return pd.DataFrame(
        {
            "model": model_name,
            "sample_position": np.arange(len(X_test)),
            "sample_index": X_test.index,
            "y_true": y_true,
            "y_pred": y_pred,
            "phishing_probability": phishing_probability,
            "predicted_confidence": confidence,
            "correct": correct,
            "error_type": error_type,
            "high_confidence_error": (
                (~correct) & (confidence >= config.HIGH_CONFIDENCE_THRESHOLD)
            ),
        }
    )


def main() -> None:
    # 1. Resolve relative data and output directories.
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data"
    run_tag = experiment_tag(config.FEATURE_SELECTION_K_VALUES)
    output_dir = module_dir / "outputs" / run_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / "train_cleaned.csv"
    test_path = data_dir / "test_cleaned.csv"

    print("=" * 80)
    print("STARTING MODEL SELECTION & PERFORMANCE PIPELINE")
    print(f"Experiment: {run_tag}")
    print(f"k candidates: {config.FEATURE_SELECTION_K_VALUES}")
    print(f"Final test evaluation enabled: {config.EVALUATE_FINAL_TEST}")
    print("=" * 80)

    # 2. Load the cleaned development and test datasets.
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
    except FileNotFoundError as error:
        print(f"Error loading datasets: {error}")
        print("Please run your preprocessing pipeline first.")
        sys.exit(1)

    X_dev = train_df.drop(columns=["Result"])
    y_dev = train_df["Result"]
    X_test = test_df.drop(columns=["Result"])
    y_test = test_df["Result"]

    print(
        f"Development Set: {X_dev.shape[0]} samples with "
        f"{X_dev.shape[1]} features."
    )
    print(
        f"Final Test Set : {X_test.shape[0]} samples with "
        f"{X_test.shape[1]} features.\n"
    )

    # Save the settings needed to reproduce this run.
    run_configuration = pd.DataFrame(
        [
            {
                "experiment_tag": run_tag,
                "feature_selection_k_values": json.dumps(
                    config.FEATURE_SELECTION_K_VALUES
                ),
                "random_state": config.RANDOM_STATE,
                "primary_scoring": config.PRIMARY_SCORING,
                "random_forest_random_iterations": config.N_RANDOM_ITERATIONS,
                "outer_folds": outer_cv.n_splits,
                "final_inner_folds": final_inner_cv.n_splits,
                "compute_permutation_importance": (
                    config.COMPUTE_PERMUTATION_IMPORTANCE
                ),
                "permutation_repeats": config.PERMUTATION_N_REPEATS,
                "high_confidence_threshold": (
                    config.HIGH_CONFIDENCE_THRESHOLD
                ),
                "final_test_evaluated": config.EVALUATE_FINAL_TEST,
                "development_observations": len(X_dev),
                "test_observations": len(X_test),
                "input_features": X_dev.shape[1],
            }
        ]
    )
    save_csv(
        run_configuration,
        output_dir / "run_configuration.csv",
        "run configuration",
    )

    # 3. Compute the shared outer splits once. Both models use the same folds.
    outer_splits = list(outer_cv.split(X_dev, y_dev))

    # 4. Nested CV: Decision Tree.
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

    # 5. Nested CV: Random Forest.
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

    # 6. Combine every nested-CV output table.
    nested_scores = combine_result_tables(model_results, "fold_scores")
    best_parameters = combine_result_tables(model_results, "best_parameters")
    selected_features = combine_result_tables(model_results, "selected_features")
    oof_predictions = combine_result_tables(model_results, "oof_predictions")
    inner_search_results = combine_result_tables(
        model_results,
        "inner_search_results",
    )
    permutation_scores = combine_result_tables(
        model_results,
        "permutation_importance",
    )

    nested_summary = (
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

    feature_frequency = summarize_feature_frequency(selected_features)
    feature_stability = compute_feature_stability(selected_features)
    permutation_summary = summarize_permutation_importance(permutation_scores)
    error_summary = build_error_summary(oof_predictions)
    error_by_feature_value = build_error_by_feature_value(
        oof_predictions,
        X_dev,
    )
    model_disagreements = build_model_disagreements(oof_predictions)
    statistical_tests = compute_statistical_tests(
        nested_scores,
    )

    print("-" * 80)
    print("Nested Cross-Validation Performance Summary")
    print("-" * 80)
    print(nested_summary.round(4))
    print()

    # 7. Save nested-CV outputs.
    save_csv(
        nested_scores,
        output_dir / "nested_cv_fold_scores.csv",
        "outer-fold scores",
    )
    save_csv(
        nested_summary,
        output_dir / "nested_cv_summary.csv",
        "aggregated performance summary",
        index=True,
    )
    save_csv(
        best_parameters,
        output_dir / "nested_cv_best_parameters.csv",
        "best parameters by outer fold",
    )
    save_csv(
        inner_search_results,
        output_dir / "nested_cv_inner_search_results.csv",
        "all inner-search candidates",
    )
    save_csv(
        selected_features,
        output_dir / "nested_cv_selected_features.csv",
        "feature-selection details by outer fold",
    )
    save_csv(
        feature_frequency,
        output_dir / "nested_cv_feature_frequency.csv",
        "feature-selection frequencies",
    )
    save_csv(
        feature_stability,
        output_dir / "nested_cv_feature_stability.csv",
        "feature-subset stability",
    )
    save_csv(
        oof_predictions,
        output_dir / "nested_cv_oof_predictions.csv",
        "out-of-fold predictions",
    )
    save_csv(
        error_summary,
        output_dir / "nested_cv_error_summary.csv",
        "out-of-fold error summary",
    )
    save_csv(
        error_by_feature_value,
        output_dir / "nested_cv_error_by_feature_value.csv",
        "error rates by feature value",
    )
    save_csv(
        model_disagreements,
        output_dir / "nested_cv_model_disagreements.csv",
        "paired model disagreements",
    )
    save_csv(
        statistical_tests,
        output_dir / "nested_cv_statistical_tests.csv",
        "paired statistical tests",
    )

    if not permutation_scores.empty:
        save_csv(
            permutation_scores,
            output_dir / "nested_cv_permutation_importance.csv",
            "permutation importance by outer fold",
        )
        save_csv(
            permutation_summary,
            output_dir / "nested_cv_permutation_importance_summary.csv",
            "aggregated permutation importance",
        )

    # 8. Render the model comparison figure.
    plot_pdf_path = output_dir / "nested_cv_model_comparison.pdf"
    plot_nested_cv_comparison(
        nested_scores=nested_scores,
        output_pdf_path=plot_pdf_path,
    )
    print(f"-> Generated model comparison chart: {plot_pdf_path.name}\n")

    # 9. Select the best model family from the nested outer-fold estimates.
    best_model_family = str(nested_summary.index[0])
    print(f"Selected Model Family for Final Deployment: {best_model_family}")

    # 10. Repeat the hyperparameter search on the complete development set.
    print("-" * 80)
    print(f"Fitting final {best_model_family} on full Development Set...")
    print("-" * 80)

    if best_model_family == "Decision Tree":
        final_search: GridSearchCV | RandomizedSearchCV = GridSearchCV(
            estimator=decision_tree_pipeline,
            param_grid=config.decision_tree_param_grid,
            scoring=config.PRIMARY_SCORING,
            cv=final_inner_cv,
            refit=True,
            n_jobs=-1,
            return_train_score=False,
            error_score="raise",
        )
    elif best_model_family == "Random Forest":
        final_search = RandomizedSearchCV(
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
    else:
        raise RuntimeError(f"Unknown model family: {best_model_family}")

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
        output_dir / "final_development_search_results.csv",
        "final development search candidates",
    )
    save_csv(
        final_best_parameters,
        output_dir / "final_best_parameters.csv",
        "final best parameters",
    )
    save_csv(
        final_features,
        output_dir / "final_selected_features.csv",
        "final feature-selection results",
    )

    # Report figure: final Mutual Information feature ranking.
    feature_ranking_pdf = output_dir / "feature_selection_ranking.pdf"
    plot_selected_feature_ranking(
        selected_features=final_features,
        output_pdf_path=feature_ranking_pdf,
        max_display=15,
    )
    print(f"-> Generated feature-selection ranking: {feature_ranking_pdf.name}")

    # Report figure: best hyperparameter-search candidates on the full
    # development set. This uses development-CV scores only, never the test set.
    hyperparameter_pdf = output_dir / "hyperparameter_optimization.pdf"
    plot_hyperparameter_optimization(
        search_results=final_search_table,
        output_pdf_path=hyperparameter_pdf,
        model_name=best_model_family,
        max_candidates=15,
    )
    print(f"-> Generated hyperparameter-search chart: {hyperparameter_pdf.name}\n")

    # 11. Evaluate the held-out test set only for the final chosen experiment.
    if config.EVALUATE_FINAL_TEST:
        print("-" * 80)
        print("FINAL TEST SET PERFORMANCE EVALUATION (Unseen Data)")
        print("-" * 80)

        final_test_metrics = compute_classification_metrics(
            fitted_pipeline=final_model,
            X_validation=X_test,
            y_validation=y_test,
        )
        final_test_metrics_table = pd.DataFrame(
            [{"model": best_model_family, **final_test_metrics}]
        )
        final_test_predictions = final_test_prediction_table(
            final_model=final_model,
            X_test=X_test,
            y_test=y_test,
            model_name=best_model_family,
        )
        final_test_error_summary = build_error_summary(final_test_predictions)

        save_csv(
            final_test_metrics_table,
            output_dir / "final_test_metrics.csv",
            "final test metrics",
        )
        save_csv(
            final_test_predictions,
            output_dir / "final_test_predictions.csv",
            "final test predictions",
        )
        save_csv(
            final_test_error_summary,
            output_dir / "final_test_error_summary.csv",
            "final test error summary",
        )

        confusion_matrix_pdf = output_dir / "final_test_confusion_matrix.pdf"
        plot_final_test_confusion_matrix(
            final_test_predictions=final_test_predictions,
            output_pdf_path=confusion_matrix_pdf,
        )
        print(f"-> Generated final confusion matrix: {confusion_matrix_pdf.name}")

        roc_curve_pdf = output_dir / "final_test_roc_curve.pdf"
        final_roc_auc = plot_final_test_roc_curve(
            final_test_predictions=final_test_predictions,
            output_pdf_path=roc_curve_pdf,
        )
        print(
            f"-> Generated final ROC curve: {roc_curve_pdf.name} "
            f"(AUC={final_roc_auc:.4f})"
        )

        for metric_name, metric_value in final_test_metrics.items():
            print(f"  {metric_name:<20}: {metric_value:.4f}")
    else:
        print("-" * 80)
        print("FINAL TEST SET NOT EVALUATED")
        print("Set EVALUATE_FINAL_TEST = True only after choosing the final setup.")
        print("-" * 80)

    print("=" * 80)
    print("MODEL SELECTION PIPELINE RUN COMPLETED SUCCESSFULLY")
    print(f"Outputs directory: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()