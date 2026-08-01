import sys
from pathlib import Path
import pandas as pd

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Ensure the module directory is in sys.path so local siblings resolve correctly
module_dir = Path(__file__).resolve().parent
if str(module_dir) not in sys.path:
    sys.path.insert(0, str(module_dir))

import config
from engine import nested_cross_validation
from plots import plot_nested_cv_comparison
from utils import compute_classification_metrics


def main() -> None:
    # 1. Resolve relative data and output directories
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data"
    output_dir = module_dir / "outputs"
    
    # Create the outputs directory if it does not exist
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / "train_cleaned.csv"
    test_path = data_dir / "test_cleaned.csv"

    print("=" * 80)
    print("STARTING MODEL SELECTION & PERFORMANCE PIPELINE")
    print("=" * 80)

    # 2. Load the cleaned train and test datasets
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
    except FileNotFoundError as e:
        print(f"Error loading datasets: {e}")
        print("Please run your preprocessing pipeline (make install) first.")
        sys.exit(1)

    # Separate features (X) and target (y)
    X_dev = train_df.drop(columns=["Result"])
    y_dev = train_df["Result"]
    
    X_test = test_df.drop(columns=["Result"])
    y_test = test_df["Result"]

    print(f"Development Set: {X_dev.shape[0]} samples with {X_dev.shape[1]} features.")
    print(f"Final Test Set : {X_test.shape[0]} samples with {X_test.shape[1]} features.\n")

    # 3. Compute outer cross-validation split indices
    outer_splits = list(config.outer_cv.split(X_dev, y_dev))

    # 4. Run Nested CV for Decision Tree
    print("-" * 80)
    print("Executing Nested CV: Decision Tree")
    print("-" * 80)
    decision_tree_scores, decision_tree_best_params = nested_cross_validation(
        model_name="Decision Tree",
        pipeline=config.decision_tree_pipeline,
        search_space=config.decision_tree_param_grid,
        search_method="grid",
        X=X_dev,
        y=y_dev,
        outer_splits=outer_splits,
    )

    # 5. Run Nested CV for Random Forest
    print("-" * 80)
    print("Executing Nested CV: Random Forest")
    print("-" * 80)
    random_forest_scores, random_forest_best_params = nested_cross_validation(
        model_name="Random Forest",
        pipeline=config.random_forest_pipeline,
        search_space=config.random_forest_param_distributions,
        search_method="random",
        X=X_dev,
        y=y_dev,
        outer_splits=outer_splits,
        n_random_iterations=config.N_RANDOM_ITERATIONS,
    )

    # 6. Combine and Summarize Outer-Fold Performance Metrics
    nested_scores = pd.concat(
        [decision_tree_scores, random_forest_scores],
        ignore_index=True,
    )

    summary = (
        nested_scores
        .groupby("model")
        .agg(
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            phishing_precision_mean=("phishing_precision", "mean"),
            phishing_recall_mean=("phishing_recall", "mean"),
            accuracy_mean=("accuracy", "mean"),
            roc_auc_mean=("roc_auc", "mean"),
        )
        .sort_values(by="macro_f1_mean", ascending=False)
    )

    print("-" * 80)
    print("Nested Cross-Validation Performance Summary")
    print("-" * 80)
    print(summary.round(4))
    print()

    # 7. Save outputs and metrics to disk
    scores_csv_path = output_dir / "nested_cv_fold_scores.csv"
    summary_csv_path = output_dir / "nested_cv_summary.csv"
    plot_pdf_path = output_dir / "nested_cv_model_comparison.pdf"

    nested_scores.to_csv(scores_csv_path, index=False)
    summary.to_csv(summary_csv_path)

    print(f"-> Saved raw fold scores to: {scores_csv_path.name}")
    print(f"-> Saved aggregated performance summary to: {summary_csv_path.name}")

    # 8. Render the model performance comparison boxplot
    plot_nested_cv_comparison(
        nested_scores=nested_scores,
        output_pdf_path=plot_pdf_path,
    )
    print(f"-> Generated performance comparison chart: {plot_pdf_path.name}\n")

    # 9. Determine and select the best model family
    best_model_family = summary.index[0]
    print(f"Selected Model Family for Final Deployment: {best_model_family}")

    # 10. Repeat Hyperparameter Search on the COMPLETE development set
    print("-" * 80)
    print(f"Fitting final {best_model_family} on full Development Set...")
    print("-" * 80)

    if best_model_family == "Decision Tree":
        final_search = GridSearchCV(
            estimator=config.decision_tree_pipeline,
            param_grid=config.decision_tree_param_grid,
            scoring=config.PRIMARY_SCORING,
            cv=config.final_inner_cv,
            refit=True,
            n_jobs=-1,
            return_train_score=False,
            error_score="raise",
        )
    elif best_model_family == "Random Forest":
        final_search = RandomizedSearchCV(
            estimator=config.random_forest_pipeline,
            param_distributions=config.random_forest_param_distributions,
            n_iter=config.N_RANDOM_ITERATIONS,
            scoring=config.PRIMARY_SCORING,
            cv=config.final_inner_cv,
            refit=True,
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
            return_train_score=False,
            error_score="raise",
        )
    else:
        raise RuntimeError(f"Unknown model family: {best_model_family}")

    # Fit selected pipeline
    final_search.fit(X_dev, y_dev)
    final_model = final_search.best_estimator_

    print(f"Final Development CV Score ({config.PRIMARY_SCORING}): {final_search.best_score_:.4f}")
    print(f"Final Optimized Hyperparameters: {final_search.best_params_}\n")

    # 11. Final Test Evaluation (Evaluate generalizing performance)
    print("-" * 80)
    print("FINAL TEST SET PERFORMANCE EVALUATION (Unseen Data)")
    print("-" * 80)
    final_test_metrics = compute_classification_metrics(
        fitted_pipeline=final_model,
        X_validation=X_test,
        y_validation=y_test,
    )

    for metric_name, metric_value in final_test_metrics.items():
        print(f"  {metric_name:<20}: {metric_value:.4f}")

    print("=" * 80)
    print("MODEL SELECTION PIPELINE RUN COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()