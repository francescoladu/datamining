from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

module_dir = Path(__file__).resolve().parent
code_dir = module_dir.parent
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

from final_evaluation import config
from model_selection import config as model_selection_config
from model_selection.diagnostics import build_error_summary
from model_selection.plots import (
    plot_final_test_confusion_matrix,
    plot_final_test_roc_curve,
)
from model_selection.utils import (
    compute_classification_metrics,
    predict_with_phishing_probability,
)
from shared.config import DATA_DIR, PHISHING_LABEL
from shared.modeling import (
    build_pipeline,
    load_development_and_test,
    load_final_model_configuration,
)


def save_csv(dataframe: pd.DataFrame, path: Path, description: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False)
    print(f"-> Saved {description}: {path.name}")


def save_json(payload: dict[str, Any], path: Path, description: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"-> Saved {description}: {path.name}")


def final_test_prediction_table(
    *,
    final_model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    w_test: pd.Series,
    model_name: str,
) -> pd.DataFrame:
    y_pred, phishing_probability = predict_with_phishing_probability(
        final_model,
        X_test,
    )
    y_true = y_test.to_numpy()
    confidence = np.where(
        y_pred == PHISHING_LABEL,
        phishing_probability,
        1.0 - phishing_probability,
    )
    correct = y_true == y_pred

    error_type = np.select(
        [
            (y_true == PHISHING_LABEL) & (y_pred == PHISHING_LABEL),
            (y_true != PHISHING_LABEL) & (y_pred != PHISHING_LABEL),
            (y_true == PHISHING_LABEL) & (y_pred != PHISHING_LABEL),
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
            "sample_weight": w_test.to_numpy(),
            "y_true": y_true,
            "y_pred": y_pred,
            "phishing_probability": phishing_probability,
            "predicted_confidence": confidence,
            "correct": correct,
            "error_type": error_type,
            "high_confidence_error": (
                (~correct)
                & (confidence >= model_selection_config.HIGH_CONFIDENCE_THRESHOLD)
            ),
        }
    )


def evaluate_experiment_pipeline(
    *,
    experiment_id: str,
    dev_path: Path,
    test_path: Path,
    parameters_path: Path,
    output_dir: Path,
    weighted: bool,
) -> dict[str, Any]:
    """Train the frozen model on its 80% train data and evaluate on its own 20% test data."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = output_dir / "metrics"
    diag_dir = output_dir / "diagnostics"
    figures_dir = output_dir / "figures"
    for d in (metrics_dir, diag_dir, figures_dir):
        d.mkdir(parents=True, exist_ok=True)

    print("-" * 80)
    print(f"Evaluating {experiment_id}")
    print(f"Train path: {dev_path.name}")
    print(f"Test path:  {test_path.name}")
    print("-" * 80)

    model_name, development_cv_score, parameters = load_final_model_configuration(parameters_path)
    X_dev, y_dev, w_dev, X_test, y_test, w_test = load_development_and_test(dev_path, test_path)

    final_model = build_pipeline(model_name)
    final_model.set_params(**parameters)

    fit_kwargs = {}
    if weighted and w_dev is not None:
        fit_kwargs["classifier__sample_weight"] = np.asarray(w_dev)

    final_model.fit(X_dev, y_dev, **fit_kwargs)

    train_metrics = compute_classification_metrics(
        fitted_pipeline=final_model,
        X_validation=X_dev,
        y_validation=y_dev,
        sample_weight=w_dev if weighted else None,
)
    test_metrics = compute_classification_metrics(
        fitted_pipeline=final_model,
        X_validation=X_test,
        y_validation=y_test,
        sample_weight=w_test if weighted else None,
    )

    predictions = final_test_prediction_table(
        final_model=final_model,
        X_test=X_test,
        y_test=y_test,
        w_test=w_test,
        model_name=model_name,
    )
    error_summary = build_error_summary(predictions)

    metrics_table = pd.DataFrame([
        {
        "experiment": experiment_id,
        "model": model_name,
        "development_cv_accuracy": development_cv_score,
        "train_accuracy": train_metrics["accuracy"],
        "test_accuracy": test_metrics["accuracy"],
        "test_phishing_precision": test_metrics["phishing_precision"],
        "test_phishing_recall": test_metrics["phishing_recall"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_roc_auc": test_metrics["roc_auc"],
        }
    ])

    save_csv(metrics_table, metrics_dir / "test_metrics.csv", f"{experiment_id} test metrics")
    save_csv(predictions, diag_dir / "test_predictions.csv", f"{experiment_id} test predictions")
    save_csv(error_summary, diag_dir / "error_summary.csv", f"{experiment_id} error summary")

    plot_final_test_confusion_matrix(predictions, figures_dir / "confusion_matrix.pdf")
    plot_final_test_roc_curve(predictions, figures_dir / "roc_curve.pdf")
    evaluation_summary = {
        "experiment": experiment_id,
        "run_name": config.RUN_NAME,
        "model": model_name,
        "weighted": weighted,
        "development_cv_accuracy": development_cv_score,
        "train_observations": len(X_dev),
        "test_observations": len(X_test),
        "train_weighted_mass": float(w_dev.sum()),
        "test_weighted_mass": float(w_test.sum()),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "frozen_parameters": parameters,
        "source_final_best_parameters": str(parameters_path),
}

    save_json(
        evaluation_summary,
        output_dir / "evaluation_summary.json",
        f"{experiment_id} evaluation summary",
    )



    return {
        "experiment": experiment_id,
        "model": model_name,
        "development_cv_accuracy": development_cv_score,
        "train_accuracy": train_metrics["accuracy"],
        "test_accuracy": test_metrics["accuracy"],
        "test_precision": test_metrics["phishing_precision"],
        "test_recall": test_metrics["phishing_recall"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_roc_auc": test_metrics["roc_auc"],
}


def parse_arguments() -> argparse.Namespace:
    """
    Select exactly one experiment for final held-out evaluation.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run final held-out test evaluation "
            "for one Data Mining experiment."
        )
    )

    parser.add_argument(
        "experiment",
        choices=["1", "2", "3"],
        help=(
            "Experiment to evaluate: "
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
        f"FINAL HELD-OUT TEST EVALUATION "
        f"- EXPERIMENT {args.experiment}"
    )
    print(f"Run tag: {config.RUN_NAME}")
    print("=" * 80)

    model_selection_outputs = (
        code_dir
        / "model_selection"
        / "outputs"
    )

    run_tag = config.RUN_NAME

    # =========================================================
    # EXPERIMENT 1
    # =========================================================
    if args.experiment == "1":
        parameters_path = (
            model_selection_outputs
            / "experiment_1_raw"
            / run_tag
            / "hyperparameter_search"
            / "final_best_parameters.csv"
        )

        if not parameters_path.exists():
            raise FileNotFoundError(
                "Experiment 1 final parameters not found.\n"
                f"{parameters_path}\n"
                "Run model selection first:\n"
                "python code/model_selection/main.py 1"
            )

        result = evaluate_experiment_pipeline(
            experiment_id="Experiment 1 (Raw Data)",
            dev_path=(
                DATA_DIR
                / "experiment_1_raw_train.csv"
            ),
            test_path=(
                DATA_DIR
                / "experiment_1_raw_test.csv"
            ),
            parameters_path=parameters_path,
            output_dir=(
                config.OUTPUT_DIR
                / "experiment_1_raw"
            ),
            weighted=False,
        )

    # =========================================================
    # EXPERIMENT 2
    # =========================================================
    elif args.experiment == "2":
        parameters_path = (
            model_selection_outputs
            / "experiment_2_standard_dedup"
            / run_tag
            / "hyperparameter_search"
            / "final_best_parameters.csv"
        )

        if not parameters_path.exists():
            raise FileNotFoundError(
                "Experiment 2 final parameters not found.\n"
                f"{parameters_path}\n"
                "Run model selection first:\n"
                "python code/model_selection/main.py 2"
            )

        result = evaluate_experiment_pipeline(
            experiment_id=(
                "Experiment 2 (Exact Dedup)"
            ),
            dev_path=(
                DATA_DIR
                / "experiment_2_standard_dedup_train.csv"
            ),
            test_path=(
                DATA_DIR
                / "experiment_2_standard_dedup_test.csv"
            ),
            parameters_path=parameters_path,
            output_dir=(
                config.OUTPUT_DIR
                / "experiment_2_standard_dedup"
            ),
            weighted=False,
        )

    # =========================================================
    # EXPERIMENT 3
    # =========================================================
    elif args.experiment == "3":
        parameters_path = (
            config.FINAL_BEST_PARAMETERS_PATH
        )

        if not parameters_path.exists():
            raise FileNotFoundError(
                "Experiment 3 final parameters not found.\n"
                f"{parameters_path}\n"
                "Run model selection first:\n"
                "python code/model_selection/main.py 3"
            )

        result = evaluate_experiment_pipeline(
            experiment_id=(
                "Experiment 3 (Weighted Dedup)"
            ),
            dev_path=config.DEVELOPMENT_DATA_PATH,
            test_path=config.TEST_DATA_PATH,
            parameters_path=parameters_path,
            output_dir=config.OUTPUT_DIR,
            weighted=True,
        )

    else:
        raise RuntimeError(
            "Unexpected experiment identifier."
        )

    print("\n" + "=" * 80)
    print(
        f"EXPERIMENT {args.experiment} "
        "FINAL TEST RESULTS"
    )
    print("=" * 80)

    result_table = pd.DataFrame(
        [result]
    )

    print(
        result_table.to_string(
            index=False
        )
    )

    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
