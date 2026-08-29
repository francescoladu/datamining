from __future__ import annotations

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
from shared.config import PHISHING_LABEL
from shared.modeling import (
    build_pipeline,
    load_development_and_test,
    load_final_model_configuration,
)


def save_csv(
    dataframe: pd.DataFrame,
    path: Path,
    description: str,
) -> None:
    """Save one final-evaluation table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False)
    print(f"-> Saved {description}: {path.name}")


def save_json(payload: dict[str, Any], path: Path, description: str) -> None:
    """Save one final-evaluation metadata file."""
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
    model_name: str,
) -> pd.DataFrame:
    """Create row-level predictions for the untouched held-out test set."""
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
            "y_true": y_true,
            "y_pred": y_pred,
            "phishing_probability": phishing_probability,
            "predicted_confidence": confidence,
            "correct": correct,
            "error_type": error_type,
            "high_confidence_error": (
                (~correct)
                & (
                    confidence
                    >= model_selection_config.HIGH_CONFIDENCE_THRESHOLD
                )
            ),
        }
    )


def main() -> None:
    """Evaluate one frozen model-selection run on the held-out test set."""
    config.create_output_directories()

    print("=" * 80)
    print("FINAL HELD-OUT TEST EVALUATION")
    print(f"Frozen model-selection run: {config.RUN_NAME}")
    print("This entry point intentionally reads the held-out test set.")
    print("=" * 80)

    # Validate the frozen configuration before the held-out test file is read.
    model_name, development_cv_score, parameters = (
        load_final_model_configuration(
            config.FINAL_BEST_PARAMETERS_PATH
        )
    )

    X_dev, y_dev, X_test, y_test = load_development_and_test(
        config.DEVELOPMENT_DATA_PATH,
        config.TEST_DATA_PATH,
    )

    final_model = build_pipeline(model_name)
    final_model.set_params(**parameters)
    final_model.fit(X_dev, y_dev)

    test_metrics = compute_classification_metrics(
        fitted_pipeline=final_model,
        X_validation=X_test,
        y_validation=y_test,
    )
    metrics_table = pd.DataFrame(
        [
            {
                "model": model_name,
                "development_cv_score": development_cv_score,
                **test_metrics,
            }
        ]
    )

    predictions = final_test_prediction_table(
        final_model=final_model,
        X_test=X_test,
        y_test=y_test,
        model_name=model_name,
    )
    error_summary = build_error_summary(predictions)

    save_csv(
        metrics_table,
        config.METRICS_DIR / "test_metrics.csv",
        "final test metrics",
    )
    save_csv(
        predictions,
        config.DIAGNOSTICS_DIR / "test_predictions.csv",
        "final test predictions",
    )
    save_csv(
        error_summary,
        config.DIAGNOSTICS_DIR / "error_summary.csv",
        "final test error summary",
    )

    confusion_matrix_pdf = config.FIGURES_DIR / "confusion_matrix.pdf"
    plot_final_test_confusion_matrix(
        final_test_predictions=predictions,
        output_pdf_path=confusion_matrix_pdf,
    )
    print(f"-> Generated confusion matrix: {confusion_matrix_pdf.name}")

    roc_curve_pdf = config.FIGURES_DIR / "roc_curve.pdf"
    final_roc_auc = plot_final_test_roc_curve(
        final_test_predictions=predictions,
        output_pdf_path=roc_curve_pdf,
    )
    print(
        f"-> Generated ROC curve: {roc_curve_pdf.name} "
        f"(AUC={final_roc_auc:.4f})"
    )

    evaluation_summary = {
        "run_name": config.RUN_NAME,
        "model": model_name,
        "development_cv_score": development_cv_score,
        "test_observations": len(X_test),
        "test_metrics": test_metrics,
        "frozen_parameters": parameters,
        "source_final_best_parameters": str(
            config.FINAL_BEST_PARAMETERS_PATH
        ),
    }
    save_json(
        evaluation_summary,
        config.OUTPUT_DIR / "evaluation_summary.json",
        "evaluation summary",
    )

    print("-" * 80)
    for metric_name, metric_value in test_metrics.items():
        print(f"  {metric_name:<20}: {metric_value:.4f}")
    print("-" * 80)
    print(f"Outputs directory: {config.OUTPUT_DIR}")
    print("FINAL TEST EVALUATION COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()
