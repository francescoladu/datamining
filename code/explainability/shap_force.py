from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from scipy import sparse
from sklearn.pipeline import Pipeline

from config import (
    CLASS_TO_EXPLAIN,
    OUTPUT_DIR,
    PLOT_DPI,
    RANDOM_STATE,
    SAMPLE_POSITION,
    SAVE_PDF,
    SAVE_PNG,
    SHAP_BACKGROUND_SIZE,
    create_output_directory,
)


def _dense(
    values: Any,
) -> np.ndarray:
    """Convert sparse transformed values to a dense array."""

    if sparse.issparse(values):
        values = values.toarray()

    return np.asarray(values)


def _extract_class_values(
    explanation: shap.Explanation,
    class_position: int,
    number_of_classes: int,
) -> tuple[float, np.ndarray]:
    """
    Extract the SHAP baseline and contributions for one class.

    Recent SHAP versions normally return:
    samples × features × classes.

    Some versions return only one output for binary
    classification. In that case, the output normally
    corresponds to classes_[1].
    """

    values = np.asarray(
        explanation.values
    )

    base_values = np.asarray(
        explanation.base_values
    )

    # Multiclass or binary output with one axis per class.
    if values.ndim == 3:
        contributions = values[
            0,
            :,
            class_position,
        ]

        if base_values.ndim == 2:
            baseline = float(
                base_values[
                    0,
                    class_position,
                ]
            )

        elif base_values.ndim == 1:
            baseline = float(
                base_values[class_position]
            )

        else:
            raise ValueError(
                "Unexpected SHAP baseline shape: "
                f"{base_values.shape}"
            )

        return baseline, contributions

    # Single-output binary explanation.
    if values.ndim == 2:
        contributions = values[0].copy()

        baseline = float(
            base_values.reshape(-1)[0]
        )

        # A single binary output normally explains classes_[1].
        # The probability of classes_[0] is obtained by inversion.
        if (
            number_of_classes == 2
            and class_position == 0
        ):
            baseline = 1.0 - baseline
            contributions = -contributions

        elif number_of_classes > 2:
            raise ValueError(
                "SHAP returned one output for a multiclass model."
            )

        return baseline, contributions

    raise ValueError(
        "Unexpected SHAP value shape: "
        f"{values.shape}"
    )


def run_shap_force(
    model: Pipeline,
    X_background: pd.DataFrame,
    X_test: pd.DataFrame,
    sample_position: int = SAMPLE_POSITION,
    class_to_explain: Any = CLASS_TO_EXPLAIN,
) -> dict[str, Any]:
    """
    Compute, save, and plot one local SHAP explanation.

    The complete pipeline is used for predictions. The feature
    selector is applied first, and TreeExplainer is then created
    for the final tree-based classifier.
    """

    if not isinstance(model, Pipeline):
        raise TypeError(
            "model must be a fitted sklearn Pipeline."
        )

    if sample_position < 0 or sample_position >= len(X_test):
        raise IndexError(
            "sample_position is outside the test set."
        )

    if list(X_background.columns) != list(X_test.columns):
        raise ValueError(
            "Training and test features must have "
            "the same order."
        )

    if "feature_selection" not in model.named_steps:
        raise ValueError(
            "The pipeline does not contain "
            "'feature_selection'."
        )

    if "classifier" not in model.named_steps:
        raise ValueError(
            "The pipeline does not contain 'classifier'."
        )

    selector = model.named_steps[
        "feature_selection"
    ]

    classifier = model.named_steps[
        "classifier"
    ]

    selected_mask = np.asarray(
        selector.get_support(),
        dtype=bool,
    )

    if selected_mask.size != X_background.shape[1]:
        raise ValueError(
            "The feature-selection mask does not match "
            "the input feature count."
        )

    selected_features = (
        X_background.columns[
            selected_mask
        ].tolist()
    )

    transformed_background = pd.DataFrame(
        _dense(
            selector.transform(
                X_background
            )
        ),
        columns=selected_features,
        index=X_background.index,
    )

    transformed_sample = pd.DataFrame(
        _dense(
            selector.transform(
                X_test.iloc[
                    [sample_position]
                ]
            )
        ),
        columns=selected_features,
        index=[
            X_test.index[
                sample_position
            ]
        ],
    )

    # Use a reproducible sample of the training set as the
    # SHAP reference population.
    background_size = min(
        SHAP_BACKGROUND_SIZE,
        len(transformed_background),
    )

    background_reference = (
        transformed_background.sample(
            n=background_size,
            random_state=RANDOM_STATE,
        )
    )

    classes = np.asarray(
        model.classes_
    )

    class_matches = np.flatnonzero(
        classes == class_to_explain
    )

    if len(class_matches) == 0:
        raise ValueError(
            f"Class {class_to_explain!r} not found in "
            f"model.classes_: {classes.tolist()}"
        )

    class_position = int(
        class_matches[0]
    )

    original_sample = X_test.iloc[
        [sample_position]
    ]

    predicted_class = model.predict(
        original_sample
    )[0]

    predicted_probability = float(
        model.predict_proba(
            original_sample
        )[
            0,
            class_position,
        ]
    )

    explainer = shap.TreeExplainer(
        classifier,
        data=background_reference,
        feature_perturbation="interventional",
        model_output="probability",
    )

    explanation = explainer(
        transformed_sample,
        check_additivity=False,
    )

    baseline, contributions = (
        _extract_class_values(
            explanation=explanation,
            class_position=class_position,
            number_of_classes=len(classes),
        )
    )

    reconstructed_probability = float(
        baseline
        + contributions.sum()
    )

    additivity_error = abs(
        reconstructed_probability
        - predicted_probability
    )

    contributions_df = pd.DataFrame(
        {
            "feature": selected_features,
            "feature_value": (
                transformed_sample
                .iloc[0]
                .to_numpy()
            ),
            "shap_value": contributions,
        }
    )

    contributions_df[
        "absolute_shap_value"
    ] = contributions_df[
        "shap_value"
    ].abs()

    contributions_df[
        "direction"
    ] = np.select(
        [
            contributions_df["shap_value"] > 0,
            contributions_df["shap_value"] < 0,
        ],
        [
            "increases_phishing_probability",
            "decreases_phishing_probability",
        ],
        default="no_effect",
    )

    contributions_df = (
        contributions_df
        .sort_values(
            "absolute_shap_value",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    contributions_df.insert(
        0,
        "rank",
        np.arange(
            1,
            len(contributions_df) + 1,
        ),
    )

    summary_df = pd.DataFrame(
        {
            "sample_position": [
                sample_position
            ],
            "sample_index": [
                X_test.index[
                    sample_position
                ]
            ],
            "predicted_class": [
                predicted_class
            ],
            "explained_class": [
                class_to_explain
            ],
            "explained_phishing_probability": [
                predicted_probability
            ],
            "base_value": [
                baseline
            ],
            "sum_of_shap_values": [
                float(
                    contributions.sum()
                )
            ],
            "reconstructed_probability": [
                reconstructed_probability
            ],
            "absolute_additivity_error": [
                additivity_error
            ],
            "background_size": [
                background_size
            ],
        }
    )

    create_output_directory()

    suffix = (
        f"sample_{sample_position:04d}"
    )

    contributions_path = (
        OUTPUT_DIR
        / f"shap_local_contributions_{suffix}.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / f"shap_prediction_summary_{suffix}.csv"
    )

    png_path = (
        OUTPUT_DIR
        / f"shap_force_{suffix}.png"
    )

    pdf_path = (
        OUTPUT_DIR
        / f"shap_force_{suffix}.pdf"
    )

    contributions_df.to_csv(
        contributions_path,
        index=False,
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    shap.force_plot(
        base_value=baseline,
        shap_values=contributions,
        features=(
            transformed_sample
            .iloc[0]
            .to_numpy()
        ),
        feature_names=selected_features,
        matplotlib=True,
        show=False,
    )

    figure = plt.gcf()

    figure.set_size_inches(
        14,
        3.5,
    )

    plt.title(
        "Local SHAP Explanation — P(phishing | x)",
        pad=25,
    )

    if SAVE_PNG:
        figure.savefig(
            png_path,
            dpi=PLOT_DPI,
            bbox_inches="tight",
        )

    if SAVE_PDF:
        figure.savefig(
            pdf_path,
            bbox_inches="tight",
        )

    plt.close(figure)

    print(
        "\nLOCAL EXPLAINABILITY — "
        "SHAP FORCE PLOT"
    )

    print(
        f"Sample position: {sample_position}"
    )

    print(
        f"Predicted class: {predicted_class}"
    )

    print(
        f"Explained class: "
        f"{class_to_explain} (phishing)"
    )

    print(
        "P(phishing | x): "
        f"{predicted_probability:.6f}"
    )

    print(
        f"SHAP baseline: {baseline:.6f}"
    )

    print(
        "Baseline + contributions: "
        f"{reconstructed_probability:.6f}"
    )

    print(
        "Additivity error: "
        f"{additivity_error:.8f}"
    )

    print(
        "\nLargest local contributions:"
    )

    print(
        contributions_df[
            [
                "rank",
                "feature",
                "feature_value",
                "shap_value",
                "direction",
            ]
        ].to_string(
            index=False,
        )
    )

    print(
        "\nSHAP contributions saved to:"
        f"\n{contributions_path}"
    )

    print(
        "\nSHAP summary saved to:"
        f"\n{summary_path}"
    )

    if SAVE_PNG:
        print(
            "\nSHAP force plot PNG saved to:"
            f"\n{png_path}"
        )

    if SAVE_PDF:
        print(
            "\nSHAP force plot PDF saved to:"
            f"\n{pdf_path}"
        )

    return {
        "contributions": contributions_df,
        "summary": summary_df,
        "png_path": (
            png_path if SAVE_PNG else None
        ),
        "pdf_path": (
            pdf_path if SAVE_PDF else None
        ),
    }