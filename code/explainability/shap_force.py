from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy import sparse
from sklearn.pipeline import Pipeline

from config import (
    OUTPUT_DIR,
    PLOT_DPI,
    RANDOM_STATE,
    SAMPLE_POSITION,
    SAVE_PDF,
    SAVE_PNG,
    SHAP_BACKGROUND_SIZE,
    SHAP_CHECK_ADDITIVITY,
    SHAP_CONTRIBUTIONS_PATH,
    SHAP_FEATURE_PERTURBATION,
    SHAP_FORCE_PDF_PATH,
    SHAP_FORCE_PNG_PATH,
    SHAP_MODEL_OUTPUT,
    SHAP_PREDICTION_SUMMARY_PATH,
    SHAP_USE_FULL_TRAINING_BACKGROUND,
    create_output_directory,
)


@dataclass
class ShapForceResult:
    """Store the result of a local SHAP explanation."""

    sample_position: int
    sample_index: Any
    predicted_class: Any
    explained_class: Any
    explained_probability: float
    base_value: float
    reconstructed_output: float
    additivity_error: float
    feature_names: list[str]
    feature_values: np.ndarray
    shap_values: np.ndarray
    contributions: pd.DataFrame
    prediction_summary: pd.DataFrame


def _validate_inputs(
    model: Any,
    X_background: pd.DataFrame,
    X_test: pd.DataFrame,
    sample_position: int,
) -> None:
    """Validate the model and datasets used by SHAP."""

    if not hasattr(model, "predict"):
        raise TypeError(
            "The model must provide a predict() method."
        )

    if not hasattr(model, "predict_proba"):
        raise TypeError(
            "The model must provide a predict_proba() method."
        )

    if not hasattr(model, "classes_"):
        raise TypeError(
            "The trained classifier must provide classes_."
        )

    if not isinstance(X_background, pd.DataFrame):
        raise TypeError(
            "X_background must be a pandas DataFrame."
        )

    if not isinstance(X_test, pd.DataFrame):
        raise TypeError(
            "X_test must be a pandas DataFrame."
        )

    if X_background.empty:
        raise ValueError("X_background cannot be empty.")

    if X_test.empty:
        raise ValueError("X_test cannot be empty.")

    if list(X_background.columns) != list(X_test.columns):
        raise ValueError(
            "X_background and X_test must have the same "
            "features in the same order."
        )

    if sample_position < 0 or sample_position >= len(X_test):
        raise IndexError(
            "sample_position is outside the valid test-set range."
        )


def _to_dataframe(
    values: Any,
    feature_names: list[str],
    index: pd.Index,
) -> pd.DataFrame:
    """Convert transformed values into a pandas DataFrame."""

    if sparse.issparse(values):
        values = values.toarray()

    values = np.asarray(values)

    if values.ndim != 2:
        raise ValueError(
            "The transformed feature matrix must be two-dimensional."
        )

    if values.shape[1] != len(feature_names):
        raise ValueError(
            "The number of transformed features does not match "
            "the number of feature names."
        )

    return pd.DataFrame(
        values,
        columns=feature_names,
        index=index,
    )


def _prepare_shap_inputs(
    model: Any,
    X_background: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[Any, pd.DataFrame, pd.DataFrame]:
    """
    Prepare the estimator and feature matrices used by SHAP.

    When the saved object is a scikit-learn Pipeline, all
    preprocessing steps are applied before passing the final
    estimator to TreeExplainer.
    """

    if not isinstance(model, Pipeline) or len(model.steps) == 1:
        return (
            model,
            X_background.copy(),
            X_test.copy(),
        )

    # All pipeline steps except the final estimator
    transformer = model[:-1]

    # Final predictive estimator
    shap_model = model.steps[-1][1]

    transformed_background = transformer.transform(
        X_background
    )

    transformed_test = transformer.transform(
        X_test
    )

    try:
        feature_names = list(
            transformer.get_feature_names_out(
                X_background.columns
            )
        )

    except (AttributeError, TypeError, ValueError):
        transformed_array = transformed_background

        if sparse.issparse(transformed_array):
            transformed_array = transformed_array.toarray()

        number_of_features = np.asarray(
            transformed_array
        ).shape[1]

        if number_of_features == X_background.shape[1]:
            feature_names = X_background.columns.tolist()

        else:
            feature_names = [
                f"feature_{position}"
                for position in range(number_of_features)
            ]

    background_df = _to_dataframe(
        values=transformed_background,
        feature_names=feature_names,
        index=X_background.index,
    )

    test_df = _to_dataframe(
        values=transformed_test,
        feature_names=feature_names,
        index=X_test.index,
    )

    return shap_model, background_df, test_df


def _select_background(
    X_background: pd.DataFrame,
) -> pd.DataFrame:
    """Select the reference observations used by SHAP."""

    if SHAP_USE_FULL_TRAINING_BACKGROUND:
        return X_background

    background_size = min(
        SHAP_BACKGROUND_SIZE,
        len(X_background),
    )

    return X_background.sample(
        n=background_size,
        random_state=RANDOM_STATE,
    )


def _resolve_class_to_explain(
    model: Any,
    sample_df: pd.DataFrame,
    class_to_explain: Any | None,
) -> tuple[Any, Any, int, float]:
    """Select the model output that must be explained."""

    classes = np.asarray(model.classes_)

    predicted_class = model.predict(sample_df)[0]

    if class_to_explain is None:
        explained_class = predicted_class
    else:
        explained_class = class_to_explain

    class_matches = np.flatnonzero(
        classes == explained_class
    )

    if len(class_matches) == 0:
        raise ValueError(
            f"Class {explained_class!r} is not contained in "
            f"model.classes_: {classes.tolist()}"
        )

    class_position = int(class_matches[0])

    probabilities = model.predict_proba(sample_df)[0]

    explained_probability = float(
        probabilities[class_position]
    )

    return (
        predicted_class,
        explained_class,
        class_position,
        explained_probability,
    )


def _extract_base_value(
    base_values: Any,
    class_position: int,
) -> float:
    """Extract the baseline from a multiclass SHAP result."""

    base_array = np.asarray(base_values)

    if base_array.ndim == 0:
        return float(base_array)

    if base_array.ndim == 1:
        if base_array.size == 1:
            return float(base_array[0])

        return float(base_array[class_position])

    return float(base_array[0, class_position])


def _extract_shap_output(
    explanation: shap.Explanation,
    class_position: int,
    number_of_classes: int,
) -> tuple[float, np.ndarray]:
    """Extract SHAP values for the selected classifier output."""

    values = np.asarray(explanation.values)

    # Multiclass format:
    # samples × features × classes
    if values.ndim == 3:
        if class_position >= values.shape[2]:
            raise IndexError(
                "The selected class is not available in the "
                "SHAP output."
            )

        base_value = _extract_base_value(
            explanation.base_values,
            class_position,
        )

        contributions = values[
            0,
            :,
            class_position,
        ]

        return base_value, contributions

    # Single-output format:
    # samples × features
    if values.ndim == 2:
        base_array = np.asarray(
            explanation.base_values
        )

        if base_array.ndim == 0:
            base_value = float(base_array)

        elif base_array.ndim == 1:
            base_value = float(base_array[0])

        else:
            base_value = float(base_array[0, 0])

        contributions = values[0].copy()

        # Some binary classifiers return only the explanation
        # for the positive class, which is classes_[1].
        if number_of_classes == 2 and class_position == 0:
            if SHAP_MODEL_OUTPUT != "probability":
                raise ValueError(
                    "A single-output binary SHAP explanation "
                    "can be inverted only on the probability scale."
                )

            base_value = 1.0 - base_value
            contributions = -contributions

        elif number_of_classes > 2:
            raise ValueError(
                "SHAP returned only one output for a multiclass "
                "classifier."
            )

        return base_value, contributions

    raise ValueError(
        "Unexpected SHAP output shape: "
        f"{values.shape}"
    )


def compute_shap_force_explanation(
    model: Any,
    X_background: pd.DataFrame,
    X_test: pd.DataFrame,
    *,
    sample_position: int = SAMPLE_POSITION,
    class_to_explain: Any | None = None,
) -> ShapForceResult:
    """
    Compute a local SHAP explanation for one test observation.

    Parameters
    ----------
    model:
        Trained tree-based classifier or trained pipeline whose
        final estimator is tree-based.

    X_background:
        Training features used as the SHAP reference population.

    X_test:
        Unseen test features containing the observation to explain.

    sample_position:
        Integer position of the test observation to explain.

    class_to_explain:
        Class whose probability must be explained. When omitted,
        the function explains the class predicted by the model.

    Returns
    -------
    ShapForceResult
        Local explanation, feature contributions and prediction
        summary.
    """

    _validate_inputs(
        model=model,
        X_background=X_background,
        X_test=X_test,
        sample_position=sample_position,
    )

    original_sample_df = X_test.iloc[
        [sample_position]
    ]

    (
        predicted_class,
        explained_class,
        class_position,
        explained_probability,
    ) = _resolve_class_to_explain(
        model=model,
        sample_df=original_sample_df,
        class_to_explain=class_to_explain,
    )

    (
        shap_model,
        transformed_background,
        transformed_test,
    ) = _prepare_shap_inputs(
        model=model,
        X_background=X_background,
        X_test=X_test,
    )

    background_df = _select_background(
        transformed_background
    )

    transformed_sample_df = transformed_test.iloc[
        [sample_position]
    ]

    try:
        explainer = shap.TreeExplainer(
            shap_model,
            data=background_df,
            feature_perturbation=SHAP_FEATURE_PERTURBATION,
            model_output=SHAP_MODEL_OUTPUT,
        )

        explanation = explainer(
            transformed_sample_df,
            check_additivity=SHAP_CHECK_ADDITIVITY,
        )

    except Exception as error:
        raise RuntimeError(
            "TreeExplainer could not explain the final estimator. "
            "This module requires a tree-based model such as a "
            "Decision Tree, Random Forest or Gradient Boosting "
            "model."
        ) from error

    base_value, local_shap_values = _extract_shap_output(
        explanation=explanation,
        class_position=class_position,
        number_of_classes=len(model.classes_),
    )

    feature_names = transformed_sample_df.columns.tolist()

    feature_values = transformed_sample_df.iloc[
        0
    ].to_numpy()

    reconstructed_output = float(
        base_value + local_shap_values.sum()
    )

    additivity_error = float(
        abs(
            reconstructed_output
            - explained_probability
        )
    )

    contributions_df = pd.DataFrame({
        "feature": feature_names,
        "feature_value": feature_values,
        "shap_value": local_shap_values,
    })

    contributions_df["absolute_shap_value"] = (
        contributions_df["shap_value"].abs()
    )

    contributions_df["direction"] = np.select(
        [
            contributions_df["shap_value"] > 0,
            contributions_df["shap_value"] < 0,
        ],
        [
            "increases_probability",
            "decreases_probability",
        ],
        default="no_effect",
    )

    contributions_df["rank"] = (
        contributions_df["absolute_shap_value"]
        .rank(
            method="dense",
            ascending=False,
        )
        .astype(int)
    )

    contributions_df = (
        contributions_df
        .sort_values(
            by="absolute_shap_value",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    sample_index = X_test.index[
        sample_position
    ]

    prediction_summary_df = pd.DataFrame({
        "sample_position": [sample_position],
        "sample_index": [sample_index],
        "predicted_class": [predicted_class],
        "explained_class": [explained_class],
        "explained_class_probability": [
            explained_probability
        ],
        "base_value": [base_value],
        "sum_of_shap_values": [
            float(local_shap_values.sum())
        ],
        "reconstructed_output": [
            reconstructed_output
        ],
        "absolute_additivity_error": [
            additivity_error
        ],
        "model_output": [SHAP_MODEL_OUTPUT],
    })

    return ShapForceResult(
        sample_position=sample_position,
        sample_index=sample_index,
        predicted_class=predicted_class,
        explained_class=explained_class,
        explained_probability=explained_probability,
        base_value=base_value,
        reconstructed_output=reconstructed_output,
        additivity_error=additivity_error,
        feature_names=feature_names,
        feature_values=feature_values,
        shap_values=local_shap_values,
        contributions=contributions_df,
        prediction_summary=prediction_summary_df,
    )


def _resolve_output_paths(
    sample_position: int,
) -> tuple[Path, Path, Path, Path]:
    """Return the output paths for the selected observation."""

    if sample_position == SAMPLE_POSITION:
        return (
            SHAP_CONTRIBUTIONS_PATH,
            SHAP_PREDICTION_SUMMARY_PATH,
            SHAP_FORCE_PNG_PATH,
            SHAP_FORCE_PDF_PATH,
        )

    filename_suffix = f"sample_{sample_position:04d}"

    return (
        OUTPUT_DIR
        / f"shap_local_contributions_{filename_suffix}.csv",
        OUTPUT_DIR
        / f"shap_prediction_summary_{filename_suffix}.csv",
        OUTPUT_DIR
        / f"shap_force_{filename_suffix}.png",
        OUTPUT_DIR
        / f"shap_force_{filename_suffix}.pdf",
    )


def save_shap_tables(
    result: ShapForceResult,
) -> tuple[Path, Path]:
    """Save the SHAP contributions and prediction summary."""

    (
        contributions_path,
        summary_path,
        _,
        _,
    ) = _resolve_output_paths(
        result.sample_position
    )

    contributions_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.contributions.to_csv(
        contributions_path,
        index=False,
    )

    result.prediction_summary.to_csv(
        summary_path,
        index=False,
    )

    return contributions_path, summary_path


def save_shap_force_plot(
    result: ShapForceResult,
) -> tuple[Path | None, Path | None]:
    """Create and save the local SHAP force plot."""

    (
        _,
        _,
        png_path,
        pdf_path,
    ) = _resolve_output_paths(
        result.sample_position
    )

    shap.force_plot(
        base_value=result.base_value,
        shap_values=result.shap_values,
        features=result.feature_values,
        feature_names=result.feature_names,
        matplotlib=True,
        show=False,
    )

    figure = plt.gcf()

    figure.set_size_inches(
        14,
        3.5,
    )

    plt.title(
        (
            "Local SHAP Explanation — "
            f"Class: {result.explained_class}"
        ),
        pad=25,
    )

    saved_png_path: Path | None = None
    saved_pdf_path: Path | None = None

    if SAVE_PNG:
        png_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        figure.savefig(
            png_path,
            dpi=PLOT_DPI,
            bbox_inches="tight",
        )

        saved_png_path = png_path

    if SAVE_PDF:
        pdf_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        figure.savefig(
            pdf_path,
            bbox_inches="tight",
        )

        saved_pdf_path = pdf_path

    plt.close(figure)

    return saved_png_path, saved_pdf_path


def print_shap_force_summary(
    result: ShapForceResult,
) -> None:
    """Print a readable summary of the local explanation."""

    print("\nLOCAL EXPLAINABILITY — SHAP FORCE PLOT")

    print(
        f"Sample position: {result.sample_position}"
    )

    print(
        f"Sample index: {result.sample_index}"
    )

    print(
        f"Predicted class: {result.predicted_class}"
    )

    print(
        f"Explained class: {result.explained_class}"
    )

    print(
        "Explained class probability: "
        f"{result.explained_probability:.6f}"
    )

    print(
        f"SHAP baseline: {result.base_value:.6f}"
    )

    print(
        "Sum of SHAP contributions: "
        f"{result.shap_values.sum():.6f}"
    )

    print(
        "Baseline + SHAP contributions: "
        f"{result.reconstructed_output:.6f}"
    )

    print(
        "Absolute additivity error: "
        f"{result.additivity_error:.8f}"
    )

    print("\nLargest local contributions:")

    columns_to_print = [
        "rank",
        "feature",
        "feature_value",
        "shap_value",
        "direction",
    ]

    print(
        result.contributions[
            columns_to_print
        ].to_string(
            index=False
        )
    )


def run_shap_force(
    model: Any,
    X_background: pd.DataFrame,
    X_test: pd.DataFrame,
    *,
    sample_position: int = SAMPLE_POSITION,
    class_to_explain: Any | None = None,
) -> ShapForceResult:
    """
    Execute the complete local SHAP explanation workflow.

    The function:
    1. creates the output directory;
    2. computes the local SHAP explanation;
    3. saves the contribution and prediction CSV files;
    4. creates and saves the force plot;
    5. prints a readable summary.
    """

    create_output_directory()

    result = compute_shap_force_explanation(
        model=model,
        X_background=X_background,
        X_test=X_test,
        sample_position=sample_position,
        class_to_explain=class_to_explain,
    )

    contributions_path, summary_path = save_shap_tables(
        result
    )

    png_path, pdf_path = save_shap_force_plot(
        result
    )

    print_shap_force_summary(
        result
    )

    print(
        "\nSHAP contributions saved to:"
        f"\n{contributions_path}"
    )

    print(
        "\nSHAP prediction summary saved to:"
        f"\n{summary_path}"
    )

    if png_path is not None:
        print(
            "\nSHAP force plot PNG saved to:"
            f"\n{png_path}"
        )

    if pdf_path is not None:
        print(
            "\nSHAP force plot PDF saved to:"
            f"\n{pdf_path}"
        )

    return result