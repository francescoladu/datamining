from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve


DEFAULT_DPI = 300


def _save_figure(
    figure: plt.Figure,
    output_pdf_path: str | Path,
) -> Path:
    """Save the report-ready vector PDF without a redundant PNG copy."""
    pdf_path = Path(output_pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    figure.savefig(
        pdf_path,
        format="pdf",
        bbox_inches="tight",
    )

    return pdf_path


def plot_nested_cv_comparison(
    nested_scores: pd.DataFrame,
    output_pdf_path: str | Path,
    model_order: tuple[str, str] = (
        "Decision Tree",
        "Random Forest",
    ),
) -> None:
    """
    Compare model families using the outer-fold macro F1-scores.

    The boxplots summarize the score distributions while the overlaid points
    preserve the individual outer-fold results. This plot is intended for the
    report's model-comparison figure.
    """
    required_columns = {"model", "outer_fold", "macro_f1"}
    missing_columns = required_columns.difference(nested_scores.columns)
    if nested_scores.empty or missing_columns:
        raise ValueError(
            "Invalid nested_scores DataFrame. "
            f"Missing columns: {sorted(missing_columns)}"
        )

    boxplot_values = [
        (
            nested_scores.loc[
                nested_scores["model"] == model_name,
                "macro_f1",
            ]
            .astype(float)
            .to_numpy()
        )
        for model_name in model_order
    ]

    if any(len(values) == 0 for values in boxplot_values):
        raise ValueError("Every model in model_order must have at least one score.")

    figure, axis = plt.subplots(figsize=(6.4, 4.4))

    axis.boxplot(
        boxplot_values,
        widths=0.42,
        showmeans=False,
        showfliers=False,
    )

    for position, values in enumerate(boxplot_values, start=1):
        # Deterministic small horizontal offsets avoid hiding overlapping dots.
        offsets = np.linspace(-0.065, 0.065, num=len(values))
        axis.scatter(
            np.full(len(values), position, dtype=float) + offsets,
            values,
            s=34,
            zorder=3,
            alpha=0.85,
        )

    all_values = np.concatenate(boxplot_values)
    score_range = float(all_values.max() - all_values.min())
    margin = max(0.004, score_range * 0.15)

    model_labels = [
        (
            f"{model_name}\n"
            f"mean {np.mean(values):.3f} ± {np.std(values, ddof=1):.3f}"
        )
        for model_name, values in zip(model_order, boxplot_values)
    ]
    axis.set_xticks(
        np.arange(1, len(model_order) + 1),
        model_labels,
    )
    axis.set_ylabel("Outer-fold macro F1-score")
    axis.set_xlabel("")
    axis.set_ylim(
        float(all_values.min() - margin),
        float(all_values.max() + margin),
    )
    axis.grid(axis="y", alpha=0.22)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    figure.tight_layout()
    _save_figure(figure, output_pdf_path)
    plt.close(figure)


def plot_selected_feature_ranking(
    selected_features: pd.DataFrame,
    output_pdf_path: str | Path,
    *,
    max_display: int = 15,
) -> None:
    """
    Plot the Mutual Information ranking fitted on the full development set.

    Features selected by the final SelectKBest step are marked with a point at
    the end of their bar. The figure is suitable for the report's feature-
    selection section.
    """
    required_columns = {
        "feature",
        "mutual_information_score",
        "selected",
    }
    missing_columns = required_columns.difference(selected_features.columns)
    if selected_features.empty or missing_columns:
        raise ValueError(
            "Invalid selected_features DataFrame. "
            f"Missing columns: {sorted(missing_columns)}"
        )
    if max_display <= 0:
        raise ValueError("max_display must be positive.")

    plot_df = (
        selected_features
        .copy()
        .sort_values("mutual_information_score", ascending=False)
        .head(max_display)
        .sort_values("mutual_information_score", ascending=True)
    )

    figure_height = max(4.8, 0.34 * len(plot_df) + 1.2)
    figure, axis = plt.subplots(figsize=(7.2, figure_height))

    y_positions = np.arange(len(plot_df))
    scores = plot_df["mutual_information_score"].astype(float).to_numpy()

    axis.barh(
        y_positions,
        scores,
        alpha=0.85,
    )

    selected_mask = plot_df["selected"].astype(bool).to_numpy()
    if selected_mask.any():
        axis.scatter(
            scores[selected_mask],
            y_positions[selected_mask],
            s=34,
            zorder=3,
            label="Selected by final pipeline",
        )
        axis.legend(frameon=False, loc="lower right")

    axis.set_yticks(y_positions, plot_df["feature"].astype(str).tolist())
    axis.set_xlabel("Mutual Information score")
    axis.set_ylabel("")
    axis.grid(axis="x", alpha=0.22)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    figure.tight_layout()
    _save_figure(figure, output_pdf_path)
    plt.close(figure)


def plot_hyperparameter_optimization(
    search_results: pd.DataFrame,
    output_pdf_path: str | Path,
    *,
    model_name: str,
    max_candidates: int = 15,
) -> None:
    """
    Plot the best SearchCV candidates ranked by mean inner-CV macro F1.

    Error bars show the standard deviation across inner folds. Candidate labels
    also report k when feature_selection__k is part of the search space.
    """
    required_columns = {
        "candidate_id",
        "rank_test_score",
        "mean_test_score",
        "std_test_score",
    }
    missing_columns = required_columns.difference(search_results.columns)
    if search_results.empty or missing_columns:
        raise ValueError(
            "Invalid search_results DataFrame. "
            f"Missing columns: {sorted(missing_columns)}"
        )
    if max_candidates <= 0:
        raise ValueError("max_candidates must be positive.")

    plot_df = (
        search_results
        .copy()
        .sort_values(
            ["rank_test_score", "mean_test_score"],
            ascending=[True, False],
        )
        .head(max_candidates)
        .sort_values("mean_test_score", ascending=True)
    )

    k_column = "param_feature_selection__k"
    labels: list[str] = []
    for _, row in plot_df.iterrows():
        label = f"candidate #{int(row['candidate_id'])}"
        if k_column in plot_df.columns:
            label += f"  |  k={row[k_column]}"
        labels.append(label)

    mean_scores = plot_df["mean_test_score"].astype(float).to_numpy()
    std_scores = plot_df["std_test_score"].astype(float).to_numpy()
    y_positions = np.arange(len(plot_df))

    figure_height = max(4.8, 0.34 * len(plot_df) + 1.5)
    figure, axis = plt.subplots(figsize=(7.8, figure_height))

    axis.errorbar(
        mean_scores,
        y_positions,
        xerr=std_scores,
        fmt="o",
        capsize=3,
    )

    best_row = plot_df.loc[plot_df["rank_test_score"].astype(int).idxmin()]
    best_score = float(best_row["mean_test_score"])
    axis.axvline(
        best_score,
        linestyle="--",
        linewidth=1.0,
        alpha=0.7,
    )

    axis.set_yticks(y_positions, labels)
    axis.set_xlabel("Mean inner-CV macro F1-score")
    axis.set_ylabel("")
    axis.set_title(f"{model_name} hyperparameter search — top candidates")
    axis.grid(axis="x", alpha=0.22)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    figure.tight_layout()
    _save_figure(figure, output_pdf_path)
    plt.close(figure)


def plot_final_test_confusion_matrix(
    final_test_predictions: pd.DataFrame,
    output_pdf_path: str | Path,
) -> None:
    """Plot the final held-out test confusion matrix using phishing as positive."""
    required_columns = {"y_true", "y_pred"}
    missing_columns = required_columns.difference(final_test_predictions.columns)
    if final_test_predictions.empty or missing_columns:
        raise ValueError(
            "Invalid final_test_predictions DataFrame. "
            f"Missing columns: {sorted(missing_columns)}"
        )

    labels = [-1, 1]
    matrix = confusion_matrix(
        final_test_predictions["y_true"],
        final_test_predictions["y_pred"],
        labels=labels,
    )

    row_totals = matrix.sum(axis=1, keepdims=True)
    row_percentages = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype=float),
        where=row_totals != 0,
    )

    figure, axis = plt.subplots(figsize=(5.4, 4.8))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Count")

    display_labels = ["Phishing (-1)", "Legitimate (1)"]
    axis.set_xticks([0, 1], display_labels)
    axis.set_yticks([0, 1], display_labels)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")

    threshold = float(matrix.max()) / 2.0 if matrix.size else 0.0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            count = int(matrix[row, column])
            percentage = row_percentages[row, column] * 100.0
            axis.text(
                column,
                row,
                f"{count}\n{percentage:.1f}%",
                ha="center",
                va="center",
                color="white" if count > threshold else "black",
                fontsize=10,
            )

    figure.tight_layout()
    _save_figure(figure, output_pdf_path)
    plt.close(figure)


def plot_final_test_roc_curve(
    final_test_predictions: pd.DataFrame,
    output_pdf_path: str | Path,
) -> float:
    """
    Plot the final held-out test ROC curve and return its AUC.

    Phishing (-1) is treated as the positive class.
    """
    required_columns = {"y_true", "phishing_probability"}
    missing_columns = required_columns.difference(final_test_predictions.columns)
    if final_test_predictions.empty or missing_columns:
        raise ValueError(
            "Invalid final_test_predictions DataFrame. "
            f"Missing columns: {sorted(missing_columns)}"
        )

    y_true_binary = (
        final_test_predictions["y_true"].to_numpy() == -1
    ).astype(int)
    probabilities = (
        final_test_predictions["phishing_probability"]
        .astype(float)
        .to_numpy()
    )

    false_positive_rate, true_positive_rate, _ = roc_curve(
        y_true_binary,
        probabilities,
    )
    auc_value = float(
        roc_auc_score(y_true_binary, probabilities)
    )

    figure, axis = plt.subplots(figsize=(5.6, 4.8))
    axis.plot(
        false_positive_rate,
        true_positive_rate,
        linewidth=1.8,
        label=f"Final model (AUC = {auc_value:.3f})",
    )
    axis.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        linestyle="--",
        linewidth=1.0,
        label="Random classifier",
    )

    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.02)
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate (phishing recall)")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False, loc="lower right")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    figure.tight_layout()
    _save_figure(figure, output_pdf_path)
    plt.close(figure)

    return auc_value
