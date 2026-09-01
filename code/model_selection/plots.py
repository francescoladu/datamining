from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import seaborn as sns

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
    Plot outer-fold macro F1-scores using a standard Tukey boxplot.

    Boxes represent Q1--Q3, the central line is the median,
    whiskers extend to 1.5 * IQR, and observations outside
    the whiskers are shown as outliers.
    """
    required_columns = {
        "model",
        "outer_fold",
        "macro_f1",
    }

    missing_columns = required_columns.difference(
        nested_scores.columns
    )

    if nested_scores.empty or missing_columns:
        raise ValueError(
            "Invalid nested_scores DataFrame. "
            f"Missing columns: {sorted(missing_columns)}"
        )

    plot_df = nested_scores.loc[
        nested_scores["model"].isin(model_order),
        ["model", "outer_fold", "macro_f1"],
    ].copy()

    plot_df["macro_f1"] = (
        plot_df["macro_f1"]
        .astype(float)
    )

    for model_name in model_order:
        if not (
            plot_df["model"] == model_name
        ).any():
            raise ValueError(
                f"No scores found for model: {model_name}"
            )

    model_palette = {
        "Decision Tree": "#4C78A8",
        "Random Forest": "#F58518",
    }

    figure, axis = plt.subplots(
        figsize=(6.4, 4.3)
    )

    sns.boxplot(
        data=plot_df,
        x="model",
        y="macro_f1",
        hue="model",
        order=list(model_order),
        hue_order=list(model_order),
        palette=model_palette,

        width=0.42,

        # Standard Tukey whiskers
        whis=1.5,

        showmeans=False,

        # Show observations outside 1.5 * IQR
        showfliers=True,

        saturation=0.85,
        linewidth=1.4,

        boxprops={
            "edgecolor": "black",
        },

        whiskerprops={
            "color": "black",
            "linewidth": 1.3,
        },

        capprops={
            "color": "black",
            "linewidth": 1.3,
        },

        medianprops={
            "color": "black",
            "linewidth": 2.0,
        },

        flierprops={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markeredgewidth": 1.2,
            "markersize": 5.5,
            "linestyle": "none",
        },

        legend=False,
        ax=axis,
    )

    all_values = (
        plot_df["macro_f1"]
        .to_numpy(dtype=float)
    )

    score_range = float(
        all_values.max() - all_values.min()
    )

    margin = max(
        0.004,
        score_range * 0.08,
    )

    axis.set_ylim(
        float(all_values.min() - margin),
        float(all_values.max() + margin),
    )

    axis.set_xlabel("")

    axis.set_ylabel(
        "Outer-fold macro F1-score"
    )

    axis.grid(
        axis="y",
        alpha=0.16,
        linewidth=0.7,
    )

    axis.grid(
        axis="x",
        visible=False,
    )

    axis.set_axisbelow(True)

    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    figure.tight_layout()

    _save_figure(
        figure,
        output_pdf_path,
    )

    plt.close(figure)


def plot_selected_feature_ranking(
    selected_features: pd.DataFrame,
    output_pdf_path: str | Path,
    *,
    max_display: int = 10,
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
    Plot a 2D hyperparameter-search heatmap.

    Each cell reports the best mean inner-CV macro F1-score obtained
    among the sampled candidates sharing the corresponding values of
    feature_selection__k and classifier__max_depth.
    """
    required_columns = {
        "mean_test_score",
        "param_feature_selection__k",
        "param_classifier__max_depth",
    }

    missing_columns = required_columns.difference(
        search_results.columns
    )

    if search_results.empty or missing_columns:
        raise ValueError(
            "Invalid search_results DataFrame. "
            f"Missing columns: {sorted(missing_columns)}"
        )

    plot_df = search_results.copy()

    # --------------------------------------------------------
    # Clean k values
    # --------------------------------------------------------

    plot_df["k"] = (
        plot_df["param_feature_selection__k"]
        .astype(str)
    )

    # --------------------------------------------------------
    # Clean max_depth values
    # NaN corresponds to max_depth=None
    # --------------------------------------------------------

    plot_df["max_depth"] = (
        plot_df["param_classifier__max_depth"]
        .apply(
            lambda value: (
                "None"
                if pd.isna(value)
                else str(int(float(value)))
            )
        )
    )

    # --------------------------------------------------------
    # For each (k, max_depth) combination, keep the best
    # mean inner-CV score found by the randomized search.
    # --------------------------------------------------------

    heatmap_data = (
        plot_df
        .groupby(
            ["k", "max_depth"],
            observed=True,
        )["mean_test_score"]
        .max()
        .unstack("max_depth")
    )

    # --------------------------------------------------------
    # Sort k values numerically
    # --------------------------------------------------------

    k_order = sorted(
        heatmap_data.index,
        key=lambda value: (
            int(value)
            if value.isdigit()
            else float("inf")
        ),
    )

    # --------------------------------------------------------
    # Sort max_depth numerically, with None at the end
    # --------------------------------------------------------

    depth_values = [
        value
        for value in heatmap_data.columns
        if value != "None"
    ]

    depth_order = sorted(
        depth_values,
        key=int,
    )

    if "None" in heatmap_data.columns:
        depth_order.append("None")

    heatmap_data = heatmap_data.reindex(
        index=k_order,
        columns=depth_order,
    )

    values = heatmap_data.to_numpy(
        dtype=float
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    figure, axis = plt.subplots(
        figsize=(7.2, 5.1)
    )

    image = axis.imshow(
        values,
        aspect="auto",
        cmap="viridis",
    )

    # --------------------------------------------------------
    # Annotate cells
    # --------------------------------------------------------

    finite_values = values[
        np.isfinite(values)
    ]

    if finite_values.size > 0:
        threshold = (
            float(finite_values.min())
            + float(finite_values.max())
        ) / 2.0
    else:
        threshold = 0.0

    for row_index in range(
        values.shape[0]
    ):
        for column_index in range(
            values.shape[1]
        ):
            score = values[
                row_index,
                column_index,
            ]

            if not np.isfinite(score):
                continue

            text_color = (
                "white"
                if score < threshold
                else "black"
            )

            axis.text(
                column_index,
                row_index,
                f"{score:.3f}",
                ha="center",
                va="center",
                fontsize=9,
                color=text_color,
            )

    # --------------------------------------------------------
    # Axes
    # --------------------------------------------------------

    axis.set_xticks(
        np.arange(
            len(depth_order)
        )
    )

    axis.set_xticklabels(
        depth_order
    )

    axis.set_yticks(
        np.arange(
            len(k_order)
        )
    )

    axis.set_yticklabels(
        k_order
    )

    axis.set_xlabel(
        "Random Forest max depth"
    )

    axis.set_ylabel(
        "Number of selected features (k)"
    )

    axis.set_title(
        f"{model_name} hyperparameter optimization"
    )

    # --------------------------------------------------------
    # Color bar
    # --------------------------------------------------------

    colorbar = figure.colorbar(
        image,
        ax=axis,
        pad=0.03,
    )

    colorbar.set_label(
        "Best mean inner-CV macro F1-score"
    )

    figure.tight_layout()

    _save_figure(
        figure,
        output_pdf_path,
    )

    plt.close(
        figure
    )


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
