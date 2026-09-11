from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_feature_histograms_by_class(
    data: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
    output_dir: Path,
    weight_column: str | None = None,
    class_label_names: Mapping[object, str] | None = None,
    *,
    features_per_figure: int = 10,
    columns_per_figure: int = 2,
) -> list[Path]:
    """
    Plot compact class-conditional distributions for all features.

    If weight_column is provided, frequencies reflect retained weighted support mass.
    """
    if target_column not in data.columns:
        raise ValueError(f"Missing target column: {target_column}")

    feature_columns = list(feature_columns)
    if not feature_columns:
        raise ValueError("No feature columns were provided.")

    missing_features = [f for f in feature_columns if f not in data.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")

    output_dir.mkdir(parents=True, exist_ok=True)

    stale_patterns = (
        "*_distribution_by_class.pdf",
        "*_distribution_by_class.png",
        "feature_distributions_*.pdf",
        "feature_distributions_*.png",
    )
    for pattern in stale_patterns:
        for stale_path in output_dir.glob(pattern):
            stale_path.unlink()

    class_labels = sorted(data[target_column].dropna().unique().tolist())
    class_names = {
        label: (
            class_label_names.get(label, str(label))
            if class_label_names is not None
            else str(label)
        )
        for label in class_labels
    }

    generated_paths: list[Path] = []
    has_weights = weight_column is not None and weight_column in data.columns

    for group_index, group_start in enumerate(
        range(0, len(feature_columns), features_per_figure),
        start=1,
    ):
        group_features = feature_columns[
            group_start : group_start + features_per_figure
        ]

        rows_per_figure = int(
            np.ceil(len(group_features) / columns_per_figure)
        )
        figure_height = max(5.0, 2.4 * rows_per_figure)

        figure, axes = plt.subplots(
            nrows=rows_per_figure,
            ncols=columns_per_figure,
            figsize=(10.5, figure_height),
            squeeze=False,
            sharey=True,
        )

        axes = axes.ravel()
        legend_handles = None
        legend_labels = None

        for feature_index, feature in enumerate(group_features):
            axis = axes[feature_index]

            cols = [feature, target_column]
            if has_weights:
                cols.append(weight_column)

            subset = data[cols].dropna()

            if has_weights:
                distribution = (
                    subset.groupby(
                        [target_column, feature],
                        observed=True,
                    )[weight_column]
                    .sum()
                    .rename("count")
                    .reset_index()
                )
            else:
                distribution = (
                    subset.groupby(
                        [target_column, feature],
                        observed=True,
                    )
                    .size()
                    .rename("count")
                    .reset_index()
                )

            class_totals = distribution.groupby(
                target_column,
                observed=True,
            )["count"].transform("sum")

            distribution["percentage"] = (
                100.0 * distribution["count"] / class_totals
            )

            distribution["class_name"] = distribution[
                target_column
            ].map(class_names)

            sns.barplot(
                data=distribution,
                x=feature,
                y="percentage",
                hue="class_name",
                hue_order=[
                    class_names[label]
                    for label in class_labels
                ],
                errorbar=None,
                saturation=0.85,
                edgecolor="black",
                linewidth=0.4,
                ax=axis,
            )

            if legend_handles is None:
                legend_handles, legend_labels = (
                    axis.get_legend_handles_labels()
                )

            if axis.get_legend() is not None:
                axis.get_legend().remove()

            axis.set_title(
                feature,
                fontsize=10,
                fontweight="bold",
                pad=6,
            )

            axis.set_xlabel(
                "Feature value",
                fontweight="bold",
            )

            axis.set_ylim(0, 100)
            axis.grid(
                axis="y",
                alpha=0.15,
                linewidth=0.7,
            )
            axis.set_axisbelow(True)

            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)

            if feature_index % columns_per_figure == 0:
                axis.set_ylabel(
                    "Within-class observations (%)",
                    fontweight="bold",
                )
            else:
                axis.set_ylabel("")

            # Tick labels in bold.
            for tick_label in (
                list(axis.get_xticklabels())
                + list(axis.get_yticklabels())
            ):
                tick_label.set_fontweight("bold")

        for unused_axis in axes[len(group_features):]:
            unused_axis.set_visible(False)

        if legend_handles is not None and legend_labels is not None:
            legend = figure.legend(
                legend_handles,
                legend_labels,
                loc="upper center",
                ncol=len(class_labels),
                frameon=False,
                bbox_to_anchor=(0.5, 0.965),
            )

            for legend_text in legend.get_texts():
                legend_text.set_fontweight("bold")

        first_feature = group_start + 1
        last_feature = group_start + len(group_features)

        figure.suptitle(
            (
                "Feature distributions by class "
                f"({first_feature}–{last_feature})"
            ),
            fontsize=13,
            fontweight="bold",
            y=0.995,
        )

        figure.tight_layout(
            rect=[0.0, 0.0, 1.0, 0.93]
        )

        file_stem = f"feature_distributions_{group_index:02d}"

        pdf_path = output_dir / f"{file_stem}.pdf"
        png_path = output_dir / f"{file_stem}.png"

        figure.savefig(
            pdf_path,
            format="pdf",
            bbox_inches="tight",
        )

        figure.savefig(
            png_path,
            format="png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(figure)
        generated_paths.append(pdf_path)

    return generated_paths


def select_top_correlated_features(
    correlation_matrix: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Select features according to their strongest absolute
    feature-feature Spearman correlation.

    For each feature:
    1. Ignore the diagonal correlation with itself.
    2. Find its maximum absolute correlation with another feature.
    3. Sort features by this value in descending order.
    4. Keep only the first top_n features.
    """
    if correlation_matrix.empty:
        raise ValueError(
            "The correlation matrix is empty."
        )

    if top_n <= 0:
        raise ValueError(
            "top_n must be greater than zero."
        )

    feature_names = [
        feature
        for feature in correlation_matrix.columns
        if feature in correlation_matrix.index
    ]

    if not feature_names:
        raise ValueError(
            "The correlation matrix has no matching "
            "row and column labels."
        )

    aligned_matrix = correlation_matrix.loc[
        feature_names,
        feature_names,
    ].copy()

    if len(feature_names) == 1:
        return aligned_matrix

    # Absolute value because both strong positive and
    # strong negative correlations are relevant.
    absolute_correlations = aligned_matrix.abs().copy()

    # Exclude the diagonal (self-correlation = 1).
    diagonal_mask = np.eye(
        len(absolute_correlations),
        dtype=bool,
    )

    absolute_correlations = absolute_correlations.mask(
        diagonal_mask
    )

    # For every feature, find its strongest correlation
    # with any other feature.
    maximum_correlations = (
        absolute_correlations
        .max(
            axis=1,
            skipna=True,
        )
        .dropna()
        .sort_values(
            ascending=False,
            kind="stable",
        )
    )

    selected_features = (
        maximum_correlations
        .head(top_n)
        .index
        .tolist()
    )

    if not selected_features:
        raise ValueError(
            "No valid pairwise feature correlations are available."
        )

    return aligned_matrix.loc[
        selected_features,
        selected_features,
    ]

def plot_correlation_heatmap(
    correlation_matrix: pd.DataFrame,
    pdf_path: Path,
    png_path: Path,
    *,
    top_n: int = 10,
) -> None:
    """
    Generate a heatmap containing only the features with
    the strongest feature-feature Spearman correlations.
    """
    correlation_matrix = select_top_correlated_features(
        correlation_matrix=correlation_matrix,
        top_n=top_n,
    )

    number_of_features = len(
        correlation_matrix.columns
    )

    figure_size = max(
        8,
        min(
            16,
            number_of_features + 4,
        ),
    )

    figure, axis = plt.subplots(
        figsize=(
            figure_size,
            figure_size - 1,
        )
    )

    sns.heatmap(
        correlation_matrix,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        annot=(number_of_features <= 15),
        fmt=".2f",
        linewidths=0.5,
        square=True,
        cbar_kws={
            "label": (
                "Spearman rank correlation "
                "($\\rho$)"
            ),
            "shrink": 0.8,
        },
        ax=axis,
    )

    axis.set_title(
        (
            "Spearman rank correlation heatmap "
            "of the top correlated features"
        ),
        pad=18,
    )

    axis.tick_params(
        axis="x",
        rotation=45,
    )

    axis.tick_params(
        axis="y",
        rotation=0,
    )

    figure.tight_layout()

    pdf_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        pdf_path,
        format="pdf",
        dpi=300,
    )

    figure.savefig(
        png_path,
        format="png",
        dpi=300,
    )

    plt.close(figure)