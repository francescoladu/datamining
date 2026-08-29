from collections.abc import Mapping, Sequence
from pathlib import Path
import re

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1. FILE-NAME UTILITY
# ============================================================

def _safe_filename(value: str) -> str:
    """
    Convert a feature name into a valid file-name component.
    """
    cleaned_value = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        value.strip(),
    )

    return cleaned_value.strip("._") or "feature"


# ============================================================
# 2. FEATURE HISTOGRAMS GROUPED BY CLASS
# ============================================================

def plot_feature_histograms_by_class(
    data: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
    output_dir: Path,
    class_label_names: Mapping[object, str] | None = None,
) -> list[Path]:
    """
    Generate one histogram for each predictive feature.

    In every histogram, feature distributions are grouped by
    the target class label.

    Each histogram is saved in PNG and PDF format.
    """
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    class_labels = sorted(
        data[target_column]
        .dropna()
        .unique()
        .tolist()
    )

    if not class_labels:
        raise ValueError(
            "No class labels are available."
        )

    generated_png_paths: list[Path] = []

    for feature_index, feature in enumerate(
        feature_columns,
        start=1,
    ):
        complete_feature_values = (
            data[feature]
            .dropna()
            .to_numpy()
        )

        unique_values = np.sort(
            np.unique(complete_feature_values)
        )

        is_discrete = (
            len(unique_values) > 0
            and len(unique_values) <= 20
            and np.allclose(
                unique_values,
                np.round(unique_values),
            )
        )

        if is_discrete:
            minimum_value = int(
                np.floor(unique_values.min())
            )

            maximum_value = int(
                np.ceil(unique_values.max())
            )

            bins = np.arange(
                minimum_value - 0.5,
                maximum_value + 1.5,
                1,
            )
        else:
            bins = np.histogram_bin_edges(
                complete_feature_values,
                bins="auto",
            )

        values_grouped_by_class = [
            (
                data.loc[
                    data[target_column] == class_label,
                    feature,
                ]
                .dropna()
                .to_numpy()
            )
            for class_label in class_labels
        ]

        legend_labels = [
            (
                class_label_names.get(
                    class_label,
                    str(class_label),
                )
                if class_label_names is not None
                else str(class_label)
            )
            for class_label in class_labels
        ]

        figure, axis = plt.subplots(
            figsize=(9, 6)
        )

        axis.hist(
            values_grouped_by_class,
            bins=bins,
            label=legend_labels,
            alpha=0.75,
            edgecolor="black",
        )

        axis.set_title(
            f"Distribution of {feature} grouped by class label",
            pad=14,
        )

        axis.set_xlabel(feature)
        axis.set_ylabel("Number of observations")

        axis.legend(
            title="Class label"
        )

        axis.grid(
            axis="y",
            alpha=0.25,
        )

        if is_discrete:
            axis.set_xticks(unique_values)

        figure.tight_layout()

        safe_feature_name = _safe_filename(
            feature
        )

        file_stem = (
            f"{feature_index:02d}_"
            f"{safe_feature_name}_"
            "distribution_by_class"
        )

        pdf_path = output_dir / f"{file_stem}.pdf"
        png_path = output_dir / f"{file_stem}.png"

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

        generated_png_paths.append(
            png_path
        )

    return generated_png_paths


# ============================================================
# 3. SPEARMAN CORRELATION HEATMAP
# ============================================================

def plot_correlation_heatmap(
    correlation_matrix: pd.DataFrame,
    pdf_path: Path,
    png_path: Path,
) -> None:
    """
    Generate a Spearman correlation heatmap containing only the
    most relevant predictive features.
    """
    if correlation_matrix.empty:
        raise ValueError(
            "The correlation matrix is empty."
        )

    number_of_features = len(
        correlation_matrix.columns
    )

    figure_size = max(
        8,
        min(16, number_of_features + 4),
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
        annot=number_of_features <= 15,
        fmt=".2f",
        linewidths=0.5,
        square=True,
        cbar_kws={
            "label": "Spearman rank correlation ($\\rho$)",
            "shrink": 0.8,
        },
        ax=axis,
    )

    axis.set_title(
        "Spearman rank correlation heatmap of the most relevant features",
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