from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_class_distribution(
    data: pd.DataFrame,
    target_column: str,
    phishing_label: int,
    legitimate_label: int,
):
    """
    Crea il grafico della distribuzione delle classi.
    """
    class_counts = data[target_column].value_counts()

    phishing_count = int(
        class_counts.get(phishing_label, 0)
    )

    legitimate_count = int(
        class_counts.get(legitimate_label, 0)
    )

    counts = [
        phishing_count,
        legitimate_count,
    ]

    labels = [
        f"Phishing ({phishing_label})",
        f"Legitimate ({legitimate_label})",
    ]

    figure, axis = plt.subplots(
        figsize=(6.2, 4.2)
    )

    bars = axis.bar(
        labels,
        counts,
    )

    axis.set_ylabel("Number of websites")

    axis.set_title(
        "Target class distribution"
    )

    maximum_count = max(counts)

    if maximum_count > 0:
        axis.set_ylim(
            0,
            maximum_count * 1.18,
        )

    for bar, count in zip(bars, counts):
        percentage = (
            100 * count / len(data)
            if len(data) > 0
            else 0.0
        )

        axis.text(
            bar.get_x() + bar.get_width() / 2,
            count + maximum_count * 0.025,
            f"{count:,}\n({percentage:.2f}%)",
            ha="center",
            va="bottom",
        )

    figure.tight_layout()

    return figure, axis


def plot_class_rate_heatmap(
    rate_table: pd.DataFrame,
    class_name: str = "Phishing",
):
    """
    Crea una heatmap con il tasso della classe
    per ciascun valore delle feature.
    """
    matrix = rate_table.to_numpy(
        dtype=float
    )

    masked_matrix = np.ma.masked_invalid(
        matrix
    )

    figure, axis = plt.subplots(
        figsize=(7.2, 5.2)
    )

    image = axis.imshow(
        masked_matrix,
        aspect="auto",
        vmin=0,
        vmax=100,
    )

    axis.set_xticks(
        range(len(rate_table.columns))
    )

    axis.set_xticklabels(
        [
            str(value)
            for value in rate_table.columns
        ]
    )

    axis.set_yticks(
        range(len(rate_table.index))
    )

    axis.set_yticklabels(
        [
            feature.replace("_", " ")
            for feature in rate_table.index
        ]
    )

    axis.set_xlabel("Feature value")

    axis.set_title(
        f"{class_name} rate for selected features"
    )

    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[
                row_index,
                column_index,
            ]

            if not np.isnan(value):
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                )

    figure.colorbar(
        image,
        ax=axis,
        label=f"{class_name} observations (%)",
    )

    figure.tight_layout()

    return figure, axis


def plot_pearson_correlation_heatmap(
    correlation_matrix: pd.DataFrame,
):
    """
    Crea la heatmap della matrice di correlazione
    di Pearson tra tutte le feature.
    """
    figure, axis = plt.subplots(
        figsize=(20, 18)
    )

    sns.heatmap(
        correlation_matrix,
        annot=True,
        linewidths=0.5,
        fmt=".2f",
        ax=axis,
        vmin=-1,
        vmax=1,
        cmap="coolwarm",
    )

    axis.set_title(
        "Pearson Correlation Matrix of Predictive Features",
        fontsize=16,
    )

    axis.set_xticklabels(
        axis.get_xticklabels(),
        rotation=90,
        fontsize=8,
    )

    axis.set_yticklabels(
        axis.get_yticklabels(),
        rotation=0,
        fontsize=8,
    )

    figure.tight_layout()

    return figure, axis


def save_figure(
    figure,
    pdf_path: str | Path,
    png_path: str | Path,
    dpi: int = 300,
) -> None:
    """
    Salva una figura sia in PDF sia in PNG.
    """
    figure.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    figure.savefig(
        png_path,
        dpi=dpi,
        bbox_inches="tight",
    )