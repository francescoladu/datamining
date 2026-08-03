from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from config import (
    PERMUTATION_MAX_DISPLAY,
    PERMUTATION_PDF_PATH,
    PERMUTATION_PNG_PATH,
    PLOT_DPI,
    SAVE_PDF,
    SAVE_PNG,
    create_output_directory,
)


def run_permutation_importance_plot(
    importance_df: pd.DataFrame,
    max_display: int | None = PERMUTATION_MAX_DISPLAY,
) -> tuple[Path | None, Path | None]:
    """
    Create and save a horizontal permutation importance plot.

    The error bars represent the standard deviation of the
    mean importance across the outer folds.
    """

    required_columns = {
        "feature",
        "importance_mean",
        "importance_std",
    }

    missing_columns = required_columns.difference(
        importance_df.columns
    )

    if importance_df.empty or missing_columns:
        raise ValueError(
            "Invalid permutation importance DataFrame. "
            f"Missing columns: {sorted(missing_columns)}"
        )

    plot_df = importance_df.sort_values(
        "importance_mean",
        ascending=False,
    )

    if max_display is not None:
        if max_display <= 0:
            raise ValueError(
                "max_display must be positive or None."
            )

        plot_df = plot_df.head(max_display)

    # Reverse the order so that the most important feature
    # appears at the top of the horizontal bar plot.
    plot_df = plot_df.sort_values(
        "importance_mean",
        ascending=True,
    )

    create_output_directory()

    figure_height = max(
        5.0,
        0.38 * len(plot_df),
    )

    figure, axis = plt.subplots(
        figsize=(10, figure_height)
    )

    axis.barh(
        plot_df["feature"],
        plot_df["importance_mean"],
        xerr=plot_df["importance_std"],
        capsize=3,
    )

    # Separate positive and negative importance values.
    axis.axvline(
        0.0,
        linewidth=1.0,
    )

    axis.set_title(
        (
            "Global Explainability — "
            "Nested-CV Permutation Importance"
        ),
        pad=15,
    )

    axis.set_xlabel(
        "Mean decrease in macro F1-score"
    )

    axis.set_ylabel(
        "Feature"
    )

    axis.grid(
        axis="x",
        linestyle="--",
        alpha=0.4,
    )

    figure.tight_layout()

    png_path: Path | None = None
    pdf_path: Path | None = None

    if SAVE_PNG:
        figure.savefig(
            PERMUTATION_PNG_PATH,
            dpi=PLOT_DPI,
            bbox_inches="tight",
        )

        png_path = PERMUTATION_PNG_PATH

    if SAVE_PDF:
        figure.savefig(
            PERMUTATION_PDF_PATH,
            bbox_inches="tight",
        )

        pdf_path = PERMUTATION_PDF_PATH

    plt.close(figure)

    if png_path is not None:
        print(
            "\nPermutation plot PNG saved to:"
            f"\n{png_path}"
        )

    if pdf_path is not None:
        print(
            "\nPermutation plot PDF saved to:"
            f"\n{pdf_path}"
        )

    return png_path, pdf_path