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


def _validate_permutation_dataframe(
    importance_df: pd.DataFrame,
) -> None:
    """Validate the permutation importance DataFrame."""

    if not isinstance(importance_df, pd.DataFrame):
        raise TypeError(
            "importance_df must be a pandas DataFrame."
        )

    if importance_df.empty:
        raise ValueError(
            "The permutation importance DataFrame cannot be empty."
        )

    required_columns = {
        "feature",
        "importance_mean",
        "importance_std",
    }

    missing_columns = required_columns.difference(
        importance_df.columns
    )

    if missing_columns:
        raise ValueError(
            "The permutation importance DataFrame is missing "
            f"the following columns: {sorted(missing_columns)}"
        )

    if importance_df["feature"].isna().any():
        raise ValueError(
            "The feature column cannot contain missing values."
        )

    if importance_df["importance_mean"].isna().any():
        raise ValueError(
            "The importance_mean column cannot contain "
            "missing values."
        )

    if importance_df["importance_std"].isna().any():
        raise ValueError(
            "The importance_std column cannot contain "
            "missing values."
        )


def _select_features_to_display(
    importance_df: pd.DataFrame,
    max_display: int | None,
) -> pd.DataFrame:
    """
    Select and order the features displayed in the plot.

    Features are first ranked from the largest to the smallest
    mean permutation importance. They are then reversed so that
    the most important feature appears at the top of the
    horizontal bar chart.
    """

    sorted_df = importance_df.sort_values(
        by="importance_mean",
        ascending=False,
    )

    if max_display is not None:
        if max_display <= 0:
            raise ValueError(
                "max_display must be greater than zero or None."
            )

        sorted_df = sorted_df.head(max_display)

    return (
        sorted_df
        .sort_values(
            by="importance_mean",
            ascending=True,
        )
        .reset_index(drop=True)
    )


def plot_permutation_importance(
    importance_df: pd.DataFrame,
    *,
    max_display: int | None = PERMUTATION_MAX_DISPLAY,
    title: str = (
        "Global Explainability — Permutation Importance"
    ),
    xlabel: str = "Mean decrease in macro F1-score",
    show: bool = False,
) -> tuple[Path | None, Path | None]:
    """
    Create and save the permutation importance plot.

    Parameters
    ----------
    importance_df:
        DataFrame returned by compute_permutation_importance().

    max_display:
        Maximum number of features displayed in the plot.
        When None, all features are displayed.

    title:
        Title displayed above the plot.

    xlabel:
        Label displayed on the horizontal axis.

    show:
        If True, display the plot in an interactive window.

    Returns
    -------
    tuple[Path | None, Path | None]
        Paths of the saved PNG and PDF files. A path is None
        when the corresponding output format is disabled.
    """

    _validate_permutation_dataframe(
        importance_df=importance_df,
    )

    plot_df = _select_features_to_display(
        importance_df=importance_df,
        max_display=max_display,
    )

    create_output_directory()

    # Increase the figure height according to the number
    # of displayed features.
    figure_height = max(
        5.0,
        0.38 * len(plot_df),
    )

    figure, axis = plt.subplots(
        figsize=(10, figure_height),
    )

    axis.barh(
        y=plot_df["feature"],
        width=plot_df["importance_mean"],
        xerr=plot_df["importance_std"],
        capsize=3,
    )

    # The vertical line separates positive importance values
    # from negative importance values.
    axis.axvline(
        x=0.0,
        linewidth=1.0,
    )

    axis.set_title(
        title,
        pad=15,
    )

    axis.set_xlabel(
        xlabel,
    )

    axis.set_ylabel(
        "Feature",
    )

    axis.grid(
        axis="x",
        linestyle="--",
        alpha=0.4,
    )

    figure.tight_layout()

    saved_png_path: Path | None = None
    saved_pdf_path: Path | None = None

    if SAVE_PNG:
        PERMUTATION_PNG_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        figure.savefig(
            PERMUTATION_PNG_PATH,
            dpi=PLOT_DPI,
            bbox_inches="tight",
        )

        saved_png_path = PERMUTATION_PNG_PATH

    if SAVE_PDF:
        PERMUTATION_PDF_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        figure.savefig(
            PERMUTATION_PDF_PATH,
            bbox_inches="tight",
        )

        saved_pdf_path = PERMUTATION_PDF_PATH

    if show:
        plt.show()

    plt.close(figure)

    return saved_png_path, saved_pdf_path


def print_plot_paths(
    png_path: Path | None,
    pdf_path: Path | None,
) -> None:
    """Print the paths of the generated plot files."""

    if png_path is not None:
        print(
            "\nPermutation importance PNG saved to:"
            f"\n{png_path}"
        )

    if pdf_path is not None:
        print(
            "\nPermutation importance PDF saved to:"
            f"\n{pdf_path}"
        )


def run_permutation_importance_plot(
    importance_df: pd.DataFrame,
    *,
    max_display: int | None = PERMUTATION_MAX_DISPLAY,
    show: bool = False,
) -> tuple[Path | None, Path | None]:
    """
    Execute the complete permutation importance plotting step.

    The function:
    1. validates the permutation importance results;
    2. selects the features to display;
    3. creates the horizontal bar chart;
    4. saves the chart as PNG and PDF;
    5. prints the generated output paths.
    """

    png_path, pdf_path = plot_permutation_importance(
        importance_df=importance_df,
        max_display=max_display,
        show=show,
    )

    print_plot_paths(
        png_path=png_path,
        pdf_path=pdf_path,
    )

    return png_path, pdf_path