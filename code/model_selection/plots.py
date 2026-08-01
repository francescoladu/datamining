from pathlib import Path
import numpy as np
import pandas as pd

# Set Agg backend to ensure safe background generation in headless environments
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_nested_cv_comparison(
    nested_scores: pd.DataFrame,
    output_pdf_path: str | Path,
    model_order: list[str] = ["Decision Tree", "Random Forest"],
) -> None:
    """
    Generate and save a boxplot comparing the outer-fold macro F1-scores of 
    the model families, overlaying the individual fold scores as scatter points.
    """
    output_pdf_path = Path(output_pdf_path)

    # Isolate the outer-fold scores for each model family
    boxplot_values = [
        nested_scores.loc[
            nested_scores["model"] == model_name,
            "macro_f1",
        ].to_numpy()
        for model_name in model_order
    ]

    # Use the object-oriented API for cleaner, environment-safe layout control
    fig, ax = plt.subplots(figsize=(7, 5))

    # Render the boxplot (showmeans adds a marker for the mean F1 score)
    ax.boxplot(
        boxplot_values,
        showmeans=True,
    )
    
    # Set labels safely (set_xticklabels is compatible across all matplotlib versions)
    ax.set_xticklabels(model_order)

    # Overlay the individual outer-fold scores as points on top of the boxes
    for position, values in enumerate(
        boxplot_values,
        start=1,
    ):
        x_positions = np.full(
            shape=len(values),
            fill_value=position,
            dtype=float,
        )

        ax.scatter(
            x_positions,
            values,
            zorder=3,          # Ensures scatter points sit on top of the box lines
            alpha=0.8,
            edgecolors="black",
            s=40               # Marker size
        )

    ax.set_ylabel("Outer-fold macro F1-score", fontsize=11)
    ax.set_xlabel("Model family", fontsize=11)
    ax.set_title("Nested Cross-Validation Performance Comparison", fontsize=12, pad=15)
    
    fig.tight_layout()

    # Ensure output folder exists before writing to disk
    output_pdf_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        output_pdf_path,
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(fig)