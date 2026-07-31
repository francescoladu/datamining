from pathlib import Path
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import config


def plot_correlation_heatmap(
    correlation_matrix: pd.DataFrame,
    pdf_path: Path,
    png_path: Path,
) -> None:
    """
    Generate and save a Pearson correlation heatmap for all feature columns.
    
    Annotations are omitted by default because a 30x30 matrix is too dense 
    to view legibly with values overlaid inside the cells.
    """
    plt.figure(figsize=(16, 14))
    
    sns.heatmap(
        correlation_matrix,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        annot=False,
        linewidths=0.5,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    
    plt.title("Pearson Correlation Matrix (Feature-to-Feature)", fontsize=16, pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    
    # Ensure parent output directory exists
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(pdf_path, format="pdf", dpi=300)
    plt.savefig(png_path, format="png", dpi=300)
    plt.close()


def plot_phishing_rate_heatmap(
    rate_table: pd.DataFrame,
    pdf_path: Path,
    png_path: Path,
) -> None:
    """
    Generate and save a heatmap detailing the phishing rate (%) 
    by feature and specific feature value.
    """
    plt.figure(figsize=(10, 8))
    
    # Convert numerical column headers into human-readable descriptions
    header_mapping = {
        -1: "-1 (Phishing / Low)",
        0: "0 (Suspicious / Mid)",
        1: "1 (Legitimate / High)"
    }
    
    # Extract only existing column codes found in the table
    columns_to_rename = {k: v for k, v in header_mapping.items() if k in rate_table.columns}
    plot_table = rate_table.rename(columns=columns_to_rename)
    
    sns.heatmap(
        plot_table,
        cmap="Reds",
        vmin=0,
        vmax=100,
        annot=True,
        fmt=".1f",
        linewidths=0.5,
        cbar_kws={"label": "Phishing Rate (%)"}
    )
    
    plt.title(
        f"Phishing Rate (%) by Feature and Value\n(Top {len(rate_table)} Features Ranked by Mutual Information)",
        fontsize=14,
        pad=15
    )
    plt.ylabel("Feature Name", fontsize=12)
    plt.xlabel("Feature Value Code", fontsize=12)
    plt.tight_layout()
    
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(pdf_path, format="pdf", dpi=300)
    plt.savefig(png_path, format="png", dpi=300)
    plt.close()


def plot_mutual_information(
    mi_df: pd.DataFrame,
    pdf_path: Path,
    png_path: Path,
    top_n: int = 15,
) -> None:
    """
    Generate and save a horizontal bar plot displaying features 
    with the highest Mutual Information scores.
    """
    plt.figure(figsize=(10, 7))
    
    plot_data = mi_df.head(top_n)
    
    sns.barplot(
        x="Mutual Information",
        y="Feature",
        data=plot_data,
        palette="viridis",
        hue="Feature",
        legend=False
    )
    
    plt.title(f"Top {top_n} Features Ranked by Mutual Information Score", fontsize=14, pad=15)
    plt.xlabel("Mutual Information Score", fontsize=12)
    plt.ylabel("Feature Name", fontsize=12)
    plt.tight_layout()
    
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(pdf_path, format="pdf", dpi=300)
    plt.savefig(png_path, format="png", dpi=300)
    plt.close()