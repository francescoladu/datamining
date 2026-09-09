import argparse
import sys
from pathlib import Path

module_dir = Path(__file__).resolve().parent
code_dir = module_dir.parent

if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

from data_preprocessing import config
from data_preprocessing import data_processing
from data_preprocessing import plots


def parse_arguments() -> argparse.Namespace:
    """Select exactly one experiment for exploratory data analysis."""

    parser = argparse.ArgumentParser(
        description=(
            "Run exploratory data analysis "
            "for one Data Mining experiment."
        )
    )

    parser.add_argument(
        "experiment",
        choices=["1", "2", "3"],
        help=(
            "Experiment to analyze: "
            "1 = raw data, "
            "2 = exact deduplication, "
            "3 = weighted deduplication."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    experiment = args.experiment

    dataset_name = "train"

    dataset_path = (
        config.EXPERIMENT_DATASET_PATHS[
            experiment
        ]
    )

    output_dir = (
        config.EXPERIMENT_OUTPUT_DIRS[
            experiment
        ]
    )

    experiment_name = (
        config.EXPERIMENT_NAMES[
            experiment
        ]
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print(f"EDA - {experiment_name}")
    print("=" * 80)

    print(
        f"Dataset: {dataset_path}"
    )

    print(
        f"Results and plots will be written to: "
        f"{output_dir}\n"
    )

    try:
        data, X, y, sample_weight = (
            data_processing.load_and_prepare_dataset(
                file_path=dataset_path,
            )
        )

    except Exception as error:
        print(
            f"Execution error loading "
            f"Experiment {experiment}: {error}"
        )

        print(
            "Verify that the corresponding "
            "make install command was run first."
        )

        sys.exit(1)

    print(
        f"Dataset rows: {X.shape[0]}"
    )

    print(
        f"Predictive features: "
        f"{X.shape[1]}"
    )

    print(
        f"Observation mass: "
        f"{sample_weight.sum():.0f}"
    )

    # ------------------------------------------------------------------
    # 1. General statistics
    # ------------------------------------------------------------------

    print(
        "\n[1/6] Generating general dataset statistics..."
    )

    statistics = (
        data_processing.calculate_dataset_statistics(
            data=data,
            target_column=config.TARGET_COLUMN,
            weight_column=config.SAMPLE_WEIGHT_COLUMN,
            phishing_label=config.PHISHING_LABEL,
            legitimate_label=config.LEGITIMATE_LABEL,
        )
    )

    statistics_path = (
        output_dir
        / f"dataset_statistics_{dataset_name}.csv"
    )

    statistics.to_csv(
        statistics_path,
        index=False,
    )

    print(
        statistics.to_string(
            index=False
        )
    )

    # ------------------------------------------------------------------
    # 2. Conflicting profiles
    # ------------------------------------------------------------------

    print(
        "\n[2/6] Inspecting conflicting profiles "
        "(identical features, different labels)..."
    )

    (
        conflict_statistics,
        conflicting_profiles,
        _,
    ) = data_processing.find_conflicting_profiles(
        data=data,
        target_column=config.TARGET_COLUMN,
        weight_column=config.SAMPLE_WEIGHT_COLUMN,
    )

    conflict_statistics_path = (
        output_dir
        / f"conflicting_profiles_statistics_{dataset_name}.csv"
    )

    conflict_statistics.to_csv(
        conflict_statistics_path,
        index=False,
    )

    print(
        conflict_statistics.to_string(
            index=False
        )
    )

    if not conflicting_profiles.empty:
        conflicting_profiles_path = (
            output_dir
            / f"conflicting_profiles_{dataset_name}.csv"
        )

        conflicting_profiles.to_csv(
            conflicting_profiles_path,
            index=False,
        )

        print(
            "-> Saved list of conflicting profiles: "
            f"{conflicting_profiles_path.name}"
        )

    # ------------------------------------------------------------------
    # 3. Mutual Information
    # ------------------------------------------------------------------

    print(
        "\n[3/6] Ranking features with Mutual Information..."
    )

    mutual_information = (
        data_processing.calculate_mutual_information(
            X,
            y,
        )
    )

    mutual_information_path = (
        output_dir
        / f"mutual_information_{dataset_name}.csv"
    )

    mutual_information.to_csv(
        mutual_information_path,
        index=False,
    )

    relevant_feature_count = min(
        config.TOP_FEATURES_NUMBER,
        X.shape[1],
    )

    relevant_features = (
        mutual_information[
            "Feature"
        ]
        .head(
            relevant_feature_count
        )
        .tolist()
    )

    print(
        f"-> Saved feature ranking: "
        f"{mutual_information_path.name}"
    )

    print(
        f"-> Selected "
        f"{len(relevant_features)} "
        "features for the Spearman heatmap."
    )

    print(
        mutual_information
        .head(relevant_feature_count)
        .to_string(index=False)
    )

    # ------------------------------------------------------------------
    # 4. Feature distributions
    # ------------------------------------------------------------------

    print(
        "\n[4/6] Generating compact "
        "feature distributions by class..."
    )

    histogram_dir = (
        config.feature_histograms_dir(
            output_dir,
            dataset_name,
        )
    )

    distribution_paths = (
        plots.plot_feature_histograms_by_class(
            data=data,
            feature_columns=(
                mutual_information[
                    "Feature"
                ].tolist()
            ),
            target_column=config.TARGET_COLUMN,
            output_dir=histogram_dir,
            weight_column=config.SAMPLE_WEIGHT_COLUMN,
            class_label_names={
                config.PHISHING_LABEL: (
                    f"Phishing "
                    f"({config.PHISHING_LABEL})"
                ),
                config.LEGITIMATE_LABEL: (
                    f"Legitimate "
                    f"({config.LEGITIMATE_LABEL})"
                ),
            },
            features_per_figure=10,
            columns_per_figure=2,
        )
    )

    print(
        f"-> Generated "
        f"{len(distribution_paths)} "
        f"figures in {histogram_dir.name}/"
    )

    # ------------------------------------------------------------------
    # 5. Spearman correlations
    # ------------------------------------------------------------------

    print(
        "\n[5/6] Computing Spearman correlations "
        "for relevant features..."
    )

    relevant_X = X.loc[
        :,
        relevant_features,
    ]

    correlation_matrix = (
        data_processing
        .calculate_spearman_correlation_matrix(
            relevant_X
        )
    )

    correlation_matrix_path = (
        config.correlation_matrix_path(
            output_dir,
            dataset_name,
        )
    )

    correlation_matrix.to_csv(
        correlation_matrix_path
    )

    correlation_pdf_path = (
        config.correlation_heatmap_pdf_path(
            output_dir,
            dataset_name,
        )
    )

    correlation_png_path = (
        config.correlation_heatmap_png_path(
            output_dir,
            dataset_name,
        )
    )

    plots.plot_correlation_heatmap(
        correlation_matrix=correlation_matrix,
        pdf_path=correlation_pdf_path,
        png_path=correlation_png_path,
    )

    print(
        f"-> Saved Spearman matrix: "
        f"{correlation_matrix_path.name}"
    )

    print(
        f"-> Generated Spearman heatmap: "
        f"{correlation_png_path.name}"
    )

    # ------------------------------------------------------------------
    # 6. Strongest correlations
    # ------------------------------------------------------------------

    print(
        "\n[6/6] Identifying strongest "
        "correlations among selected features..."
    )

    strongest_correlations = (
        data_processing.find_strongest_correlations(
            correlation_matrix=correlation_matrix,
            top_n=config.TOP_CORRELATIONS_NUMBER,
        )
    )

    strongest_correlations_path = (
        output_dir
        / (
            "strongest_correlations_"
            f"top_features_{dataset_name}.csv"
        )
    )

    strongest_correlations.to_csv(
        strongest_correlations_path,
        index=False,
    )

    print(
        f"-> Saved strongest correlations: "
        f"{strongest_correlations_path.name}\n"
    )

    print("=" * 80)

    print(
        f"EDA EXPERIMENT {experiment} "
        "COMPLETED SUCCESSFULLY."
    )

    print("=" * 80)


if __name__ == "__main__":
    main()