import sys
from pathlib import Path

# Ensure the module directory is in sys.path to locate local imports.
module_dir = Path(__file__).resolve().parent
if str(module_dir) not in sys.path:
    sys.path.insert(0, str(module_dir))

import config
import data_processing
import plots


def main() -> None:
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Results and plots will be written to: {config.OUTPUT_DIR}\n")

    for dataset_name in config.DATASET_PATHS:
        print("=" * 80)
        print(f"ANALYZING SPLIT: {dataset_name.upper()}")
        print("=" * 80)

        try:
            data, X, y = data_processing.load_and_prepare_dataset(dataset_name)
        except Exception as error:
            print(f"Execution error loading dataset '{dataset_name}': {error}")
            print(
                "Verify that the database cleaning and split script "
                "was run successfully.\n"
            )
            continue

        print(f"Dataset size: {X.shape[0]} samples with {X.shape[1]} features.")

        # 1. General statistics: CSV only, no plot.
        print("\n[1/6] Generating general dataset statistics...")
        statistics = data_processing.calculate_dataset_statistics(
            data=data,
            target_column=config.TARGET_COLUMN,
            phishing_label=config.PHISHING_LABEL,
            legitimate_label=config.LEGITIMATE_LABEL,
        )
        statistics_path = (
            config.OUTPUT_DIR / f"dataset_statistics_{dataset_name}.csv"
        )
        statistics.to_csv(statistics_path, index=False)
        print(statistics.to_string(index=False))

        # 2. Conflicting profiles: CSV only, no plot.
        print(
            "\n[2/6] Inspecting conflicting profiles "
            "(identical features, different labels)..."
        )
        conflict_statistics, conflicting_profiles, _ = (
            data_processing.find_conflicting_profiles(
                data=data,
                target_column=config.TARGET_COLUMN,
            )
        )
        conflict_statistics_path = (
            config.OUTPUT_DIR
            / f"conflicting_profiles_statistics_{dataset_name}.csv"
        )
        conflict_statistics.to_csv(conflict_statistics_path, index=False)
        print(conflict_statistics.to_string(index=False))

        if not conflicting_profiles.empty:
            conflicting_profiles_path = (
                config.OUTPUT_DIR / f"conflicting_profiles_{dataset_name}.csv"
            )
            conflicting_profiles.to_csv(conflicting_profiles_path, index=False)
            print(
                "-> Saved list of conflicting profiles: "
                f"{conflicting_profiles_path.name}"
            )

        # 3. Mutual Information is used only to select relevant features.
        # No Mutual Information chart is generated.
        print("\n[3/6] Ranking features with Mutual Information...")
        mutual_information = data_processing.calculate_mutual_information(X, y)
        mutual_information_path = (
            config.OUTPUT_DIR / f"mutual_information_{dataset_name}.csv"
        )
        mutual_information.to_csv(mutual_information_path, index=False)

        relevant_feature_count = min(config.TOP_FEATURES_NUMBER, X.shape[1])
        relevant_features = (
            mutual_information["Feature"]
            .head(relevant_feature_count)
            .tolist()
        )

        print(f"-> Saved feature ranking: {mutual_information_path.name}")
        print(
            f"-> Selected {len(relevant_features)} features for the "
            "Spearman heatmap."
        )
        print(mutual_information.head(relevant_feature_count).to_string(index=False))

        # 4. Compact feature-distribution panels grouped by class.
        print(
            "\n[4/6] Generating compact feature distributions by class..."
        )

        distribution_paths = plots.plot_feature_histograms_by_class(
            data=data,

            # Mutual-Information order puts the most informative
            # features first.
            feature_columns=(
                mutual_information[
                    "Feature"
                ]
                .tolist()
            ),

            target_column=config.TARGET_COLUMN,

            output_dir=(
                config.feature_histograms_dir(
                    dataset_name
                )
            ),

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

        print(
            f"-> Generated "
            f"{len(distribution_paths)} "
            "multi-panel figures in "
            f"{config.feature_histograms_dir(dataset_name).name}/"
        )

        # 5. Second and only plot family: Spearman heatmap for relevant features.
        print("\n[5/6] Computing Spearman correlations for relevant features...")
        relevant_X = X.loc[:, relevant_features]
        correlation_matrix = (
            data_processing.calculate_spearman_correlation_matrix(relevant_X)
        )
        correlation_matrix_path = config.correlation_matrix_path(dataset_name)
        correlation_matrix.to_csv(correlation_matrix_path)

        plots.plot_correlation_heatmap(
            correlation_matrix=correlation_matrix,
            pdf_path=config.correlation_heatmap_pdf_path(dataset_name),
            png_path=config.correlation_heatmap_png_path(dataset_name),
        )
        print(f"-> Saved Spearman matrix: {correlation_matrix_path.name}")
        print(
            "-> Generated Spearman heatmap: "
            f"{config.correlation_heatmap_png_path(dataset_name).name}"
        )

        # 6. Strongest correlations: CSV only, no plot.
        print("\n[6/6] Identifying strongest correlations among selected features...")
        strongest_correlations = data_processing.find_strongest_correlations(
            correlation_matrix=correlation_matrix,
            top_n=config.TOP_CORRELATIONS_NUMBER,
        )
        strongest_correlations_path = (
            config.OUTPUT_DIR
            / f"strongest_correlations_top_features_{dataset_name}.csv"
        )
        strongest_correlations.to_csv(strongest_correlations_path, index=False)
        print(
            "-> Saved strongest correlations: "
            f"{strongest_correlations_path.name}\n"
        )

    print("=" * 80)
    print("EDA ANALYSIS PIPELINE COMPLETED SUCCESSFULLY.")
    print("=" * 80)


if __name__ == "__main__":
    main()