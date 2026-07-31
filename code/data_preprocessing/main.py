import sys
from pathlib import Path

# Ensure the module directory is in the sys.path to easily locate local imports
module_dir = Path(__file__).resolve().parent
if str(module_dir) not in sys.path:
    sys.path.insert(0, str(module_dir))

import config
import data_processing
import plots


def main() -> None:
    # 1. Initialize outputs directory
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Results and plots will be written to: {config.OUTPUT_DIR}\n")

    # 2. Iterate through configured dataset splits
    for dataset_name in config.DATASET_PATHS.keys():
        print("=" * 80)
        print(f"ANALYZING SPLIT: {dataset_name.upper()}")
        print("=" * 80)

        # 2.1 Load & Prepare
        try:
            data, X, y = data_processing.load_and_prepare_dataset(dataset_name)
        except Exception as e:
            print(f"Execution Error loading dataset '{dataset_name}': {e}")
            print("Verify that your database cleaning and split script was run successfully.\n")
            continue

        print(f"Dataset Size: {X.shape[0]} samples with {X.shape[1]} features.")

        # 2.2 Calculate General Dataset Statistics
        print("\n[1/6] Generating general dataset statistics...")
        stats = data_processing.calculate_dataset_statistics(
            data=data,
            target_column=config.TARGET_COLUMN,
            phishing_label=config.PHISHING_LABEL,
            legitimate_label=config.LEGITIMATE_LABEL
        )
        stats_path = config.OUTPUT_DIR / f"dataset_statistics_{dataset_name}.csv"
        stats.to_csv(stats_path, index=False)
        print(stats.to_string(index=False))

        # 2.3 Investigate Conflicting Feature Profiles
        print("\n[2/6] Inspecting for conflicting profiles (identical features, different labels)...")
        conf_stats, conf_profiles, conf_rows = data_processing.find_conflicting_profiles(
            data=data,
            target_column=config.TARGET_COLUMN
        )
        conf_stats_path = config.OUTPUT_DIR / f"conflicting_profiles_statistics_{dataset_name}.csv"
        conf_stats.to_csv(conf_stats_path, index=False)
        print(conf_stats.to_string(index=False))

        if len(conf_profiles) > 0:
            conf_profiles_path = config.OUTPUT_DIR / f"conflicting_profiles_{dataset_name}.csv"
            conf_profiles.to_csv(conf_profiles_path, index=False)
            print(f"-> Saved list of conflicting profiles: {conf_profiles_path.name}")

        # 2.4 Calculate & Plot Mutual Information (Feature vs. Target)
        print("\n[3/6] Running Mutual Information calculations...")
        mi_results = data_processing.calculate_mutual_information(X, y)
        mi_path = config.OUTPUT_DIR / f"mutual_information_{dataset_name}.csv"
        mi_results.to_csv(mi_path, index=False)
        
        # Save MI plots
        mi_pdf = config.OUTPUT_DIR / f"mutual_information_{dataset_name}.pdf"
        mi_png = config.OUTPUT_DIR / f"mutual_information_{dataset_name}.png"
        plots.plot_mutual_information(mi_results, mi_pdf, mi_png, top_n=15)
        print(f"-> Saved MI rankings to {mi_path.name}")
        print(f"-> Generated Mutual Information plot: {mi_png.name}")
        print("\nTop 5 predictive features by Mutual Information:")
        print(mi_results.head(5).to_string(index=False))

        # 2.5 Generate Feature-to-Feature Pearson Correlation Matrix
        print("\n[4/6] Computing Pearson correlation matrix...")
        corr_matrix = data_processing.calculate_pearson_correlation_matrix(X)
        corr_path = config.correlation_matrix_path(dataset_name)
        corr_matrix.to_csv(corr_path)
        
        # Plot Correlation Heatmap
        plots.plot_correlation_heatmap(
            correlation_matrix=corr_matrix,
            pdf_path=config.correlation_heatmap_pdf_path(dataset_name),
            png_path=config.correlation_heatmap_png_path(dataset_name)
        )
        print(f"-> Saved Pearson correlation matrix: {corr_path.name}")
        print(f"-> Generated correlation heatmap: {config.correlation_heatmap_png_path(dataset_name).name}")

        # 2.6 Extract Strongest Correlations
        print("\n[5/6] Identifying top collinear feature pairs...")
        strongest_corr = data_processing.find_strongest_correlations(
            correlation_matrix=corr_matrix,
            top_n=config.TOP_CORRELATIONS_NUMBER
        )
        strongest_path = config.OUTPUT_DIR / f"strongest_correlations_{dataset_name}.csv"
        strongest_corr.to_csv(strongest_path, index=False)
        print(f"-> Saved top collinear relationships to {strongest_path.name}")
        print("\nTop 5 strongest correlations:")
        print(strongest_corr.head(5).to_string(index=False))

        # 2.7 Compute & Plot Phishing Rates for Top MI Features
        print(f"\n[6/6] Computing phishing rate details for the top {config.TOP_FEATURES_NUMBER} features...")
        top_features = mi_results["Feature"].head(config.TOP_FEATURES_NUMBER).tolist()
        rate_table = data_processing.calculate_class_rate_by_feature_value(
            data=data,
            features=top_features,
            target_column=config.TARGET_COLUMN,
            class_label=config.PHISHING_LABEL
        )
        rate_path = config.OUTPUT_DIR / f"phishing_rate_{dataset_name}.csv"
        rate_table.to_csv(rate_path)
        
        # Plot Phishing Rate Heatmap
        rate_pdf = config.OUTPUT_DIR / f"phishing_rate_heatmap_{dataset_name}.pdf"
        rate_png = config.OUTPUT_DIR / f"phishing_rate_heatmap_{dataset_name}.png"
        plots.plot_phishing_rate_heatmap(rate_table, rate_pdf, rate_png)
        print(f"-> Saved class rate table details to {rate_path.name}")
        print(f"-> Generated phishing rate heatmap: {rate_png.name}")
        print("\n")

    print("=" * 80)
    print("EDA ANALYSIS PIPELINE COMPLETED SUCCESSFULLY.")
    print("=" * 80)


if __name__ == "__main__":
    main()