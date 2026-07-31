import matplotlib.pyplot as plt

from config import (
    DATASET_PATH,
    TARGET_COLUMN,
    INDEX_COLUMNS,
    PHISHING_LABEL,
    LEGITIMATE_LABEL,
    RANDOM_STATE,
    CORRELATION_MATRIX_PATH,
    CORRELATION_HEATMAP_PDF_PATH,
    CORRELATION_HEATMAP_PNG_PATH,
)

from data_processing import (
    load_clean_dataset,
    validate_dataset,
    split_features_target,
)

from dataset_statistics import (
    calculate_dataset_statistics,
    find_conflicting_profiles,
)

from associations import (
    calculate_mutual_information,
    calculate_pearson_correlation_matrix,
    find_strongest_correlations,
    calculate_class_rate_by_feature_value,
)

from plots import (
    plot_class_distribution,
    plot_class_rate_heatmap,
    plot_pearson_correlation_heatmap,
    save_figure,
)


def main() -> None:
    # ========================================================
    # 1. CARICAMENTO DEL DATASET
    # ========================================================

    phishing_df = load_clean_dataset(
        file_path=DATASET_PATH,
        index_columns=INDEX_COLUMNS,
    )


    # ========================================================
    # 2. CONTROLLO DEL DATASET
    # ========================================================

    validate_dataset(
        data=phishing_df,
        target_column=TARGET_COLUMN,
    )

    print("Dimensione del dataset:")
    print(phishing_df.shape)

    print("\nPrime righe:")
    print(phishing_df.head())

    print("\nTipi delle colonne:")
    print(phishing_df.dtypes)

    print("\nValori mancanti:")
    print(phishing_df.isnull().sum())


    # ========================================================
    # 3. SEPARAZIONE DEL TARGET
    # ========================================================

    X, y = split_features_target(
        data=phishing_df,
        target_column=TARGET_COLUMN,
    )


    # ========================================================
    # 4. STATISTICHE DEL DATASET
    # ========================================================

    dataset_statistics = calculate_dataset_statistics(
        data=phishing_df,
        target_column=TARGET_COLUMN,
        phishing_label=PHISHING_LABEL,
        legitimate_label=LEGITIMATE_LABEL,
    )

    print("\nDATASET STATISTICS")
    print(
        dataset_statistics.to_string(
            index=False
        )
    )


    # ========================================================
    # 5. PROFILI CONTRADDITTORI
    # ========================================================

    (
        conflict_statistics,
        conflicting_profiles,
        conflicting_rows,
    ) = find_conflicting_profiles(
        data=phishing_df,
        target_column=TARGET_COLUMN,
    )

    print("\nCONFLICTING PROFILES")
    print(
        conflict_statistics.to_string(
            index=False
        )
    )


    # ========================================================
    # 6. MUTUAL INFORMATION
    # ========================================================

    mutual_information = calculate_mutual_information(
        X=X,
        y=y,
        random_state=RANDOM_STATE,
    )

    print("\nMUTUAL INFORMATION")
    print(
        mutual_information.head(15).to_string(
            index=False
        )
    )


    # ========================================================
    # 7. MATRICE DI CORRELAZIONE DI PEARSON
    # ========================================================

    correlation_matrix = (
        calculate_pearson_correlation_matrix(
            X=X
        )
    )

    print("\nPEARSON CORRELATION MATRIX")
    print(correlation_matrix)


    # ========================================================
    # 8. CORRELAZIONI PIÙ FORTI
    # ========================================================

    strongest_correlations = (
        find_strongest_correlations(
            correlation_matrix=correlation_matrix,
            top_n=15,
        )
    )

    print("\nSTRONGEST PEARSON CORRELATIONS")
    print(
        strongest_correlations.to_string(
            index=False
        )
    )


    # ========================================================
    # 9. SALVATAGGIO DELLA MATRICE
    # ========================================================

    correlation_matrix.to_csv(
        CORRELATION_MATRIX_PATH
    )


    # ========================================================
    # 10. TASSO DI PHISHING
    # ========================================================

    top_features = (
        mutual_information
        .head(8)["Feature"]
        .tolist()
    )

    phishing_rate_table = (
        calculate_class_rate_by_feature_value(
            data=phishing_df,
            features=top_features,
            target_column=TARGET_COLUMN,
            class_label=PHISHING_LABEL,
            possible_values=(-1, 0, 1),
        )
    )

    print("\nPHISHING RATE BY FEATURE VALUE")
    print(
        phishing_rate_table.round(2)
    )


    # ========================================================
    # 11. DISTRIBUZIONE DELLE CLASSI
    # ========================================================

    plot_class_distribution(
        data=phishing_df,
        target_column=TARGET_COLUMN,
        phishing_label=PHISHING_LABEL,
        legitimate_label=LEGITIMATE_LABEL,
    )


    # ========================================================
    # 12. HEATMAP DEL TASSO DI PHISHING
    # ========================================================

    plot_class_rate_heatmap(
        rate_table=phishing_rate_table,
        class_name="Phishing",
    )


    # ========================================================
    # 13. HEATMAP DI PEARSON
    # ========================================================

    correlation_figure, correlation_axis = (
        plot_pearson_correlation_heatmap(
            correlation_matrix=correlation_matrix
        )
    )


    # ========================================================
    # 14. SALVATAGGIO DELLA HEATMAP
    # ========================================================

    save_figure(
        figure=correlation_figure,
        pdf_path=CORRELATION_HEATMAP_PDF_PATH,
        png_path=CORRELATION_HEATMAP_PNG_PATH,
        dpi=300,
    )


    # ========================================================
    # 15. VISUALIZZAZIONE
    # ========================================================

    plt.show()


if __name__ == "__main__":
    main()