from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split


RAW_TRAIN_FILENAME = "experiment_1_raw_train.csv"
RAW_TEST_FILENAME = "experiment_1_raw_test.csv"

STANDARD_DEDUP_TRAIN_FILENAME = "experiment_2_standard_dedup_train.csv"
STANDARD_DEDUP_TEST_FILENAME = "experiment_2_standard_dedup_test.csv"

WEIGHTED_TRAIN_FILENAME = "train_cleaned.csv"
WEIGHTED_TEST_FILENAME = "test_cleaned.csv"


def load_source_dataset(
    input_path: str | Path,
) -> tuple[pd.DataFrame, str]:
    """
    Load the original ARFF dataset and apply only
    source-format cleaning.
    """
    input_path = Path(input_path)

    print(f"Loading ARFF file from: {input_path}")

    if not input_path.exists():
        raise FileNotFoundError(
            f"Source file not found at {input_path}."
        )

    raw_data, _ = arff.loadarff(input_path)
    df = pd.DataFrame(raw_data)

    # Decode ARFF byte columns and enforce integer values.
    for column in df.columns:
        if df[column].dtype == object:
            df[column] = (
                df[column]
                .str.decode("utf-8")
                .astype(int)
            )
        else:
            df[column] = df[column].astype(int)

    # Remove possible identifier/index columns.
    id_columns = [
        column
        for column in df.columns
        if column.lower()
        in {
            "id",
            "index",
            "idx",
            "unnamed: 0",
        }
    ]

    if id_columns:
        print(
            f"Removing identifier column(s): "
            f"{id_columns}"
        )

        df = df.drop(
            columns=id_columns
        )

    # Identify target column.
    possible_targets = [
        "Result",
        "result",
        "class",
        "Class",
    ]

    target_column = next(
        (
            column
            for column in possible_targets
            if column in df.columns
        ),
        None,
    )

    if target_column is None:
        raise ValueError(
            "Could not identify target column."
        )

    # Standardize the target name used by the project.
    if target_column != "Result":
        df = df.rename(
            columns={
                target_column: "Result"
            }
        )
        target_column = "Result"

    print(
        f"Target column identified: "
        f"'{target_column}'"
    )

    print(
        f"Original dataset size: "
        f"{len(df)} rows"
    )

    return (
        df.reset_index(drop=True),
        target_column,
    )


def build_common_profile_split(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create the common train/test partition used by all experiments.

    The split is performed on unique feature signatures BEFORE
    applying any experiment-specific deduplication.

    Therefore, all observations sharing the same predictor values
    always belong entirely to either train or test.
    """
    feature_columns = [
        column
        for column in df.columns
        if column != target_column
    ]

    if not feature_columns:
        raise ValueError(
            "No predictive features were found."
        )

    # ---------------------------------------------------------
    # Build one row for each unique feature signature.
    #
    # For stratification:
    # - non-conflicting profiles use their only label;
    # - conflicting profiles with a majority use that majority;
    # - exact ties are handled separately.
    # ---------------------------------------------------------
    label_counts = (
        df.groupby(
            feature_columns,
            dropna=False,
            sort=False,
        )[target_column]
        .value_counts()
        .unstack(fill_value=0)
    )

    maximum_count = label_counts.max(
        axis=1
    )

    tied_profiles_mask = (
        label_counts.eq(
            maximum_count,
            axis=0,
        )
        .sum(axis=1)
        > 1
    )

    majority_labels = label_counts.idxmax(
        axis=1
    )

    profile_table = (
        label_counts.index
        .to_frame(index=False)
        .reset_index(drop=True)
    )

    profile_table[
        "_majority_label"
    ] = majority_labels.to_numpy()

    profile_table[
        "_is_tie"
    ] = tied_profiles_mask.to_numpy()

    non_tied_profiles = profile_table.loc[
        ~profile_table["_is_tie"]
    ].copy()

    tied_profiles = profile_table.loc[
        profile_table["_is_tie"]
    ].copy()

    # ---------------------------------------------------------
    # Split non-tied profiles.
    # Stratification is based on majority/unique class.
    # ---------------------------------------------------------
    if len(non_tied_profiles) < 2:
        raise ValueError(
            "Not enough non-tied feature profiles "
            "to create train/test partitions."
        )

    class_counts = (
        non_tied_profiles[
            "_majority_label"
        ]
        .value_counts()
    )

    number_of_profiles = len(
        non_tied_profiles
    )

    number_of_test_profiles = int(
        np.ceil(
            number_of_profiles
            * test_size
        )
    )

    number_of_train_profiles = (
        number_of_profiles
        - number_of_test_profiles
    )

    number_of_classes = len(
        class_counts
    )

    can_stratify = (
        number_of_classes >= 2
        and (class_counts >= 2).all()
        and number_of_test_profiles
        >= number_of_classes
        and number_of_train_profiles
        >= number_of_classes
    )

    if can_stratify:
        stratification_labels = (
            non_tied_profiles[
                "_majority_label"
            ]
        )

        print(
            "Common profile split: "
            "stratified by profile majority label."
        )

    else:
        stratification_labels = None

        print(
            "Warning: profile-level stratification "
            "is not possible with the available "
            "class counts. Using a deterministic "
            "unstratified profile split."
        )

    (
        non_tied_train,
        non_tied_test,
    ) = train_test_split(
        non_tied_profiles,
        test_size=test_size,
        stratify=stratification_labels,
        random_state=random_state,
    )

    # ---------------------------------------------------------
    # Split tied profiles separately.
    #
    # They have no valid majority label and will later be
    # removed only by Experiment 3.
    #
    # Experiments 1 and 2 still retain them.
    # ---------------------------------------------------------
    if len(tied_profiles) >= 2:
        (
            tied_train,
            tied_test,
        ) = train_test_split(
            tied_profiles,
            test_size=test_size,
            random_state=random_state,
        )

    elif len(tied_profiles) == 1:
        # Keep the single tied profile in train.
        tied_train = tied_profiles.copy()

        tied_test = tied_profiles.iloc[
            0:0
        ].copy()

    else:
        tied_train = tied_profiles.copy()
        tied_test = tied_profiles.copy()

    train_profiles = pd.concat(
        [
            non_tied_train,
            tied_train,
        ],
        ignore_index=True,
    )

    test_profiles = pd.concat(
        [
            non_tied_test,
            tied_test,
        ],
        ignore_index=True,
    )

    train_profiles = train_profiles[
        feature_columns
    ]

    test_profiles = test_profiles[
        feature_columns
    ]

    # ---------------------------------------------------------
    # Recover all original observations belonging to each
    # profile partition.
    # ---------------------------------------------------------
    all_row_keys = pd.MultiIndex.from_frame(
        df[feature_columns]
    )

    train_profile_keys = (
        pd.MultiIndex.from_frame(
            train_profiles
        )
    )

    test_profile_keys = (
        pd.MultiIndex.from_frame(
            test_profiles
        )
    )

    train_mask = all_row_keys.isin(
        train_profile_keys
    )

    test_mask = all_row_keys.isin(
        test_profile_keys
    )

    train_df = (
        df.loc[train_mask]
        .reset_index(drop=True)
    )

    test_df = (
        df.loc[test_mask]
        .reset_index(drop=True)
    )

    verify_profile_separation(
        train_df=train_df,
        test_df=test_df,
        target_column=target_column,
    )

    print(
        "\n--- Common Profile Split ---"
    )

    print(
        f"Unique feature profiles: "
        f"Train = {len(train_profiles)}, "
        f"Test = {len(test_profiles)}"
    )

    print(
        f"Raw observations: "
        f"Train = {len(train_df)}, "
        f"Test = {len(test_df)}"
    )

    print(
        "Verified: no feature signature "
        "appears in both train and test."
    )

    return train_df, test_df


def verify_profile_separation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_column: str,
) -> None:
    """
    Verify that no identical feature signature occurs
    in both train and test.
    """
    feature_columns = [
        column
        for column in train_df.columns
        if column != target_column
        and column != "sample_weight"
    ]

    train_profiles = (
        train_df[
            feature_columns
        ]
        .drop_duplicates()
    )

    test_profiles = (
        test_df[
            feature_columns
        ]
        .drop_duplicates()
    )

    train_keys = pd.MultiIndex.from_frame(
        train_profiles
    )

    test_keys = pd.MultiIndex.from_frame(
        test_profiles
    )

    overlap = train_keys.intersection(
        test_keys
    )

    if len(overlap) > 0:
        raise RuntimeError(
            "Train/test profile leakage detected: "
            f"{len(overlap)} feature signatures "
            "occur in both partitions."
        )


def build_standard_deduplicated_dataset(
    df: pd.DataFrame,
    target_column: str,
) -> pd.DataFrame:
    """
    Remove only exact duplicates.

    Rows are considered duplicates only when both
    their feature values and target labels are equal.

    Identical feature profiles carrying different labels
    are therefore retained.
    """
    if target_column not in df.columns:
        raise ValueError(
            f"Target column "
            f"'{target_column}' "
            "is missing."
        )

    return (
        df.drop_duplicates(
            keep="first"
        )
        .reset_index(drop=True)
    )


def deduplicate_with_weights(
    df: pd.DataFrame,
    target_column: str,
) -> pd.DataFrame:
    """
    Collapse identical feature signatures for Experiment 3.

    Rules:
    - no conflict:
        keep the unique label and use its frequency
        as sample_weight;

    - strict label majority:
        keep the majority label and use only the
        majority support as sample_weight;

    - exact tie:
        remove the profile.
    """
    feature_columns = [
        column
        for column in df.columns
        if column != target_column
    ]

    rows: list[dict] = []

    dropped_ties = 0
    resolved_conflicts = 0

    for _, group in df.groupby(
        feature_columns,
        dropna=False,
        sort=False,
    ):
        counts = (
            group[target_column]
            .value_counts()
        )

        # No conflict.
        if len(counts) == 1:
            target = counts.index[0]
            weight = counts.iloc[0]

            row = (
                group.iloc[0][
                    feature_columns
                ]
                .to_dict()
            )

            row[target_column] = target
            row["sample_weight"] = weight

            rows.append(row)

            continue

        top_count = counts.iloc[0]
        second_count = counts.iloc[1]

        # Exact tie -> discard ambiguous profile.
        if top_count == second_count:
            dropped_ties += 1
            continue

        # Strict majority.
        majority_target = counts.index[0]

        row = (
            group.iloc[0][
                feature_columns
            ]
            .to_dict()
        )

        row[target_column] = (
            majority_target
        )

        # Weight represents retained majority support,
        # not the complete original profile mass.
        row["sample_weight"] = (
            top_count
        )

        rows.append(row)

        resolved_conflicts += 1

    unique_df = pd.DataFrame(
        rows
    )

    if unique_df.empty:
        raise ValueError(
            "Weighted deduplication removed "
            "all feature profiles."
        )

    unique_df[target_column] = (
        unique_df[target_column]
        .astype(int)
    )

    unique_df["sample_weight"] = (
        unique_df["sample_weight"]
        .astype(int)
    )

    if resolved_conflicts > 0:
        print(
            f"Resolved "
            f"{resolved_conflicts} "
            "conflicting profiles "
            "via majority vote."
        )

    if dropped_ties > 0:
        print(
            f"Dropped "
            f"{dropped_ties} "
            "perfectly tied profiles "
            "as ambiguous."
        )

    return unique_df.reset_index(
        drop=True
    )


def prepare_experiment_1(
    train_source: pd.DataFrame,
    test_source: pd.DataFrame,
    output_dir: Path,
    target_column: str,
) -> tuple[Path, Path]:
    """
    Prepare Experiment 1:
    raw observations without deduplication.
    """
    print(
        "\nPreparing EXPERIMENT 1 "
        "- Raw Data"
    )

    train_df = train_source.copy()
    test_df = test_source.copy()

    verify_profile_separation(
        train_df,
        test_df,
        target_column,
    )

    train_path = (
        output_dir
        / RAW_TRAIN_FILENAME
    )

    test_path = (
        output_dir
        / RAW_TEST_FILENAME
    )

    train_df.to_csv(
        train_path,
        index=False,
    )

    test_df.to_csv(
        test_path,
        index=False,
    )

    print(
        f"Train rows: {len(train_df)}"
    )

    print(
        f"Test rows: {len(test_df)}"
    )

    print(
        f"Saved: {train_path}"
    )

    print(
        f"Saved: {test_path}"
    )

    return train_path, test_path


def prepare_experiment_2(
    train_source: pd.DataFrame,
    test_source: pd.DataFrame,
    output_dir: Path,
    target_column: str,
) -> tuple[Path, Path]:
    """
    Prepare Experiment 2:
    exact feature+target deduplication.
    """
    print(
        "\nPreparing EXPERIMENT 2 "
        "- Exact Deduplication"
    )

    train_df = (
        build_standard_deduplicated_dataset(
            train_source,
            target_column,
        )
    )

    test_df = (
        build_standard_deduplicated_dataset(
            test_source,
            target_column,
        )
    )

    verify_profile_separation(
        train_df,
        test_df,
        target_column,
    )

    train_path = (
        output_dir
        / STANDARD_DEDUP_TRAIN_FILENAME
    )

    test_path = (
        output_dir
        / STANDARD_DEDUP_TEST_FILENAME
    )

    train_df.to_csv(
        train_path,
        index=False,
    )

    test_df.to_csv(
        test_path,
        index=False,
    )

    print(
        f"Train rows before deduplication: "
        f"{len(train_source)}"
    )

    print(
        f"Train rows after deduplication: "
        f"{len(train_df)}"
    )

    print(
        f"Train exact duplicates removed: "
        f"{len(train_source) - len(train_df)}"
    )

    print(
        f"Test rows before deduplication: "
        f"{len(test_source)}"
    )

    print(
        f"Test rows after deduplication: "
        f"{len(test_df)}"
    )

    print(
        f"Test exact duplicates removed: "
        f"{len(test_source) - len(test_df)}"
    )

    print(
        f"Saved: {train_path}"
    )

    print(
        f"Saved: {test_path}"
    )

    return train_path, test_path


def prepare_experiment_3(
    train_source: pd.DataFrame,
    test_source: pd.DataFrame,
    output_dir: Path,
    target_column: str,
) -> tuple[Path, Path]:
    """
    Prepare Experiment 3:
    majority-vote deduplication with retained-support weights.
    """
    print(
        "\nPreparing EXPERIMENT 3 "
        "- Weighted Deduplication"
    )

    print(
        "\nProcessing training partition..."
    )

    train_df = deduplicate_with_weights(
        train_source,
        target_column,
    )

    print(
        "\nProcessing test partition..."
    )

    test_df = deduplicate_with_weights(
        test_source,
        target_column,
    )

    verify_profile_separation(
        train_df,
        test_df,
        target_column,
    )

    train_path = (
        output_dir
        / WEIGHTED_TRAIN_FILENAME
    )

    test_path = (
        output_dir
        / WEIGHTED_TEST_FILENAME
    )

    train_df.to_csv(
        train_path,
        index=False,
    )

    test_df.to_csv(
        test_path,
        index=False,
    )

    print(
        "\n--- Weighted Dataset Summary ---"
    )

    print(
        f"Train retained profiles: "
        f"{len(train_df)}"
    )

    print(
        f"Test retained profiles: "
        f"{len(test_df)}"
    )

    print(
        f"Train retained weighted mass: "
        f"{train_df['sample_weight'].sum()}"
    )

    print(
        f"Test retained weighted mass: "
        f"{test_df['sample_weight'].sum()}"
    )

    print(
        f"Saved: {train_path}"
    )

    print(
        f"Saved: {test_path}"
    )

    return train_path, test_path


def parse_arguments() -> argparse.Namespace:
    """
    Read the experiment to prepare.

    Each execution prepares exactly one experiment.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Prepare train/test datasets "
            "for one Data Mining experiment."
        )
    )

    parser.add_argument(
        "experiment",
        choices=[
            "1",
            "2",
            "3",
        ],
        help=(
            "Experiment to prepare: "
            "1 = raw, "
            "2 = exact deduplication, "
            "3 = weighted deduplication."
        ),
    )

    parser.add_argument(
        "--input",
        default=(
            "data/Training Dataset.arff"
        ),
        help=(
            "Path to the original ARFF dataset."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default="data",
        help=(
            "Directory where the selected "
            "experiment CSV files are saved."
        ),
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help=(
            "Fraction of feature profiles "
            "assigned to the final test set."
        ),
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help=(
            "Random seed used for the "
            "common profile split."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    if not 0.0 < args.test_size < 1.0:
        raise ValueError(
            "--test-size must be "
            "between 0 and 1."
        )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Load exactly the same raw source every time.
    df, target_column = (
        load_source_dataset(
            args.input
        )
    )

    # This deterministic split is recomputed independently
    # for each run. Because the source data, random_state,
    # and algorithm are identical, every experiment receives
    # exactly the same feature-profile partition.
    train_source, test_source = (
        build_common_profile_split(
            df=df,
            target_column=target_column,
            test_size=args.test_size,
            random_state=args.random_state,
        )
    )

    if args.experiment == "1":
        prepare_experiment_1(
            train_source=train_source,
            test_source=test_source,
            output_dir=output_dir,
            target_column=target_column,
        )

    elif args.experiment == "2":
        prepare_experiment_2(
            train_source=train_source,
            test_source=test_source,
            output_dir=output_dir,
            target_column=target_column,
        )

    elif args.experiment == "3":
        prepare_experiment_3(
            train_source=train_source,
            test_source=test_source,
            output_dir=output_dir,
            target_column=target_column,
        )

    else:
        raise RuntimeError(
            "Unexpected experiment identifier."
        )

    print(
        "\nDataset preparation completed "
        f"for Experiment {args.experiment}."
    )


if __name__ == "__main__":
    main()