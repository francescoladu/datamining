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
    """Load the original ARFF dataset."""

    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Source file not found: {input_path}"
        )

    print(f"Loading dataset from: {input_path}")

    raw_data, _ = arff.loadarff(input_path)
    df = pd.DataFrame(raw_data)

    # Decode ARFF byte columns.
    for column in df.columns:
        if df[column].dtype == object:
            df[column] = (
                df[column]
                .str.decode("utf-8")
                .astype(int)
            )
        else:
            df[column] = df[column].astype(int)

    # Remove possible identifier columns.
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
        df = df.drop(columns=id_columns)

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

    if target_column != "Result":
        df = df.rename(
            columns={
                target_column: "Result"
            }
        )
        target_column = "Result"

    df = df.reset_index(drop=True)

    print(f"Rows loaded: {len(df)}")
    print(f"Target column: {target_column}")

    return df, target_column


def split_dataset(
    df: pd.DataFrame,
    target_column: str,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Perform a standard stratified train/test split."""

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[target_column],
    )

    return (
        train_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def build_standard_deduplicated_dataset(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Remove exact duplicates.

    A duplicate must have both the same feature values
    and the same target label.
    """

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


def save_dataset(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
    train_filename: str,
    test_filename: str,
) -> None:
    """Save train and test CSV files."""

    train_path = output_dir / train_filename
    test_path = output_dir / test_filename

    train_df.to_csv(
        train_path,
        index=False,
    )

    test_df.to_csv(
        test_path,
        index=False,
    )

    print(f"Train rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")
    print(f"Saved: {train_path}")
    print(f"Saved: {test_path}")


def prepare_experiment_1(
    df: pd.DataFrame,
    target_column: str,
    output_dir: Path,
    test_size: float,
    random_state: int,
) -> None:
    """
    Experiment 1:
    raw dataset followed directly by train/test split.
    """

    print("\nEXPERIMENT 1 - RAW DATA")

    train_df, test_df = split_dataset(
        df=df,
        target_column=target_column,
        test_size=test_size,
        random_state=random_state,
    )

    save_dataset(
        train_df=train_df,
        test_df=test_df,
        output_dir=output_dir,
        train_filename=RAW_TRAIN_FILENAME,
        test_filename=RAW_TEST_FILENAME,
    )


def prepare_experiment_2(
    df: pd.DataFrame,
    target_column: str,
    output_dir: Path,
    test_size: float,
    random_state: int,
) -> None:
    """
    Experiment 2:
    exact deduplication first, then train/test split.
    """

    print("\nEXPERIMENT 2 - EXACT DEDUPLICATION")

    before = len(df)

    deduplicated_df = (
        build_standard_deduplicated_dataset(
            df
        )
    )

    after = len(deduplicated_df)

    print(f"Rows before deduplication: {before}")
    print(f"Rows after deduplication: {after}")
    print(f"Exact duplicates removed: {before - after}")

    train_df, test_df = split_dataset(
        df=deduplicated_df,
        target_column=target_column,
        test_size=test_size,
        random_state=random_state,
    )

    save_dataset(
        train_df=train_df,
        test_df=test_df,
        output_dir=output_dir,
        train_filename=STANDARD_DEDUP_TRAIN_FILENAME,
        test_filename=STANDARD_DEDUP_TEST_FILENAME,
    )


def prepare_experiment_3(
    df: pd.DataFrame,
    target_column: str,
    output_dir: Path,
    test_size: float,
    random_state: int,
) -> None:
    """
    Experiment 3:
    weighted majority-vote deduplication first,
    then train/test split.
    """

    print("\nEXPERIMENT 3 - WEIGHTED DEDUPLICATION")

    before = len(df)

    weighted_df = deduplicate_with_weights(
        df=df,
        target_column=target_column,
    )

    print(f"Rows before deduplication: {before}")
    print(
        f"Profiles after weighted deduplication: "
        f"{len(weighted_df)}"
    )
    print(
        f"Retained weighted mass: "
        f"{weighted_df['sample_weight'].sum()}"
    )

    train_df, test_df = split_dataset(
        df=weighted_df,
        target_column=target_column,
        test_size=test_size,
        random_state=random_state,
    )

    print(
        f"Train retained weighted mass: "
        f"{train_df['sample_weight'].sum()}"
    )

    print(
        f"Test retained weighted mass: "
        f"{test_df['sample_weight'].sum()}"
    )

    save_dataset(
        train_df=train_df,
        test_df=test_df,
        output_dir=output_dir,
        train_filename=WEIGHTED_TRAIN_FILENAME,
        test_filename=WEIGHTED_TEST_FILENAME,
    )


def parse_arguments() -> argparse.Namespace:
    """Read command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Prepare one Data Mining experiment."
        )
    )

    parser.add_argument(
        "experiment",
        choices=["1", "2", "3"],
        help=(
            "1 = raw data, "
            "2 = exact deduplication, "
            "3 = weighted deduplication"
        ),
    )

    parser.add_argument(
        "--input",
        default="data/Training Dataset.arff",
    )

    parser.add_argument(
        "--output-dir",
        default="data",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    if not 0.0 < args.test_size < 1.0:
        raise ValueError(
            "--test-size must be between 0 and 1."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    df, target_column = load_source_dataset(
        args.input
    )

    if args.experiment == "1":
        prepare_experiment_1(
            df=df,
            target_column=target_column,
            output_dir=output_dir,
            test_size=args.test_size,
            random_state=args.random_state,
        )

    elif args.experiment == "2":
        prepare_experiment_2(
            df=df,
            target_column=target_column,
            output_dir=output_dir,
            test_size=args.test_size,
            random_state=args.random_state,
        )

    elif args.experiment == "3":
        prepare_experiment_3(
            df=df,
            target_column=target_column,
            output_dir=output_dir,
            test_size=args.test_size,
            random_state=args.random_state,
        )

    print(
        f"\nExperiment {args.experiment} "
        "dataset preparation completed."
    )


if __name__ == "__main__":
    main()