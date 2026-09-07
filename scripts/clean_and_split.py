import os
import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split


def deduplicate_with_weights(df, target_col):
    feature_cols = [col for col in df.columns if col != target_col]
    
    rows = []
    dropped_ties = 0
    resolved_conflicts = 0

    for _, group in df.groupby(feature_cols):
        counts = group[target_col].value_counts()
        
        # No conflict
        if len(counts) == 1:
            target = counts.index[0]
            weight = counts.iloc[0]
            row = group.iloc[0][feature_cols].to_dict()
            row[target_col] = target
            row['sample_weight'] = weight
            rows.append(row)

        # Conflict exists
        else:
            top_count = counts.iloc[0]
            second_count = counts.iloc[1]
            
            # Exact tie (e.g., 1 vs 1, 2 vs 2) -> DROP as pure noise
            if top_count == second_count:
                dropped_ties += 1
                continue
            
            # Clear majority (e.g., 4 vs 1) -> KEEP majority label and its count
            majority_target = counts.index[0]
            row = group.iloc[0][feature_cols].to_dict()
            row[target_col] = majority_target
            row['sample_weight'] = top_count  # Weight based on true majority support
            rows.append(row)
            resolved_conflicts += 1

    unique_df = pd.DataFrame(rows)
    
    # Ensure types
    unique_df[target_col] = unique_df[target_col].astype(int)
    unique_df['sample_weight'] = unique_df['sample_weight'].astype(int)

    if resolved_conflicts > 0:
        print(f"Resolved {resolved_conflicts} conflicting profiles via majority vote.")
    if dropped_ties > 0:
        print(f"Dropped {dropped_ties} perfectly tied profiles (pure ambiguity/noise).")

    return unique_df


def clean_and_split_deduplicated(input_path, output_dir, test_size=0.2, random_state=42):
    print(f"Loading ARFF file from: {input_path}")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Source file not found at {input_path}.")

    # Load ARFF file
    raw_data, meta = arff.loadarff(input_path)
    df = pd.DataFrame(raw_data)

    # Decode bytes to integers
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.decode('utf-8').astype(int)
        else:
            df[col] = df[col].astype(int)

    # Drop index/ID columns if present
    id_cols = [col for col in df.columns if col.lower() in ['id', 'index', 'idx', 'unnamed: 0']]
    if id_cols:
        print(f"Removing identifier column(s): {id_cols}")
        df = df.drop(columns=id_cols)

    # Identify target column
    possible_targets = ['Result', 'result', 'class', 'Class']
    target_col = next((col for col in possible_targets if col in df.columns), None)
    if target_col is None:
        raise ValueError("Could not identify target column.")

    print(f"Target column identified: '{target_col}'")
    print(f"Original dataset size: {len(df)} rows")

    # Deduplicate and compute sample weights
    print("\nDeduplicating identical feature signatures and assigning sample weights...")
    unique_df = deduplicate_with_weights(df, target_col)
    print(f"Compressed into {len(unique_df)} unique feature profiles (Total weight: {unique_df['sample_weight'].sum()})")

    # Standard Stratified Split on the unique profiles
    print(f"\nSplitting unique profiles (test size: {test_size:.1%})...")
    train_df, test_df = train_test_split(
        unique_df,
        test_size=test_size,
        stratify=unique_df[target_col],
        random_state=random_state
    )

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Save output files (features + target + sample_weight)
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train_cleaned.csv")
    test_path = os.path.join(output_dir, "test_cleaned.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    # --- Verification Statistics ---
    print("\n--- Split Verification ---")
    print(f"Unique profiles: Train = {len(train_df)} ({len(train_df)/len(unique_df):.2%}), "
          f"Test = {len(test_df)} ({len(test_df)/len(unique_df):.2%})")
    print(f"Weighted instances: Train = {train_df['sample_weight'].sum()}, "
          f"Test = {test_df['sample_weight'].sum()}")

    # Check unweighted vs weighted class distribution
    print("\nClass Distributions (Unweighted / Unique Profiles):")
    for label in sorted(unique_df[target_col].unique()):
        orig_pct = (unique_df[target_col] == label).mean()
        train_pct = (train_df[target_col] == label).mean()
        test_pct = (test_df[target_col] == label).mean()
        print(f"  Class '{label}': Overall={orig_pct:.2%}, Train={train_pct:.2%}, Test={test_pct:.2%}")

    print("\nClass Distributions (Weighted / True Instance Mass):")
    total_w = unique_df['sample_weight'].sum()
    train_w = train_df['sample_weight'].sum()
    test_w = test_df['sample_weight'].sum()
    for label in sorted(unique_df[target_col].unique()):
        orig_wpct = unique_df.loc[unique_df[target_col] == label, 'sample_weight'].sum() / total_w
        train_wpct = train_df.loc[train_df[target_col] == label, 'sample_weight'].sum() / train_w
        test_wpct = test_df.loc[test_df[target_col] == label, 'sample_weight'].sum() / test_w
        print(f"  Class '{label}': Overall={orig_wpct:.2%}, Train={train_wpct:.2%}, Test={test_wpct:.2%}")

    print("\nFiles saved successfully with 'sample_weight' column included.")


if __name__ == "__main__":
    INPUT_FILE = "data/Training Dataset.arff"
    OUTPUT_FOLDER = "data"
    clean_and_split_deduplicated(INPUT_FILE, OUTPUT_FOLDER)