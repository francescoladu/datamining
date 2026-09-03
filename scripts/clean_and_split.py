import os
import numpy as np
import pandas as pd
from scipy.io import arff

def balanced_group_split(df, group_col, test_size=0.2, random_state=42, tolerance=0.02):
    np.random.seed(random_state)

    group_sizes = df[group_col].value_counts().to_dict()
    group_ids = list(group_sizes.keys())

    np.random.shuffle(group_ids)                          
    group_ids.sort(key=lambda g: group_sizes[g], reverse=True) 

    train_groups, test_groups = [], []
    train_samples, test_samples = 0, 0
    total_samples = len(df)
    target_test_samples = int(total_samples * test_size)

    for gid in group_ids:
        size = group_sizes[gid]
        if test_samples + size <= target_test_samples:
            test_groups.append(gid)
            test_samples += size
        else:
            train_groups.append(gid)
            train_samples += size

    train_df = df[df[group_col].isin(train_groups)].reset_index(drop=True)
    test_df = df[df[group_col].isin(test_groups)].reset_index(drop=True)

    actual_test_frac = len(test_df) / total_samples
    print(f"Target test rows: {target_test_samples}")
    print(f"Actual split: Train={len(train_df)} ({len(train_df)/total_samples:.2%}), "
          f"Test={len(test_df)} ({actual_test_frac:.2%})")

    if abs(actual_test_frac - test_size) > tolerance:
        print(f"Warning: split deviates from target by more than {tolerance:.0%}. "
              f"Consider checking for a dominant group size or using a different random_state.")

    return train_df, test_df


def clean_and_split_by_signature(input_path, output_dir, test_size=0.2, random_state=42):
    print(f"Loading ARFF file from: {input_path}")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Source file not found at {input_path}.")

    raw_data, meta = arff.loadarff(input_path)
    df = pd.DataFrame(raw_data)

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.decode('utf-8').astype(int)
        else:
            df[col] = df[col].astype(int)

    id_cols = [col for col in df.columns if col.lower() in ['id', 'index', 'idx']]
    if id_cols:
        df = df.drop(columns=id_cols)

    possible_targets = ['Result', 'result', 'class', 'Class']
    target_col = next((col for col in possible_targets if col in df.columns), None)
    if target_col is None:
        raise ValueError("Could not identify target column.")

    feature_cols = [col for col in df.columns if col != target_col]
    df['signature_id'] = df.groupby(feature_cols).ngroup()

    train_df, test_df = balanced_group_split(df, 'signature_id', test_size=test_size, random_state=random_state)

    train_df = train_df.drop(columns=['signature_id'])
    test_df = test_df.drop(columns=['signature_id'])

    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train_grouped.csv")
    test_path = os.path.join(output_dir, "test_grouped.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Split complete. Train set size: {len(train_df)}, Test set size: {len(test_df)}")
    print("No identical feature vectors exist across the train and test split boundary.")


if __name__ == "__main__":
    INPUT_FILE = "data/Training Dataset.arff"
    OUTPUT_FOLDER = "data"
    clean_and_split_by_signature(INPUT_FILE, OUTPUT_FOLDER)