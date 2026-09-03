import os
import numpy as np
import pandas as pd
from scipy.io import arff

def stratified_group_split(df, group_col, target_col, test_size=0.2, random_state=42):
    """
    Splits the dataset into train and test sets while satisfying three constraints:
    1. Zero Leakage: Identical feature signatures are never split across train/test.
    2. No Sorting Bias: Group IDs are shuffled purely randomly.
    3. Stratification: The target class distribution is preserved in both splits.
    """
    np.random.seed(random_state)

    # 1. Map each unique signature group to its target class and its row count
    group_meta = df.groupby(group_col).agg(
        target=(target_col, 'first'),
        size=(group_col, 'size')
    ).reset_index()

    train_groups = []
    test_groups = []

    # 2. Perform the group split independently for each class to guarantee stratification
    unique_classes = group_meta['target'].unique()
    for cls in unique_classes:
        # Filter groups belonging strictly to the current class
        cls_groups = group_meta[group_meta['target'] == cls].copy()
        
        # Purely randomize the group IDs (avoids size-sorting bias)
        cls_group_ids = cls_groups[group_col].tolist()
        np.random.shuffle(cls_group_ids)
        
        cls_group_sizes = cls_groups.set_index(group_col)['size'].to_dict()
        cls_total_rows = cls_groups['size'].sum()
        cls_target_test_rows = int(cls_total_rows * test_size)
        
        cls_test_samples = 0
        cls_test_groups = []
        cls_train_groups = []
        
        # 3. Randomly allocate groups to the test set for this class until we hit the 20% mark
        for gid in cls_group_ids:
            size = cls_group_sizes[gid]
            if cls_test_samples < cls_target_test_rows:
                cls_test_groups.append(gid)
                cls_test_samples += size
            else:
                cls_train_groups.append(gid)
                
        train_groups.extend(cls_train_groups)
        test_groups.extend(cls_test_groups)

    # 4. Filter the original dataframe based on the assigned groups
    train_df = df[df[group_col].isin(train_groups)].reset_index(drop=True)
    test_df = df[df[group_col].isin(test_groups)].reset_index(drop=True)

    return train_df, test_df


def clean_and_split_by_signature(input_path, output_dir, test_size=0.2, random_state=42):
    print(f"Loading ARFF file from: {input_path}")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Source file not found at {input_path}.")

    # Load ARFF file
    raw_data, meta = arff.loadarff(input_path)
    df = pd.DataFrame(raw_data)

    # Decode bytes
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.decode('utf-8').astype(int)
        else:
            df[col] = df[col].astype(int)

    # Drop ID columns
    id_cols = [col for col in df.columns if col.lower() in ['id', 'index', 'idx', 'unnamed: 0']]
    if id_cols:
        print(f"Removing identifier column(s): {id_cols}")
        df = df.drop(columns=id_cols)

    # Locate target column
    possible_targets = ['Result', 'result', 'class', 'Class']
    target_col = next((col for col in possible_targets if col in df.columns), None)
    if target_col is None:
        raise ValueError("Could not identify target column.")

    print(f"Target column identified: '{target_col}'")

    # Define features and generate unique signature IDs
    feature_cols = [col for col in df.columns if col != target_col]
    df['signature_id'] = df.groupby(feature_cols).ngroup()

    # Split using the new stratified group logic
    print(f"Splitting data with stratified group logic (test size: {test_size:.1%})...")
    train_df, test_df = stratified_group_split(
        df, 
        group_col='signature_id', 
        target_col=target_col, 
        test_size=test_size, 
        random_state=random_state
    )

    # Remove the temporary signature column before saving
    train_df = train_df.drop(columns=['signature_id'])
    test_df = test_df.drop(columns=['signature_id'])

    # Save output files
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train_cleaned.csv")
    test_path = os.path.join(output_dir, "test_cleaned.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    # Verify class balance
    print("\n--- Split Verification ---")
    print(f"Total rows: {len(df)}")
    print(f"Train set: {len(train_df)} rows ({len(train_df)/len(df):.2%})")
    print(f"Test set:  {len(test_df)} rows ({len(test_df)/len(df):.2%})")
    
    # Print out class distributions to verify stratification
    for label in sorted(df[target_col].unique()):
        orig_pct = (df[target_col] == label).mean()
        train_pct = (train_df[target_col] == label).mean()
        test_pct = (test_df[target_col] == label).mean()
        print(f"Class '{label}': Original={orig_pct:.2%}, Train={train_pct:.2%}, Test={test_pct:.2%}")
        
    print("\nProcessing completed successfully. No identical signatures cross the split boundary.")


if __name__ == "__main__":
    INPUT_FILE = "data/Training Dataset.arff"
    OUTPUT_FOLDER = "data"
    clean_and_split_by_signature(INPUT_FILE, OUTPUT_FOLDER)