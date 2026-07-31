import os
import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split

def clean_and_split_dataset(input_path, output_dir, test_size=0.2, random_state=42):
    """
    Loads the UCI Phishing Websites ARFF file, decodes bytes, cleans 
    potential leakage sources (duplicates/index columns), and outputs
    clean, stratified train and test CSV files.
    """
    print(f"Loading ARFF file from: {input_path}")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Source file not found at {input_path}. Run download target first.")

    # 1. Load the ARFF file
    raw_data, meta = arff.loadarff(input_path)
    df = pd.DataFrame(raw_data)

    # 2. Decode byte-strings into integers
    print("Decoding byte-strings to standard integers...")
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.decode('utf-8').astype(int)
        else:
            df[col] = df[col].astype(int)

    # 3. Drop index or sequential ID columns if present
    id_cols = [col for col in df.columns if col.lower() in ['id', 'index', 'idx', 'unnamed: 0']]
    if id_cols:
        print(f"Removing identifier column(s) to avoid leakage: {id_cols}")
        df = df.drop(columns=id_cols)
    else:
        print("No index/ID column detected in raw file (ideal for preventing index leakage).")

    # 4. Remove Duplicate Rows
    initial_rows = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    duplicates_removed = initial_rows - len(df)
    print(f"Removed {duplicates_removed} duplicate row(s). Total unique rows: {len(df)}")

    # 5. Locate target column
    possible_targets = ['Result', 'result', 'class', 'Class']
    target_col = None
    for col in possible_targets:
        if col in df.columns:
            target_col = col
            break
            
    if target_col is None:
        raise ValueError(f"Could not identify the target column. Columns present: {list(df.columns)}")
        
    print(f"Target column identified: '{target_col}'")
    
    # 6. Split features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 7. Stratified Train/Test Split
    # Stratification ensures that both subsets maintain the same class ratio
    print(f"Splitting data (test size: {test_size:.1%}, stratified on '{target_col}')...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        stratify=y, 
        random_state=random_state
    )

    # Recombine features and targets for saving
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    # 8. Save output files
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train_cleaned.csv")
    test_path = os.path.join(output_dir, "test_cleaned.csv")
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Cleaned training set saved to: {train_path} ({len(train_df)} samples)")
    print(f"Cleaned test set saved to: {test_path} ({len(test_df)} samples)")
    print("Preprocessing completed successfully.")

if __name__ == "__main__":
    INPUT_FILE = "data/Training Dataset.arff"
    OUTPUT_FOLDER = "data"
    
    clean_and_split_dataset(INPUT_FILE, OUTPUT_FOLDER)