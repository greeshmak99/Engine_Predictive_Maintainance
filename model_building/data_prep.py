"""
Data Preparation Script - Production Version
Loads pre-split data from Hugging Face or re-splits if needed
"""

import os
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi, login

# Configuration
DATASET_REPO_ID = "Quantum9999/engine-predictive-maintenance"
TARGET_COLUMN = "Engine Condition"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Control flags
USE_PRESPLIT_DATA = os.environ.get("USE_PRESPLIT_DATA", "true").lower() == "true"
UPLOAD_TO_HF = os.environ.get("UPLOAD_DATA_TO_HF", "false").lower() == "true"


def authenticate_hf():
    """Authenticate with Hugging Face"""
    print("=" * 70)
    print("AUTHENTICATING WITH HUGGING FACE")
    print("=" * 70)
    
    hf_token = os.environ.get("HF_EN_TOKEN")
    if not hf_token:
        raise EnvironmentError("HF_EN_TOKEN environment variable not found")
    
    login(token=hf_token)
    print("✓ Successfully authenticated\n")
    return hf_token


def load_presplit_data():
    """Load already-split train and test data from Hugging Face"""
    print("=" * 70)
    print("LOADING PRE-SPLIT DATA FROM HUGGING FACE")
    print("=" * 70)
    
    # Load train data
    train_dataset = load_dataset(
        DATASET_REPO_ID,
        data_files="train.csv",
        split="train"
    )
    train_df = train_dataset.to_pandas()
    
    # Load test data
    test_dataset = load_dataset(
        DATASET_REPO_ID,
        data_files="test.csv",
        split="train"
    )
    test_df = test_dataset.to_pandas()
    
    print(f"✓ Pre-split data loaded successfully")
    print(f"  Train shape: {train_df.shape}")
    print(f"  Test shape: {test_df.shape}")
    print(f"\n  Train target distribution:")
    print(f"{train_df[TARGET_COLUMN].value_counts()}")
    print(f"\n  Test target distribution:")
    print(f"{test_df[TARGET_COLUMN].value_counts()}\n")
    
    return train_df, test_df


def load_and_split_raw_data():
    """Load raw data and perform fresh split (for data updates)"""
    print("=" * 70)
    print("LOADING RAW DATA AND PERFORMING FRESH SPLIT")
    print("=" * 70)
    
    # Load full dataset
    dataset = load_dataset(DATASET_REPO_ID, split="train")
    df = dataset.to_pandas()
    
    print(f"✓ Raw dataset loaded")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {df.columns.tolist()}")
    
    print("\n" + "-" * 70)
    print("PERFORMING STRATIFIED SPLIT")
    print("-" * 70)
    
    # Stratified split
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        stratify=df[TARGET_COLUMN],
        random_state=RANDOM_STATE
    )
    
    print(f"✓ Split completed")
    print(f"  Train shape: {train_df.shape}")
    print(f"  Test shape: {test_df.shape}")
    print(f"\n  Train target distribution:")
    print(f"{train_df[TARGET_COLUMN].value_counts()}")
    print(f"\n  Test target distribution:")
    print(f"{test_df[TARGET_COLUMN].value_counts()}\n")
    
    return train_df, test_df


def save_datasets_locally(train_df, test_df):
    """Save train and test datasets locally"""
    print("=" * 70)
    print("SAVING DATASETS LOCALLY")
    print("=" * 70)
    
    os.makedirs("data", exist_ok=True)
    
    train_path = "data/train.csv"
    test_path = "data/test.csv"
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"✓ Datasets saved locally")
    print(f"  Train: {train_path}")
    print(f"  Test: {test_path}\n")
    
    return train_path, test_path


def upload_to_huggingface(train_path, test_path, hf_token):
    """Upload datasets to Hugging Face"""
    if not UPLOAD_TO_HF:
        print("=" * 70)
        print("SKIPPING UPLOAD TO HUGGING FACE")
        print("=" * 70)
        print("ℹ  Using existing split on Hugging Face")
        print("ℹ  Set UPLOAD_DATA_TO_HF=true to upload new split\n")
        return
    
    print("=" * 70)
    print("UPLOADING TO HUGGING FACE")
    print("=" * 70)
    
    api = HfApi()
    
    api.upload_file(
        path_or_fileobj=train_path,
        path_in_repo="train.csv",
        repo_id=DATASET_REPO_ID,
        repo_type="dataset",
        token=hf_token
    )
    print("✓ Train dataset uploaded")
    
    api.upload_file(
        path_or_fileobj=test_path,
        path_in_repo="test.csv",
        repo_id=DATASET_REPO_ID,
        repo_type="dataset",
        token=hf_token
    )
    print("✓ Test dataset uploaded\n")


def main():
    """Main execution pipeline"""
    print("\n" + "=" * 70)
    print("DATA PREPARATION PIPELINE")
    print("=" * 70)
    print(f"Use pre-split data: {USE_PRESPLIT_DATA}")
    print(f"Upload to HF: {UPLOAD_TO_HF}")
    print("=" * 70 + "\n")
    
    # Authenticate
    hf_token = authenticate_hf()
    
    # Choose loading strategy
    if USE_PRESPLIT_DATA:
        # Load existing train/test from HF (EFFICIENT - Real-world approach)
        train_df, test_df = load_presplit_data()
    else:
        # Load raw data and re-split (Only when data is updated)
        train_df, test_df = load_and_split_raw_data()
    
    # Save locally for pipeline
    train_path, test_path = save_datasets_locally(train_df, test_df)
    
    # Upload to HF (only if needed)
    upload_to_huggingface(train_path, test_path, hf_token)
    
    print("=" * 70)
    print("DATA PREPARATION COMPLETED SUCCESSFULLY")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
