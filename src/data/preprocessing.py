"""
Data preprocessing functions for the malware dataset.
"""

import os
import pandas as pd
import numpy as np
import logging
import zipfile
import shutil
from pathlib import Path
import hashlib
from tqdm import tqdm


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('preprocessing.log')
        ]
    )
    return logging.getLogger(__name__)


def extract_kaggle_dataset(zip_path, extract_path):
    """
    Extract the Kaggle malware dataset from a zip file.
    
    Parameters:
    -----------
    zip_path : str
        Path to the zip file
    extract_path : str
        Path to extract the files to
    
    Returns:
    --------
    bool
        Success status
    """
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Extracting {zip_path} to {extract_path}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get total files for progress tracking
            total_files = len(zip_ref.namelist())
            
            # Extract each file with progress tracking
            for i, file in enumerate(zip_ref.namelist(), 1):
                zip_ref.extract(file, extract_path)
                if i % 100 == 0 or i == total_files:
                    logger.info(f"Extracted {i}/{total_files} files")
        
        logger.info("Extraction complete")
        return True
    
    except Exception as e:
        logger.error(f"Error extracting dataset: {e}")
        return False


def compute_file_hash(file_path, algorithm='sha256'):
    """
    Compute hash for a file.
    
    Parameters:
    -----------
    file_path : str
        Path to the file
    algorithm : str
        Hash algorithm to use
    
    Returns:
    --------
    str
        Computed hash
    """
    if algorithm == 'md5':
        hasher = hashlib.md5()
    elif algorithm == 'sha1':
        hasher = hashlib.sha1()
    elif algorithm == 'sha256':
        hasher = hashlib.sha256()
    else:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")
    
    with open(file_path, 'rb') as f:
        # Read and update hash in chunks to avoid loading large files into memory
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    
    return hasher.hexdigest()


def create_dataset_catalog(data_dir, output_path):
    """
    Create a catalog of files in the dataset with hashes and basic metadata.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing the dataset files
    output_path : str
        Path to save the catalog CSV
    
    Returns:
    --------
    pd.DataFrame
        DataFrame containing the catalog
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Creating catalog for files in {data_dir}")
    
    # List to store file metadata
    file_metadata = []
    
    # Walk through directory
    for root, _, files in os.walk(data_dir):
        for file in tqdm(files, desc="Processing files"):
            file_path = os.path.join(root, file)
            
            try:
                # Get file stats
                file_stats = os.stat(file_path)
                
                # Compute hash
                file_hash = compute_file_hash(file_path)
                
                # Store metadata
                metadata = {
                    'file_path': file_path,
                    'file_name': file,
                    'directory': root,
                    'size_bytes': file_stats.st_size,
                    'sha256_hash': file_hash,
                    'last_modified': pd.Timestamp(file_stats.st_mtime, unit='s')
                }
                
                file_metadata.append(metadata)
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
    
    # Create DataFrame
    catalog_df = pd.DataFrame(file_metadata)
    
    # Save to CSV
    catalog_df.to_csv(output_path, index=False)
    logger.info(f"Catalog saved to {output_path}")
    
    return catalog_df


def label_dataset(catalog_path, labels_path, output_path):
    """
    Label the dataset using provided labels.
    
    Parameters:
    -----------
    catalog_path : str
        Path to the catalog CSV
    labels_path : str
        Path to the labels CSV
    output_path : str
        Path to save the labeled catalog
    
    Returns:
    --------
    pd.DataFrame
        DataFrame containing the labeled catalog
    """
    logger = logging.getLogger(__name__)
    logger.info("Labeling dataset")
    
    # Load catalog and labels
    catalog_df = pd.read_csv(catalog_path)
    labels_df = pd.read_csv(labels_path)
    
    # Merge catalog with labels
    # This assumes labels_df has columns that can be matched with catalog_df
    # Adjust the merge logic based on your actual labels format
    if 'sha256_hash' in labels_df.columns:
        merged_df = pd.merge(
            catalog_df,
            labels_df,
            on='sha256_hash',
            how='left'
        )
    elif 'file_name' in labels_df.columns:
        merged_df = pd.merge(
            catalog_df,
            labels_df,
            on='file_name',
            how='left'
        )
    else:
        logger.error("Cannot determine how to merge labels - no matching columns found")
        return catalog_df
    
    # Fill missing labels
    if 'is_malicious' not in merged_df.columns:
        logger.error("No 'is_malicious' column found in merged dataset")
        return catalog_df
    
    # Save labeled catalog
    merged_df.to_csv(output_path, index=False)
    logger.info(f"Labeled catalog saved to {output_path}")
    
    return merged_df


def split_dataset(labeled_catalog_path, output_dir, test_size=0.2, validation_size=0.1):
    """
    Split the dataset into training, validation, and test sets.
    
    Parameters:
    -----------
    labeled_catalog_path : str
        Path to the labeled catalog CSV
    output_dir : str
        Directory to save the split dataset CSVs
    test_size : float
        Fraction of data to use for testing
    validation_size : float
        Fraction of data to use for validation
    
    Returns:
    --------
    tuple
        (train_df, val_df, test_df) DataFrames
    """
    from sklearn.model_selection import train_test_split
    
    logger = logging.getLogger(__name__)
    logger.info("Splitting dataset into train/validation/test sets")
    
    # Load labeled catalog
    labeled_df = pd.read_csv(labeled_catalog_path)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # First split: separate test set
    train_val_df, test_df = train_test_split(
        labeled_df,
        test_size=test_size,
        stratify=labeled_df['is_malicious'],
        random_state=42
    )
    
    # Second split: separate validation set from training set
    # Adjust validation size to account for the reduced dataset size
    adjusted_val_size = validation_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=adjusted_val_size,
        stratify=train_val_df['is_malicious'],
        random_state=42
    )
    
    # Save splits
    train_path = os.path.join(output_dir, 'train.csv')
    val_path = os.path.join(output_dir, 'validation.csv')
    test_path = os.path.join(output_dir, 'test.csv')
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    logger.info(f"Train set: {len(train_df)} samples")
    logger.info(f"Validation set: {len(val_df)} samples")
    logger.info(f"Test set: {len(test_df)} samples")
    
    return train_df, val_df, test_df


def clean_data(raw_data_path, processed_data_path):
    """
    Clean and preprocess the raw malware dataset.
    
    Parameters:
    -----------
    raw_data_path : str
        Path to the raw data file
    processed_data_path : str
        Path to save the processed data file
    
    Returns:
    --------
    pd.DataFrame
        Preprocessed data
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Cleaning data from {raw_data_path}")
    
    # Load raw data
    df = pd.read_csv(raw_data_path)
    
    # Remove duplicate files based on hash
    if 'sha256_hash' in df.columns:
        df_no_dups = df.drop_duplicates(subset='sha256_hash')
        logger.info(f"Removed {len(df) - len(df_no_dups)} duplicate files")
        df = df_no_dups
    
    # Remove rows with missing values
    df_no_nulls = df.dropna()
    logger.info(f"Removed {len(df) - len(df_no_nulls)} rows with missing values")
    df = df_no_nulls
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
    
    # Save processed data
    df.to_csv(processed_data_path, index=False)
    logger.info(f"Processed data saved to {processed_data_path}")
    
    return df


def main():
    """Main function for data preprocessing."""
    # Set up logging
    logger = setup_logging()
    logger.info("Starting data preprocessing")
    
    # Create paths
    raw_dir = os.path.join('data', 'raw')
    processed_dir = os.path.join('data', 'processed')
    
    # Create directories if they don't exist
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    # Check if Kaggle dataset needs to be extracted
    kaggle_zip = os.path.join(raw_dir, 'malware_dataset.zip')
    if os.path.exists(kaggle_zip):
        extract_kaggle_dataset(kaggle_zip, raw_dir)
    
    # Create dataset catalog
    catalog_path = os.path.join(processed_dir, 'dataset_catalog.csv')
    create_dataset_catalog(raw_dir, catalog_path)
    
    # Check if labels file exists
    labels_path = os.path.join(raw_dir, 'labels.csv')
    if os.path.exists(labels_path):
        # Label the dataset
        labeled_path = os.path.join(processed_dir, 'labeled_dataset.csv')
        label_dataset(catalog_path, labels_path, labeled_path)
        
        # Split the dataset
        split_dataset(labeled_path, processed_dir)
    else:
        logger.warning(f"Labels file not found at {labels_path}. Skipping labeling and splitting.")
    
    logger.info("Data preprocessing complete")


if __name__ == "__main__":
    main()