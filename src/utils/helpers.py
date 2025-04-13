"""
Helper functions for the project.
"""

import os
import logging
import pandas as pd
import numpy as np
import time
import json
from pathlib import Path
import hashlib
import shutil
import tempfile
import zipfile
import requests
from datetime import datetime
from tqdm import tqdm


def setup_logging(log_file=None, level=logging.INFO):
    """
    Set up logging configuration.
    
    Parameters:
    -----------
    log_file : str, optional
        Path to log file
    level : int, optional
        Logging level
    
    Returns:
    --------
    logging.Logger
        Configured logger
    """
    # Create logger
    logger = logging.getLogger("enhanced_yara_ml")
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log file is provided
    if log_file:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def calculate_file_hash(file_path, algorithm='sha256'):
    """
    Calculate hash of a file.
    
    Parameters:
    -----------
    file_path : str
        Path to the file
    algorithm : str
        Hash algorithm to use ('md5', 'sha1', 'sha256')
    
    Returns:
    --------
    str
        Calculated hash
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
        # Read and update hash in chunks
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    
    return hasher.hexdigest()


def download_file(url, output_path, progress=True):
    """
    Download file from a URL with progress bar.
    
    Parameters:
    -----------
    url : str
        URL to download from
    output_path : str
        Path to save the downloaded file
    progress : bool
        Whether to display progress bar
    
    Returns:
    --------
    str
        Path to the downloaded file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Start download
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    # Get total file size
    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192  # 8 KB
    
    # Download with progress bar
    if progress:
        print(f"Downloading {url} to {output_path}")
        
        with open(output_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
    else:
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    f.write(chunk)
    
    return output_path


def extract_zip(zip_path, extract_path, progress=True):
    """
    Extract a zip file with progress bar.
    
    Parameters:
    -----------
    zip_path : str
        Path to the zip file
    extract_path : str
        Path to extract to
    progress : bool
        Whether to display progress bar
    
    Returns:
    --------
    str
        Path to the extracted directory
    """
    # Create directory if it doesn't exist
    os.makedirs(extract_path, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        if progress:
            print(f"Extracting {zip_path} to {extract_path}")
            
            # Get total number of files
            total_files = len(zip_ref.namelist())
            
            for i, member in enumerate(tqdm(zip_ref.namelist(), desc="Extracting", total=total_files)):
                zip_ref.extract(member, extract_path)
        else:
            zip_ref.extractall(extract_path)
    
    return extract_path


def get_file_paths(directory, recursive=False, extensions=None):
    """
    Get list of file paths in a directory.
    
    Parameters:
    -----------
    directory : str
        Directory to search
    recursive : bool
        Whether to search recursively
    extensions : list, optional
        List of file extensions to include
    
    Returns:
    --------
    list
        List of file paths
    """
    file_paths = []
    
    if recursive:
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                
                # Filter by extension if specified
                if extensions:
                    _, ext = os.path.splitext(file_path)
                    if ext.lower() in extensions:
                        file_paths.append(file_path)
                else:
                    file_paths.append(file_path)
    else:
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            
            if os.path.isfile(file_path):
                # Filter by extension if specified
                if extensions:
                    _, ext = os.path.splitext(file_path)
                    if ext.lower() in extensions:
                        file_paths.append(file_path)
                else:
                    file_paths.append(file_path)
    
    return file_paths


def save_json(data, output_path, indent=2):
    """
    Save data to a JSON file.
    
    Parameters:
    -----------
    data : dict or list
        Data to save
    output_path : str
        Path to save the JSON file
    indent : int, optional
        JSON indentation level
    
    Returns:
    --------
    str
        Path to the saved file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=indent)
    
    return output_path


def load_json(input_path):
    """
    Load data from a JSON file.
    
    Parameters:
    -----------
    input_path : str
        Path to the JSON file
    
    Returns:
    --------
    dict or list
        Loaded data
    """
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    return data


def save_to_csv(df, output_path, index=False):
    """
    Save DataFrame to a CSV file.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to save
    output_path : str
        Path to save the CSV file
    index : bool
        Whether to include index
    
    Returns:
    --------
    str
        Path to the saved file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df.to_csv(output_path, index=index)
    
    return output_path


def create_temp_directory():
    """
    Create a temporary directory.
    
    Returns:
    --------
    str
        Path to the temporary directory
    """
    return tempfile.mkdtemp()


def clean_temp_directory(temp_dir):
    """
    Clean up a temporary directory.
    
    Parameters:
    -----------
    temp_dir : str
        Path to the temporary directory
    """
    shutil.rmtree(temp_dir)


def time_function(func):
    """
    Decorator to measure function execution time.
    
    Parameters:
    -----------
    func : callable
        Function to time
    
    Returns:
    --------
    callable
        Wrapped function
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Function {func.__name__} executed in {execution_time:.2f} seconds")
        return result
    
    return wrapper


def split_dataset(df, test_size=0.2, random_state=42):
    """
    Split a DataFrame into training and testing sets.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to split
    test_size : float
        Fraction of data to use for testing
    random_state : int
        Random seed
    
    Returns:
    --------
    tuple
        (train_df, test_df) split DataFrames
    """
    from sklearn.model_selection import train_test_split
    
    # Generate indices for the split
    indices = np.arange(len(df))
    train_indices, test_indices = train_test_split(
        indices, test_size=test_size, random_state=random_state
    )
    
    # Split the DataFrame
    train_df = df.iloc[train_indices].copy()
    test_df = df.iloc[test_indices].copy()
    
    return train_df, test_df


def create_timestamp():
    """
    Create a formatted timestamp string.
    
    Returns:
    --------
    str
        Formatted timestamp
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")