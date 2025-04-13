"""
Functions for making predictions using the trained malware detection model.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
import logging
import argparse
from pathlib import Path
import json

# Import feature extraction functionality
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.feature_extraction import extract_features


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('prediction.log')
        ]
    )
    return logging.getLogger(__name__)


def load_model(model_path):
    """
    Load a trained model from disk.
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model
    
    Returns:
    --------
    tuple
        (model, metadata) - Trained model and its metadata
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Check if model file exists
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return None, None
        
        # Load model
        model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
        
        # Load metadata if available
        metadata_path = model_path.replace('.joblib', '_metadata.json')
        metadata = None
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            logger.info(f"Model metadata loaded from {metadata_path}")
        
        return model, metadata
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None, None


def predict_file(model, file_path):
    """
    Make prediction for a single file.
    
    Parameters:
    -----------
    model : object
        Trained model
    file_path : str
        Path to the file
    
    Returns:
    --------
    dict
        Prediction result
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Analyzing file: {file_path}")
    
    try:
        # Extract features
        features = extract_features(file_path)
        features_df = pd.DataFrame([features])
        
        # Get feature names from model
        model_features = model.feature_names_in_
        
        # Ensure all required features are present
        for feature in model_features:
            if feature not in features_df.columns:
                features_df[feature] = 0
        
        # Select only the features used by the model
        features_df = features_df[model_features]
        
        # Make prediction
        prediction = model.predict(features_df)[0]
        
        # Get probability if available
        probability = 0.5
        if hasattr(model, 'predict_proba'):
            try:
                proba = model.predict_proba(features_df)[0]
                
                # Find probability for the positive class (True or 1)
                if len(proba) > 1:
                    # Check if model.classes_ contains True
                    if hasattr(model, 'classes_') and True in model.classes_:
                        true_idx = np.where(model.classes_ == True)[0][0]
                        probability = proba[true_idx]
                    else:
                        # Assume second class (index 1) is positive
                        probability = proba[1]
                else:
                    # Single class model
                    probability = 1.0 if prediction else 0.0
            except Exception as e:
                logger.warning(f"Error getting prediction probability: {e}")
        
        # Create result dictionary
        result = {
            'file_path': file_path,
            'file_name': os.path.basename(file_path),
            'prediction': bool(prediction),
            'probability': float(probability),
            'features': {
                'entropy': features.get('entropy', 0),
                'file_size': features.get('file_size', 0),
                'api_call_count': features.get('api_call_count', 0),
                'string_count': features.get('string_count', 0),
            }
        }
        
        # Log prediction
        if result['prediction']:
            logger.warning(f"MALICIOUS: {file_path} (confidence: {probability:.2f})")
        else:
            logger.info(f"BENIGN: {file_path} (confidence: {1-probability:.2f})")
        
        return result
    
    except Exception as e:
        logger.error(f"Error predicting file {file_path}: {e}")
        return {
            'file_path': file_path,
            'file_name': os.path.basename(file_path),
            'prediction': False,
            'probability': 0.0,
            'error': str(e)
        }


def batch_predict(model, file_paths, output_path=None):
    """
    Make predictions for multiple files.
    
    Parameters:
    -----------
    model : object
        Trained model
    file_paths : list
        List of file paths
    output_path : str, optional
        Path to save results
    
    Returns:
    --------
    list
        List of prediction results
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Analyzing {len(file_paths)} files")
    
    results = []
    
    for file_path in file_paths:
        result = predict_file(model, file_path)
        results.append(result)
    
    # Save results if output path is provided
    if output_path:
        try:
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            # Convert to DataFrame for easier saving
            results_df = pd.DataFrame(results)
            results_df.to_csv(output_path, index=False)
            
            logger.info(f"Results saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    return results


def main():
    """Main function for prediction."""
    parser = argparse.ArgumentParser(description='Make predictions with trained model')
    parser.add_argument('--model', '-m', default='../../models/trained/random_forest_model.joblib', help='Path to trained model')
    parser.add_argument('--file', '-f', help='Single file to analyze')
    parser.add_argument('--dir', '-d', help='Directory of files to analyze')
    parser.add_argument('--csv', '-c', help='CSV file with list of files to analyze')
    parser.add_argument('--output', '-o', help='Output file for results (CSV)')
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging()
    logger.info("Starting prediction")
    
    # Load model
    model, metadata = load_model(args.model)
    if model is None:
        logger.error("Failed to load model")
        return
    
    # Get file paths
    file_paths = []
    
    if args.file:
        if os.path.exists(args.file):
            file_paths.append(args.file)
        else:
            logger.error(f"File not found: {args.file}")
            return
    
    elif args.dir:
        if os.path.exists(args.dir) and os.path.isdir(args.dir):
            for root, _, files in os.walk(args.dir):
                for file in files:
                    file_paths.append(os.path.join(root, file))
            logger.info(f"Found {len(file_paths)} files in directory")
        else:
            logger.error(f"Directory not found: {args.dir}")
            return
    
    elif args.csv:
        if os.path.exists(args.csv):
            try:
                df = pd.read_csv(args.csv)
                
                # Look for file path column
                path_column = None
                for col in df.columns:
                    if 'path' in col.lower() or 'file' in col.lower():
                        path_column = col
                        break
                
                if path_column is None and len(df.columns) > 0:
                    path_column = df.columns[0]
                
                if path_column:
                    file_paths = [path for path in df[path_column] if os.path.exists(str(path))]
                    logger.info(f"Found {len(file_paths)} existing files in CSV")
                else:
                    logger.error("No suitable column found in CSV")
                    return
            except Exception as e:
                logger.error(f"Error reading CSV: {e}")
                return
        else:
            logger.error(f"CSV file not found: {args.csv}")
            return
    
    else:
        logger.error("No input specified. Use --file, --dir, or --csv")
        parser.print_help()
        return
    
    if not file_paths:
        logger.error("No valid files found for prediction")
        return
    
    # Make predictions
    results = batch_predict(model, file_paths, args.output)
    
    # Print summary
    malicious_count = sum(1 for r in results if r.get('prediction', False))
    benign_count = len(results) - malicious_count
    
    logger.info(f"Results: {malicious_count} malicious, {benign_count} benign")


if __name__ == "__main__":
    main()