"""
Functions for scanning files with YARA rules.
"""

import os
import sys
import yara
import glob
import logging
import pandas as pd
import json
import argparse
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
import hashlib
import re


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('yara_scanner.log')
        ]
    )
    return logging.getLogger(__name__)


def compile_rules(rules_dir, compiled_rules_path=None):
    """
    Compile YARA rules from a directory.
    
    Parameters:
    -----------
    rules_dir : str
        Directory containing YARA rule files
    compiled_rules_path : str, optional
        Path to save the compiled rules
    
    Returns:
    --------
    yara.Rules
        Compiled YARA rules
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Compiling YARA rules from {rules_dir}")
    
    # Get all .yar files in the directory
    rule_files = glob.glob(os.path.join(rules_dir, "*.yar"))
    
    if not rule_files:
        logger.warning(f"No YARA rule files found in {rules_dir}")
        # Create an empty rule to prevent errors
        empty_rule_path = os.path.join(rules_dir, "empty.yar")
        with open(empty_rule_path, 'w') as f:
            f.write('rule empty { condition: false }\n')
        rule_files = [empty_rule_path]
    
    # Create a dictionary of rule_name: rule_path
    rules_dict = {}
    for rule_file in rule_files:
        rule_name = os.path.splitext(os.path.basename(rule_file))[0]
        rules_dict[rule_name] = rule_file
    
    # Compile rules
    try:
        rules = yara.compile(filepaths=rules_dict)
        logger.info(f"Successfully compiled {len(rule_files)} rule files")
        
        # Save compiled rules if path is provided
        if compiled_rules_path:
            os.makedirs(os.path.dirname(compiled_rules_path), exist_ok=True)
            rules.save(compiled_rules_path)
            logger.info(f"Compiled rules saved to {compiled_rules_path}")
        
        return rules
    
    except yara.Error as e:
        logger.error(f"Error compiling YARA rules: {e}")
        # Try to compile rules individually to identify the problematic ones
        valid_rules_dict = {}
        for name, path in rules_dict.items():
            try:
                yara.compile(filepath=path)
                valid_rules_dict[name] = path
            except yara.Error as err:
                logger.error(f"Error in rule file {path}: {err}")
        
        if valid_rules_dict:
            logger.info(f"Re-compiling with {len(valid_rules_dict)} valid rule files")
            rules = yara.compile(filepaths=valid_rules_dict)
            
            if compiled_rules_path:
                rules.save(compiled_rules_path)
                logger.info(f"Compiled valid rules saved to {compiled_rules_path}")
            
            return rules
        else:
            logger.error("No valid YARA rules found")
            raise


def load_compiled_rules(compiled_rules_path):
    """
    Load compiled YARA rules from file.
    
    Parameters:
    -----------
    compiled_rules_path : str
        Path to the compiled rules file
    
    Returns:
    --------
    yara.Rules
        Loaded YARA rules
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading compiled rules from {compiled_rules_path}")
    
    try:
        rules = yara.load(compiled_rules_path)
        return rules
    except Exception as e:
        logger.error(f"Error loading compiled rules: {e}")
        raise


def get_rule_metadata(rules_dir):
    """
    Extract metadata from YARA rule files.
    
    Parameters:
    -----------
    rules_dir : str
        Directory containing YARA rule files
    
    Returns:
    --------
    dict
        Dictionary of rule metadata by rule name
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Extracting rule metadata from {rules_dir}")
    
    metadata = {}
    rule_files = glob.glob(os.path.join(rules_dir, "*.yar"))
    
    for rule_file in rule_files:
        rule_name = os.path.splitext(os.path.basename(rule_file))[0]
        
        with open(rule_file, 'r') as f:
            content = f.read()
            
            # Extract rule name from content
            name_match = re.search(r'rule\s+(\w+)', content)
            if name_match:
                actual_name = name_match.group(1)
            else:
                actual_name = rule_name
            
            # Extract metadata section
            meta_match = re.search(r'meta:\s*(.*?)(?:strings:|condition:)', content, re.DOTALL)
            if meta_match:
                meta_text = meta_match.group(1)
                
                # Parse metadata items
                rule_meta = {}
                for line in meta_text.strip().split('\n'):
                    line = line.strip()
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # Remove quotes if present
                        if value.startswith('"') and value.endswith('"'):
                            value = value[1:-1]
                        
                        rule_meta[key] = value
                
                metadata[actual_name] = rule_meta
    
    logger.info(f"Extracted metadata for {len(metadata)} rules")
    return metadata


def scan_file(file_path, rules):
    """
    Scan a file with YARA rules.
    
    Parameters:
    -----------
    file_path : str
        Path to the file to scan
    rules : yara.Rules
        Compiled YARA rules
    
    Returns:
    --------
    list
        List of matching rules
    """
    try:
        # Scan the file
        matches = rules.match(file_path, timeout=60)  # Add timeout to prevent hanging
        return matches
    except yara.TimeoutError:
        logging.warning(f"Timeout scanning {file_path}")
        return []
    except Exception as e:
        logging.error(f"Error scanning {file_path}: {e}")
        return []


def batch_scan(file_paths, rules, output_path=None, max_files=None):
    """
    Scan multiple files with YARA rules.
    
    Parameters:
    -----------
    file_paths : list
        List of paths to files to scan
    rules : yara.Rules
        Compiled YARA rules
    output_path : str, optional
        Path to save scan results as CSV
    max_files : int, optional
        Maximum number of files to scan
    
    Returns:
    --------
    pd.DataFrame
        DataFrame containing scan results for all files
    """
    logger = logging.getLogger(__name__)
    
    # Limit number of files if specified
    if max_files and len(file_paths) > max_files:
        logger.info(f"Limiting scan to {max_files} files out of {len(file_paths)}")
        file_paths = file_paths[:max_files]
    
    logger.info(f"Scanning {len(file_paths)} files with YARA rules")
    all_results = []
    
    for file_path in tqdm(file_paths, desc="Scanning files"):
        try:
            # Get file size
            file_size = os.path.getsize(file_path)
            
            # Skip files that are too large
            max_file_size = 100 * 1024 * 1024  # 100 MB
            if file_size > max_file_size:
                logger.warning(f"Skipping large file {file_path} ({file_size/1024/1024:.1f} MB)")
                continue
            
            # Calculate file hash
            with open(file_path, 'rb') as f:
                file_data = f.read()
                md5_hash = hashlib.md5(file_data).hexdigest()
            
            # Scan the file
            matches = scan_file(file_path, rules)
            
            # Process matches
            result = {
                'file_path': file_path,
                'file_name': os.path.basename(file_path),
                'file_size': file_size,
                'md5_hash': md5_hash,
                'match_count': len(matches),
                'matches': [match.rule for match in matches],
                'is_detected': len(matches) > 0
            }
            
            all_results.append(result)
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save to CSV if output path is provided
    if output_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        results_df.to_csv(output_path, index=False)
        logger.info(f"Scan results saved to {output_path}")
    
    # Log summary statistics
    total_files = len(results_df)
    detected_count = results_df['is_detected'].sum()
    detection_rate = detected_count / total_files * 100 if total_files > 0 else 0
    
    logger.info(f"Scan complete: {total_files} files scanned")
    logger.info(f"Files detected: {detected_count} ({detection_rate:.1f}%)")
    
    return results_df


def hybrid_scan(file_path, yara_rules, ml_model_path, features_function=None):
    """
    Perform hybrid scanning using both YARA and ML model.
    
    Parameters:
    -----------
    file_path : str
        Path to the file to scan
    yara_rules : yara.Rules
        Compiled YARA rules
    ml_model_path : str
        Path to the trained ML model
    features_function : callable, optional
        Function to extract features for ML prediction
    
    Returns:
    --------
    dict
        Results from both YARA and ML scanning
    """
    logger = logging.getLogger(__name__)
    
    # Import needed modules here to avoid circular imports
    import joblib
    import sys
    from pathlib import Path
    
    # Ensure we can import from parent directory
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    
    # Import feature extraction function
    if features_function is None:
        from src.data.feature_extraction import extract_features as features_function
    
    # YARA scan
    logger.info(f"Performing YARA scan on {file_path}")
    yara_matches = scan_file(file_path, yara_rules)
    yara_detected = len(yara_matches) > 0
    
    logger.info(f"YARA scan results: {'Detected' if yara_detected else 'Clean'}")
    if yara_detected:
        logger.info(f"Matching rules: {[match.rule for match in yara_matches]}")
    
    # ML scan (only if YARA doesn't detect)
    ml_result = None
    if not yara_detected and ml_model_path:
        logger.info(f"YARA did not detect, performing ML scan")
        
        try:
            # Load model
            model = joblib.load(ml_model_path)
            
            # Extract features
            features = features_function(file_path)
            
            # Remove non-feature fields
            for field in ['file_path', 'file_name', 'md5', 'sha1', 'sha256']:
                if field in features:
                    del features[field]
            
            # Convert to DataFrame for prediction
            import pandas as pd
            features_df = pd.DataFrame([features])
            
            # Convert non-numeric columns to numeric
            features_df = pd.get_dummies(features_df)
            
            # Make prediction
            # Handle missing columns that were present during training
            missing_cols = set(model.feature_names_in_) - set(features_df.columns)
            for col in missing_cols:
                features_df[col] = 0
            
            # Ensure correct column order
            features_df = features_df[model.feature_names_in_]
            
            prediction = model.predict(features_df)[0]
            probabilities = model.predict_proba(features_df)[0]
            
            # Determine class and probability
            is_malicious = bool(prediction)
            probability = probabilities[1] if is_malicious else probabilities[0]
            
            # Process ML result
            ml_result = {
                'is_malicious': is_malicious,
                'probability': float(probability),
                'confidence': f"{probability * 100:.2f}%"
            }
            
            logger.info(f"ML scan results: {'Malicious' if is_malicious else 'Clean'} with {probability * 100:.2f}% confidence")
        
        except Exception as e:
            logger.error(f"Error during ML scanning: {e}")
            ml_result = {
                'is_malicious': False,
                'probability': 0.0,
                'confidence': "Error",
                'error': str(e)
            }
    
    # Combine results
    result = {
        'file_path': file_path,
        'file_name': os.path.basename(file_path),
        'file_size': os.path.getsize(file_path),
        'yara_detected': yara_detected,
        'yara_matches': [match.rule for match in yara_matches] if yara_matches else [],
        'ml_scan_performed': ml_result is not None,
        'ml_result': ml_result,
        'final_verdict': yara_detected or (ml_result and ml_result['is_malicious']) if ml_result else yara_detected,
        'detection_method': 'YARA' if yara_detected else ('ML' if ml_result and ml_result['is_malicious'] else 'None')
    }
    
    # Log final verdict
    logger.info(f"Final verdict: {'Malicious' if result['final_verdict'] else 'Clean'} (Method: {result['detection_method']})")
    
    return result


def main():
    """Main function for YARA scanning."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Scan files with YARA rules')
    parser.add_argument('--file', '-f', help='Path to file for scanning')
    parser.add_argument('--directory', '-d', help='Directory containing files to scan')
    parser.add_argument('--recursive', '-r', action='store_true', help='Scan directory recursively')
    parser.add_argument('--rules-dir', default='data/yara_rules', help='Directory containing YARA rules')
    parser.add_argument('--compiled-rules', help='Path to compiled rules file')
    parser.add_argument('--output', '-o', help='Output file for scan results (CSV format)')
    parser.add_argument('--max-files', type=int, help='Maximum number of files to scan')
    parser.add_argument('--hybrid', action='store_true', help='Use hybrid scanning with ML model')
    parser.add_argument('--model', default='models/trained/random_forest_model.joblib', help='Path to ML model for hybrid scanning')
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging()
    logger.info("Starting YARA scanning")
    
    # Check inputs
    if not args.file and not args.directory:
        logger.error("Either --file or --directory must be specified")
        parser.print_help()
        sys.exit(1)
    
    # Load or compile YARA rules
    if args.compiled_rules and os.path.exists(args.compiled_rules):
        rules = load_compiled_rules(args.compiled_rules)
    else:
        if not os.path.exists(args.rules_dir):
            logger.error(f"Rules directory not found: {args.rules_dir}")
            sys.exit(1)
        
        rules = compile_rules(args.rules_dir, args.compiled_rules)
    
    # Get files to scan
    if args.file:
        file_paths = [args.file]
    else:
        file_paths = []
        if args.recursive:
            # Recursive directory walk
            for root, _, files in os.walk(args.directory):
                for file in files:
                    file_paths.append(os.path.join(root, file))
        else:
            # Non-recursive directory listing
            file_paths = [os.path.join(args.directory, f) for f in os.listdir(args.directory) 
                         if os.path.isfile(os.path.join(args.directory, f))]
    
    # Check if any files were found
    if not file_paths:
        logger.error("No files found for scanning")
        sys.exit(1)
    
    logger.info(f"Found {len(file_paths)} files for scanning")
    
    # Perform scanning
    if args.hybrid and len(file_paths) == 1:
        # Hybrid scan for a single file
        result = hybrid_scan(file_paths[0], rules, args.model)
        
        # Print result
        print("\nScan Result:")
        print(f"File: {result['file_name']}")
        print(f"Verdict: {'Malicious' if result['final_verdict'] else 'Clean'}")
        print(f"Detection Method: {result['detection_method']}")
        
        if result['yara_detected']:
            print("\nYARA Matches:")
            for match in result['yara_matches']:
                print(f"- {match}")
        
        if result['ml_scan_performed']:
            print("\nML Scan Results:")
            print(f"Verdict: {'Malicious' if result['ml_result']['is_malicious'] else 'Clean'}")
            print(f"Confidence: {result['ml_result']['confidence']}")
        
        # Save result if output path provided
        if args.output:
            pd.DataFrame([result]).to_csv(args.output, index=False)
            logger.info(f"Scan result saved to {args.output}")
    
    else:
        # Batch scanning
        batch_scan(file_paths, rules, args.output, args.max_files)
    
    logger.info("Scanning complete")


if __name__ == "__main__":
    main()