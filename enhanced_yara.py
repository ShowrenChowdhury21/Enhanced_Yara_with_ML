"""
Enhanced YARA: Combining YARA rules with ML-based detection.
This implements a sequential approach: YARA first, then ML if YARA fails.
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
import argparse
from pathlib import Path
import json
import traceback

try:
    import yara
    YARA_AVAILABLE = True
except ImportError:
    YARA_AVAILABLE = False
    print("Warning: YARA-python not installed. Only ML-based detection will be available.")

import joblib
from src.data.feature_extraction import extract_features


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('enhanced_yara.log')
        ]
    )
    return logging.getLogger(__name__)


def load_yara_rules(rule_path):
    """Load YARA rules from a directory."""
    logger = logging.getLogger(__name__)
    rules = {}
    
    if not YARA_AVAILABLE:
        logger.warning("YARA-python not installed. Skipping rule loading.")
        return rules
        
    try:
        # If rule_path is a single file
        if os.path.isfile(rule_path) and (rule_path.endswith('.yar') or rule_path.endswith('.yara')):
            try:
                rule_name = os.path.basename(rule_path)
                rules[rule_name] = yara.compile(rule_path)
                logger.info(f"Loaded rule: {rule_name}")
            except Exception as e:
                logger.error(f"Error compiling rule {rule_path}: {e}")
                
        # If rule_path is a directory
        elif os.path.isdir(rule_path):
            for root, _, files in os.walk(rule_path):
                for file in files:
                    if file.endswith('.yar') or file.endswith('.yara'):
                        try:
                            rule_path = os.path.join(root, file)
                            rules[file] = yara.compile(rule_path)
                            logger.info(f"Loaded rule: {file}")
                        except Exception as e:
                            logger.error(f"Error compiling rule {file}: {e}")
        else:
            logger.warning(f"Invalid rule path: {rule_path}")
            
    except Exception as e:
        logger.error(f"Error loading YARA rules: {e}")
        
    logger.info(f"Loaded {len(rules)} YARA rules")
    return rules


def scan_with_yara(file_path, yara_rules):
    """Scan a file with YARA rules."""
    logger = logging.getLogger(__name__)
    results = {}
    
    if not YARA_AVAILABLE or not yara_rules:
        return results
        
    try:
        for rule_name, rule in yara_rules.items():
            try:
                matches = rule.match(file_path)
                results[rule_name] = len(matches) > 0
                if matches:
                    logger.info(f"YARA match: {rule_name} matched {file_path}")
            except Exception as e:
                logger.error(f"Error scanning {file_path} with rule {rule_name}: {e}")
                results[rule_name] = False
    except Exception as e:
        logger.error(f"Error in YARA scanning: {e}")
        
    return results


def scan_with_ml(file_path, model_path):
    """Scan a file with the ML model."""
    logger = logging.getLogger(__name__)
    
    try:
        # Extract features
        features = extract_features(file_path)
        
        # Convert to DataFrame
        features_df = pd.DataFrame([features])
        
        # Load model
        model, _ = joblib.load(model_path)
        
        # Handle column mismatch
        expected_columns = set(model.feature_names_in_)
        current_columns = set(features_df.columns)
        
        # Add missing columns
        for col in expected_columns - current_columns:
            features_df[col] = 0
            
        # Remove extra columns
        features_df = features_df[[col for col in model.feature_names_in_ if col in features_df.columns]]
        
        # Add any missing columns with zeros
        for col in model.feature_names_in_:
            if col not in features_df.columns:
                features_df[col] = 0
        
        # Reorder columns to match model's expected order
        features_df = features_df[model.feature_names_in_]
        
        # Make prediction - handle both single and multi-class models
        try:
            # For probability prediction, handle models with 1 or 2+ classes
            if hasattr(model, 'classes_') and len(model.classes_) > 1:
                # Multi-class case
                probabilities = model.predict_proba(features_df)
                if len(model.classes_) == 2:
                    # Binary classification
                    malicious_idx = np.where(model.classes_ == True)[0]
                    if len(malicious_idx) > 0:
                        malicious_prob = probabilities[0][malicious_idx[0]]
                    else:
                        # If True not in classes, assume second class is malicious
                        malicious_prob = probabilities[0][1]
                else:
                    # Multi-class, use highest non-benign probability
                    malicious_prob = np.max(probabilities[0][1:])
            else:
                # Single class model
                prediction = model.predict(features_df)[0]
                malicious_prob = 1.0 if prediction else 0.0
        except Exception as e:
            logger.error(f"Error in prediction: {e}. Falling back to simple predict.")
            # Fallback to simple prediction
            prediction = model.predict(features_df)[0]
            malicious_prob = 1.0 if prediction else 0.0
        
        return {
            'ml_probability': float(malicious_prob),
            'ml_is_malicious': bool(malicious_prob > 0.5)
        }
    except Exception as e:
        logger.error(f"Error scanning {file_path} with ML model: {e}")
        traceback.print_exc()
        return {
            'ml_probability': 0,
            'ml_is_malicious': False,
            'ml_error': str(e)
        }


def enhanced_scan(file_path, yara_rules, model_path, threshold=0.5):
    """
    Perform enhanced scanning: YARA first, then ML if YARA fails.
    
    Parameters:
    -----------
    file_path : str
        Path to the file to scan
    yara_rules : dict
        Dictionary of compiled YARA rules
    model_path : str
        Path to the ML model
    threshold : float
        Threshold for ML classification
        
    Returns:
    --------
    dict
        Scan results including both YARA and ML findings
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Scanning: {file_path}")
    
    # YARA scan first
    yara_results = scan_with_yara(file_path, yara_rules)
    yara_match = any(yara_results.values())
    
    # If YARA detected malware, we're done
    if yara_match:
        logger.warning(f"MALICIOUS (YARA): {file_path}")
        return {
            'file_path': file_path,
            'file_name': os.path.basename(file_path),
            'is_malicious': True,
            'detection_method': 'YARA',
            'yara_match': yara_match,
            'yara_details': yara_results,
            'ml_used': False
        }
    
    # If YARA didn't detect anything, use ML
    logger.info(f"YARA found no matches, using ML for: {file_path}")
    ml_results = scan_with_ml(file_path, model_path)
    ml_is_malicious = ml_results.get('ml_is_malicious', False)
    ml_probability = ml_results.get('ml_probability', 0.0)
    
    # Determine final result based on ML
    if ml_is_malicious:
        logger.warning(f"MALICIOUS (ML): {file_path} (probability: {ml_probability:.2f})")
        detection_method = 'ML'

        # 🔥 Generate a new YARA rule based on the filename
        try:
            generated_rule_name = f"ML_Detected_{os.path.basename(file_path).replace('.', '_')}"
            generated_rule = f"""
    rule {generated_rule_name}
    {{
        strings:
            $a = "{os.path.basename(file_path)}"
        condition:
            $a
    }}
    """
            generated_rule_path = os.path.join("yara_rules", "generated_rules.yar")
            with open(generated_rule_path, "a") as f:
                f.write(generated_rule + "\n")
            logger.info(f"[+] Generated rule saved to {generated_rule_path}")
        except Exception as rule_error:
            logger.error(f"Failed to generate YARA rule for {file_path}: {rule_error}")
    else:
        logger.info(f"BENIGN: {file_path} (ML probability: {ml_probability:.2f})")
        detection_method = 'None'
    
    result = {
        'file_path': file_path,
        'file_name': os.path.basename(file_path),
        'is_malicious': ml_is_malicious,
        'detection_method': detection_method,
        'yara_match': False,
        'yara_details': yara_results,
        'ml_used': True,
        'ml_probability': ml_probability
    }
        
    return result


def main():
    """Main function for enhanced YARA scanning."""
    parser = argparse.ArgumentParser(description='Enhanced YARA scanning with ML')
    parser.add_argument('--file', '-f', help='File to scan')
    parser.add_argument('--dir', '-d', help='Directory to scan')
    parser.add_argument('--rules', '-r', default='data/yara_rules', help='Directory containing YARA rules')
    parser.add_argument('--model', '-m', default='models/trained/random_forest_model.joblib', help='Path to ML model')
    parser.add_argument('--output', '-o', help='Output JSON file for results')
    parser.add_argument('--threshold', '-t', type=float, default=0.5, help='Threshold for ML classification')
    
    args = parser.parse_args()
    
    logger = setup_logging()
    logger.info("Starting enhanced YARA scanning")
    
    # Check if either file or directory is provided
    if not args.file and not args.dir:
        logger.error("Either --file or --dir must be provided")
        parser.print_help()
        return
    
    # Load YARA rules
    logger.info(f"Loading YARA rules from {args.rules}")
    yara_rules = load_yara_rules(args.rules)
    
    # Check if model exists
    if not os.path.exists(args.model):
        logger.error(f"Model file not found: {args.model}")
        return
    
    # Scan file or directory
    results = []
    if args.file:
        if not os.path.exists(args.file):
            logger.error(f"File not found: {args.file}")
            return
            
        result = enhanced_scan(args.file, yara_rules, args.model, args.threshold)
        results.append(result)
        
    elif args.dir:
        if not os.path.exists(args.dir):
            logger.error(f"Directory not found: {args.dir}")
            return
            
        file_count = 0
        for root, _, files in os.walk(args.dir):
            for file in files:
                file_path = os.path.join(root, file)
                
                try:
                    result = enhanced_scan(file_path, yara_rules, args.model, args.threshold)
                    results.append(result)
                    file_count += 1
                except Exception as e:
                    logger.error(f"Error scanning {file_path}: {e}")
        
        logger.info(f"Scanned {file_count} files")
    
    # Calculate statistics
    if results:
        malicious_count = sum(1 for r in results if r.get('is_malicious', False))
        benign_count = len(results) - malicious_count
        
        yara_detected = sum(1 for r in results if r.get('detection_method', '') == 'YARA')
        ml_detected = sum(1 for r in results if r.get('detection_method', '') == 'ML')
        
        logger.info(f"Results: {malicious_count} malicious, {benign_count} benign")
        logger.info(f"Detection breakdown: YARA: {yara_detected}, ML: {ml_detected}")
    
    # Save results
    if args.output:
        try:
            output_dir = os.path.dirname(args.output)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
                
            logger.info(f"Results saved to {args.output}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Unhandled error: {e}")
        traceback.print_exc()