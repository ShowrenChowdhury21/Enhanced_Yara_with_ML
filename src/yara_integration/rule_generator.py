"""
Functions for generating YARA rules.
"""

import os
import re
import pandas as pd
import numpy as np
import logging
import json
import argparse
from datetime import datetime
from pathlib import Path
import hashlib


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('rule_generator.log')
        ]
    )
    return logging.getLogger(__name__)


def sanitize_rule_name(name):
    """
    Sanitize a string to be used as a YARA rule name.
    
    Parameters:
    -----------
    name : str
        String to sanitize
    
    Returns:
    --------
    str
        Sanitized string valid for YARA rule name
    """
    # Replace non-alphanumeric characters with underscores
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    
    # Ensure it starts with a letter or underscore
    if sanitized and not (sanitized[0].isalpha() or sanitized[0] == '_'):
        sanitized = 'rule_' + sanitized
    
    # Truncate if too long
    if len(sanitized) > 128:
        sanitized = sanitized[:128]
    
    return sanitized


def sanitize_string_for_yara(s):
    """
    Sanitize a string to be used in a YARA rule.
    
    Parameters:
    -----------
    s : str
        String to sanitize
    
    Returns:
    --------
    str
        Sanitized string
    """
    # Replace problematic characters
    sanitized = s.replace('\\', '\\\\').replace('"', '\\"')
    
    return sanitized


def generate_basic_rule(rule_name, strings_dict, description="", author="", tags=None):
    """
    Generate a basic YARA rule from strings.
    
    Parameters:
    -----------
    rule_name : str
        Name of the YARA rule
    strings_dict : dict
        Dictionary of string identifiers and their values
    description : str, optional
        Description of the rule
    author : str, optional
        Author of the rule
    tags : list, optional
        List of tags for the rule
    
    Returns:
    --------
    str
        Generated YARA rule as string
    """
    # Sanitize rule name
    rule_name = sanitize_rule_name(rule_name)
    
    # Start building the rule
    rule = f"rule {rule_name}"
    
    # Add tags if provided
    if tags and len(tags) > 0:
        tags_str = " ".join(tags)
        rule += f" : {tags_str}"
    
    rule += " {\n"
    
    # Add metadata
    rule += "    meta:\n"
    if description:
        rule += f"        description = \"{sanitize_string_for_yara(description)}\"\n"
    if author:
        rule += f"        author = \"{sanitize_string_for_yara(author)}\"\n"
    rule += f"        date = \"{datetime.now().strftime('%Y-%m-%d')}\"\n"
    
    # Add strings
    rule += "    strings:\n"
    for identifier, value in strings_dict.items():
        # Sanitize identifier
        identifier = sanitize_rule_name(identifier)
        
        # Handle different types of string definitions
        if isinstance(value, bytes):
            # Binary string
            hex_string = ' '.join([f"{b:02x}" for b in value])
            rule += f"        ${identifier} = {{ {hex_string} }}\n"
        elif value.startswith('/') and value.endswith('/'):
            # Regular expression
            rule += f"        ${identifier} = {value}\n"
        else:
            # Text string
            rule += f"        ${identifier} = \"{sanitize_string_for_yara(value)}\"\n"
    
    # Add condition
    rule += "    condition:\n"
    if len(strings_dict) > 1:
        rule += f"        any of them\n"
    else:
        # Use the single string identifier
        identifier = list(strings_dict.keys())[0]
        identifier = sanitize_rule_name(identifier)
        rule += f"        ${identifier}\n"
    
    rule += "}\n"
    
    return rule


def save_rule(rule_text, output_path):
    """
    Save a YARA rule to a file.
    
    Parameters:
    -----------
    rule_text : str
        YARA rule text
    output_path : str
        Path to save the rule
    
    Returns:
    --------
    str
        Path to the saved rule file
    """
    logger = logging.getLogger(__name__)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write the rule to file
    with open(output_path, 'w') as f:
        f.write(rule_text)
    
    logger.info(f"Rule saved to {output_path}")
    return output_path


def generate_rules_from_malware_strings(file_path, output_dir, min_string_length=6):
    """
    Extract strings from a malware sample and generate YARA rules.
    
    Parameters:
    -----------
    file_path : str
        Path to the malware file
    output_dir : str
        Directory to save generated rules
    min_string_length : int
        Minimum string length to include
    
    Returns:
    --------
    str
        Path to the generated rule file
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Generating rules from malware file: {file_path}")
    
    # Read the file
    with open(file_path, 'rb') as f:
        data = f.read()
    
    # Calculate file hash for rule naming
    file_hash = hashlib.md5(data).hexdigest()
    
    # Extract ASCII strings
    ascii_strings = re.findall(b'[ -~]{%d,}' % min_string_length, data)
    
    # Extract wide (UTF-16) strings
    wide_strings = re.findall(b'(?:[ -~]\x00){%d,}' % min_string_length, data)
    
    # Convert binary strings to regular strings
    ascii_strings = [s.decode('ascii', errors='ignore') for s in ascii_strings]
    wide_strings = [s.decode('utf-16-le', errors='ignore') for s in wide_strings]
    
    # Filter strings
    filtered_strings = []
    
    # Common string patterns to exclude
    exclude_patterns = [
        r'^[0-9A-F]+$',  # Hex strings
        r'^[a-zA-Z0-9_]+$',  # Simple alphanumeric
        r'WINDOWS',  # Common Windows strings
        r'windows',
        r'Microsoft',
        r'Visual C\+\+',
        r'\.dll$',
        r'\.exe$',
        r'\\\\',  # Path separators
        r'[/\\]Program Files[/\\]',
        r'CommonProgramFiles',
        r'Application Data',
        r'http://',
        r'https://',
        r'www\.',
        r'UTF-',
        r'Mozilla',
        r'Firefox',
        r'Chrome',
        r'Opera',
        r'MSIE',
        r'USER32',
        r'KERNEL32',
        r'ADVAPI32',
        r'SHELL32',
        r'COMCTL32'
    ]
    
    # Compile regex patterns
    exclude_regex = re.compile('|'.join(exclude_patterns))
    
    # Filter ASCII strings
    for string in ascii_strings:
        if len(string) >= min_string_length and not exclude_regex.search(string):
            filtered_strings.append(string)
    
    # Filter Wide strings
    for string in wide_strings:
        if len(string) >= min_string_length and not exclude_regex.search(string):
            filtered_strings.append(string)
    
    # Keep only unique strings
    unique_strings = list(set(filtered_strings))
    
    # Sort by length (prioritize longer strings)
    unique_strings.sort(key=len, reverse=True)
    
    # Limit to top N strings
    max_strings = 10
    selected_strings = unique_strings[:max_strings]
    
    # Check if we have enough strings
    if len(selected_strings) < 3:
        logger.warning(f"Not enough unique strings found in {file_path}")
        
        # Use byte patterns instead
        byte_patterns = {}
        
        # Find some distinctive byte sequences
        for i in range(0, min(len(data), 10000), 20):
            chunk = data[i:i+20]
            if len(chunk) == 20:
                byte_patterns[f"byte_pattern_{i}"] = chunk
        
        if byte_patterns:
            rule_name = f"malware_{file_hash[:8]}_bytes"
            description = f"Auto-generated rule for file {os.path.basename(file_path)} (MD5: {file_hash})"
            
            rule_text = generate_basic_rule(
                rule_name=rule_name,
                strings_dict=byte_patterns,
                description=description,
                author="Enhanced YARA ML Rule Generator",
                tags=["auto-generated", "bytes"]
            )
            
            output_path = os.path.join(output_dir, f"{rule_name}.yar")
            return save_rule(rule_text, output_path)
        else:
            logger.error(f"Could not generate rule for {file_path}")
            return None
    
    # Create strings dictionary for rule
    strings_dict = {}
    for i, string in enumerate(selected_strings):
        strings_dict[f"string_{i}"] = string
    
    # Generate the rule
    rule_name = f"malware_{file_hash[:8]}_strings"
    description = f"Auto-generated rule for file {os.path.basename(file_path)} (MD5: {file_hash})"
    
    rule_text = generate_basic_rule(
        rule_name=rule_name,
        strings_dict=strings_dict,
        description=description,
        author="Enhanced YARA ML Rule Generator",
        tags=["auto-generated", "strings"]
    )
    
    # Save the rule
    output_path = os.path.join(output_dir, f"{rule_name}.yar")
    return save_rule(rule_text, output_path)


def generate_rules_from_model(model, feature_names, output_dir, threshold=0.01, max_rules=10):
    """
    Generate YARA rules based on feature importance from a trained model.
    
    Parameters:
    -----------
    model : RandomForestClassifier
        Trained model
    feature_names : list
        List of feature names
    output_dir : str
        Directory to save generated rules
    threshold : float, optional
        Importance threshold for including features in rules
    max_rules : int, optional
        Maximum number of rules to generate
    
    Returns:
    --------
    list
        List of paths to the generated rule files
    """
    logger = logging.getLogger(__name__)
    logger.info("Generating YARA rules from model feature importance")
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Create a DataFrame for easier handling
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    # Filter features by importance threshold
    important_features = feature_importance[feature_importance['Importance'] > threshold]
    
    # Limit to max_rules
    if len(important_features) > max_rules:
        important_features = important_features.head(max_rules)
    
    logger.info(f"Found {len(important_features)} important features for rule generation")
    
    # Generate rules for important features
    rule_files = []
    
    for _, row in important_features.iterrows():
        feature = row['Feature']
        importance = row['Importance']
        
        # Create rule based on feature type
        # This is a simplified approach and would need customization
        # based on actual features in your model
        if feature.startswith('api_call_'):
            # API call feature
            api_name = feature.replace('api_call_', '')
            rule_name = f"ML_API_{api_name}"
            strings_dict = {
                "api": api_name
            }
            description = f"Auto-generated rule for API call {api_name} with importance {importance:.4f}"
        elif feature.startswith('string_'):
            # String feature
            string_value = feature.replace('string_', '')
            rule_name = f"ML_String_{hashlib.md5(string_value.encode()).hexdigest()[:8]}"
            strings_dict = {
                "str": string_value
            }
            description = f"Auto-generated rule for string feature with importance {importance:.4f}"
        elif feature.startswith('byte_pattern_'):
            # Byte pattern feature
            pattern = bytes.fromhex(feature.replace('byte_pattern_', ''))
            rule_name = f"ML_BytePattern_{hashlib.md5(pattern).hexdigest()[:8]}"
            strings_dict = {
                "pattern": pattern
            }
            description = f"Auto-generated rule for byte pattern with importance {importance:.4f}"
        elif 'entropy' in feature.lower():
            # Entropy feature - can't directly translate to YARA
            continue
        else:
            # Generic feature
            rule_name = f"ML_Feature_{sanitize_rule_name(feature)}"
            # Create a very basic rule
            strings_dict = {
                "feature_name": feature
            }
            description = f"Auto-generated rule for feature {feature} with importance {importance:.4f}"
        
        # Generate the rule
        rule_text = generate_basic_rule(
            rule_name=rule_name,
            strings_dict=strings_dict,
            description=description,
            author="ML Rule Generator",
            tags=["auto-generated", "ml-based"]
        )
        
        # Save the rule
        rule_path = os.path.join(output_dir, f"{rule_name}.yar")
        save_rule(rule_text, rule_path)
        rule_files.append(rule_path)
    
    logger.info(f"Generated {len(rule_files)} YARA rules")
    return rule_files


def generate_combined_rule_file(rule_files, output_path):
    """
    Combine multiple YARA rule files into a single file.
    
    Parameters:
    -----------
    rule_files : list
        List of paths to rule files
    output_path : str
        Path to save the combined rule file
    
    Returns:
    --------
    str
        Path to the combined rule file
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Combining {len(rule_files)} rules into {output_path}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Combine rules
    with open(output_path, 'w') as outfile:
        outfile.write("/* \n")
        outfile.write(f" * Combined YARA rules\n")
        outfile.write(f" * Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        outfile.write(f" * Rules: {len(rule_files)}\n")
        outfile.write(" */\n\n")
        
        for rule_file in rule_files:
            outfile.write(f"// Rule from: {os.path.basename(rule_file)}\n")
            with open(rule_file, 'r') as infile:
                outfile.write(infile.read())
            outfile.write("\n\n")
    
    logger.info(f"Combined rule file saved to {output_path}")
    return output_path


def main():
    """Main function for rule generation."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Generate YARA rules')
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Model-based rule generation command
    model_parser = subparsers.add_parser('model', help='Generate rules from model feature importance')
    model_parser.add_argument('--model', '-m', required=True, help='Path to trained model file')
    model_parser.add_argument('--features', '-f', required=True, help='Path to features file (CSV)')
    model_parser.add_argument('--output-dir', '-o', default='data/yara_rules', help='Directory to save generated rules')
    model_parser.add_argument('--threshold', '-t', type=float, default=0.01, help='Feature importance threshold')
    model_parser.add_argument('--max-rules', type=int, default=10, help='Maximum number of rules to generate')
    
    # String-based rule generation command
    string_parser = subparsers.add_parser('strings', help='Generate rules from strings in malware samples')
    string_parser.add_argument('--file', '-f', help='Path to malware file')
    string_parser.add_argument('--directory', '-d', help='Directory containing malware files')
    string_parser.add_argument('--output-dir', '-o', default='data/yara_rules', help='Directory to save generated rules')
    string_parser.add_argument('--min-length', '-l', type=int, default=6, help='Minimum string length')
    
    # Combine rules command
    combine_parser = subparsers.add_parser('combine', help='Combine multiple YARA rule files')
    combine_parser.add_argument('--directory', '-d', required=True, help='Directory containing YARA rule files')
    combine_parser.add_argument('--output', '-o', required=True, help='Output file path')
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging()
    logger.info("Starting YARA rule generation")
    
    # Handle commands
    if args.command == 'model':
        # Model-based rule generation
        import joblib
        from sklearn.ensemble import RandomForestClassifier
        
        # Load model
        logger.info(f"Loading model from {args.model}")
        model = joblib.load(args.model)
        
        # Load features file to get feature names
        logger.info(f"Loading features from {args.features}")
        features_df = pd.read_csv(args.features)
        
        # Extract feature names
        feature_names = features_df.columns.tolist()
        for col in ['file_path', 'file_name', 'md5', 'sha1', 'sha256', 'is_malicious']:
            if col in feature_names:
                feature_names.remove(col)
        
        # Generate rules
        generate_rules_from_model(
            model, 
            feature_names, 
            args.output_dir, 
            args.threshold,
            args.max_rules
        )
    
    elif args.command == 'strings':
        # String-based rule generation
        if args.file:
            # Single file
            generate_rules_from_malware_strings(
                args.file, 
                args.output_dir, 
                args.min_length
            )
        elif args.directory:
            # Directory of files
            file_count = 0
            for root, _, files in os.walk(args.directory):
                for file in files:
                    file_path = os.path.join(root, file)
                    generate_rules_from_malware_strings(
                        file_path, 
                        args.output_dir, 
                        args.min_length
                    )
                    file_count += 1
            logger.info(f"Generated rules from {file_count} files")
        else:
            logger.error("Either --file or --directory must be specified")
    
    elif args.command == 'combine':
        # Combine rules
        rule_files = []
        for file in os.listdir(args.directory):
            if file.endswith('.yar'):
                rule_files.append(os.path.join(args.directory, file))
        
        if rule_files:
            generate_combined_rule_file(rule_files, args.output)
        else:
            logger.error(f"No YARA rule files found in {args.directory}")
    
    else:
        parser.print_help()
    
    logger.info("Rule generation complete")


if __name__ == "__main__":
    main()