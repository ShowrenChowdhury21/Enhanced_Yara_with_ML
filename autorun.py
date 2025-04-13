#!/usr/bin/env python3
"""
Automation script for running the Enhanced YARA with Machine Learning project.
This script assumes the project structure and code are already set up.
"""

import os
import sys
import subprocess
import argparse
import time

def run_command(command, description=None):
    """Run a shell command and print status."""
    if description:
        print(f"\n[+] {description}")
    print(f"Running: {command}")
    
    start_time = time.time()
    result = subprocess.run(command, shell=True)
    elapsed_time = time.time() - start_time
    
    if result.returncode == 0:
        print(f"✓ Command completed successfully in {elapsed_time:.2f} seconds")
        return True
    else:
        print(f"✗ Command failed with error code {result.returncode}")
        return False

def process_data():
    """Process data, extract features, and train model."""
    # Create processed directory if it doesn't exist
    os.makedirs("data/processed", exist_ok=True)
    
    run_command("python src/data/preprocessing.py", "Preprocessing data")
    run_command(
        "python src/data/feature_extraction.py --input data/processed/labeled_dataset.csv --output data/processed/extracted_features.csv --kaggle", 
        "Extracting features"
    )

def train_model():
    """Train the machine learning model."""
    # Create model directories if they don't exist
    os.makedirs("models/trained", exist_ok=True)
    os.makedirs("models/evaluation", exist_ok=True)
    
    run_command(
        "python src/models/train_model.py --features data/processed/extracted_features.csv",
        "Training machine learning model"
    )

def generate_rules():
    """Generate YARA rules from model."""
    # Create YARA rules directory if it doesn't exist
    os.makedirs("data/yara_rules", exist_ok=True)
    
    run_command(
        "python src/yara_integration/rule_generator.py model --model models/trained/random_forest_model.joblib --features data/processed/extracted_features.csv",
        "Generating YARA rules from model insights"
    )

def run_scan(file_path=None, directory_path=None, recursive=False):
    """Run enhanced YARA scan on file or directory."""
    if file_path:
        run_command(
            f"python enhanced_yara.py scan --file {file_path}",
            f"Scanning file: {file_path}"
        )
    elif directory_path:
        recursive_flag = "--recursive" if recursive else ""
        run_command(
            f"python enhanced_yara.py scan --directory {directory_path} {recursive_flag}",
            f"Scanning directory: {directory_path}"
        )
    else:
        print("[!] No file or directory specified for scanning.")

def run_webapp():
    """Run the web application."""
    run_command("python app.py", "Starting web application")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Enhanced YARA ML command automation")
    
    # Command subparsers
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Process data command
    process_parser = subparsers.add_parser('process', help='Process data and extract features')
    
    # Train model command
    train_parser = subparsers.add_parser('train', help='Train the ML model')
    
    # Generate rules command
    rules_parser = subparsers.add_parser('rules', help='Generate YARA rules from model')
    
    # Scan command
    scan_parser = subparsers.add_parser('scan', help='Run enhanced YARA scan')
    scan_parser.add_argument('--file', '-f', help='Path to file to scan')
    scan_parser.add_argument('--directory', '-d', help='Path to directory to scan')
    scan_parser.add_argument('--recursive', '-r', action='store_true', help='Scan directory recursively')
    
    # Web app command
    webapp_parser = subparsers.add_parser('webapp', help='Run web application')
    
    # Pipeline command
    pipeline_parser = subparsers.add_parser('pipeline', help='Run full pipeline (process, train, rules)')
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    # Execute the appropriate command
    if args.command == 'process':
        process_data()
    elif args.command == 'train':
        train_model()
    elif args.command == 'rules':
        generate_rules()
    elif args.command == 'scan':
        run_scan(args.file, args.directory, args.recursive)
    elif args.command == 'webapp':
        run_webapp()
    elif args.command == 'pipeline':
        process_data()
        train_model()
        generate_rules()
    else:
        parser.print_help()

if __name__ == "__main__":
    print("\n[+] Enhanced YARA with Machine Learning - Command Runner")
    main()