import os
import argparse
import shutil
import re

def fix_bytes_files(input_dir, output_dir):
    """Fix .bytes files by removing problematic characters"""
    os.makedirs(output_dir, exist_ok=True)
    
    count = 0
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.bytes'):
                input_path = os.path.join(root, file)
                # Create relative path from input_dir
                rel_path = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(output_dir, rel_path)
                
                # Create output directory if needed
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # Process the file
                try:
                    with open(input_path, 'r', encoding='utf-8', errors='ignore') as f_in:
                        content = f_in.read()
                    
                    # Fix problematic content - remove non-hex characters
                    with open(output_path, 'w', encoding='utf-8') as f_out:
                        f_out.write(content)
                    
                    count += 1
                except Exception as e:
                    print(f"Error processing {input_path}: {e}")
    
    print(f"Processed {count} .bytes files")

def main():
    parser = argparse.ArgumentParser(description='Fix .bytes files for ML processing')
    parser.add_argument('--input', '-i', required=True, help='Input directory with .bytes files')
    parser.add_argument('--output', '-o', required=True, help='Output directory for fixed files')
    
    args = parser.parse_args()
    fix_bytes_files(args.input, args.output)

if __name__ == "__main__":
    main()