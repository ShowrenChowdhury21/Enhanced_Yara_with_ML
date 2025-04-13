#!/usr/bin/env python3
"""
Enhanced YARA ML - Web Application
Provides a simple web interface for the enhanced YARA ML system.

Run with: python app.py
Then open a browser to http://localhost:5000
"""

import os
import argparse
import tempfile
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for, flash, send_file
import yara

from src.yara_integration.scanner import compile_rules, hybrid_scan
from src.data.feature_extraction import extract_features

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Default paths
DEFAULT_RULES_DIR = os.path.join('data', 'yara_rules')
DEFAULT_MODEL_PATH = os.path.join('models', 'trained', 'random_forest_model.joblib')

# Global variables
rules = None
model_path = None


@app.route('/')
def index():
    """Home page with file upload form."""
    return render_template('index.html')


@app.route('/scan', methods=['POST'])
def scan():
    """Handle file upload and scanning."""
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file:
        # Save uploaded file to temporary location
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, file.filename)
        file.save(file_path)
        
        try:
            # Perform hybrid scan
            result = hybrid_scan(file_path, rules, model_path)
            
            # Add additional information for display
            result['file_size'] = os.path.getsize(file_path)
            
            # Extract some features for display
            features = extract_features(file_path)
            result['entropy'] = features.get('entropy', 'N/A')
            
            return render_template('result.html', result=result)
        
        except Exception as e:
            flash(f'Error scanning file: {str(e)}')
            return redirect(url_for('index'))
        
        finally:
            # Clean up temporary file
            if os.path.exists(file_path):
                os.remove(file_path)
            os.rmdir(temp_dir)


@app.route('/batch', methods=['GET', 'POST'])
def batch():
    """Batch scanning interface."""
    if request.method == 'POST':
        # Handle batch scan
        pass
    
    return render_template('batch.html')


def initialize_app(rules_dir, model_path_arg):
    """Initialize the application with YARA rules and ML model."""
    global rules, model_path
    
    # Compile YARA rules
    try:
        rules = yara.compile(filepaths={
            "base": os.path.join(rules_dir, "example.yar"),
            "generated": os.path.join(rules_dir, "generated_rules.yar")
        })
        print(f"Compiled YARA rules from {rules_dir}")
    except Exception as e:
        print(f"Error compiling YARA rules: {e}")
        rules = None
    
    # Set model path
    model_path = model_path_arg
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Warning: Model file not found at {model_path}")


def create_app(rules_dir=DEFAULT_RULES_DIR, model_path=DEFAULT_MODEL_PATH):
    """Create and configure the Flask application."""
    initialize_app(rules_dir, model_path)
    return app


def main():
    """Main function to run the application."""
    parser = argparse.ArgumentParser(description='Enhanced YARA ML Web Application')
    parser.add_argument('--rules', default=DEFAULT_RULES_DIR, help='Directory containing YARA rules')
    parser.add_argument('--model', default=DEFAULT_MODEL_PATH, help='Path to trained ML model')
    parser.add_argument('--host', default='127.0.0.1', help='Host to run the server on')
    parser.add_argument('--port', default=5000, type=int, help='Port to run the server on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    
    args = parser.parse_args()
    
    app = create_app(args.rules, args.model)
    
    # Create templates directory if it doesn't exist
    templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
    os.makedirs(templates_dir, exist_ok=True)
    
    # Create simple HTML templates if they don't exist
    index_template = os.path.join(templates_dir, 'index.html')
    if not os.path.exists(index_template):
        with open(index_template, 'w') as f:
            f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>Enhanced YARA ML Scanner</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
        .container { max-width: 800px; margin: 0 auto; }
        h1 { color: #333; }
        form { margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
        input[type="file"] { margin: 10px 0; }
        button { background: #4CAF50; color: white; padding: 10px 15px; border: none; cursor: pointer; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Enhanced YARA ML Scanner</h1>
        
        {% with messages = get_flashed_messages() %}
            {% if messages %}
                <ul class="flashes">
                {% for message in messages %}
                    <li>{{ message }}</li>
                {% endfor %}
                </ul>
            {% endif %}
        {% endwith %}
        
        <form action="/scan" method="post" enctype="multipart/form-data">
            <h2>File Scan</h2>
            <p>Upload a file to scan for malicious content using our hybrid YARA + ML detection.</p>
            <input type="file" name="file" required>
            <button type="submit">Scan File</button>
        </form>
        
        <p><a href="/batch">Batch Scanning</a></p>
    </div>
</body>
</html>
            """)
    
    result_template = os.path.join(templates_dir, 'result.html')
    if not os.path.exists(result_template):
        with open(result_template, 'w') as f:
            f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>Scan Results - Enhanced YARA ML</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
        .container { max-width: 800px; margin: 0 auto; }
        h1, h2 { color: #333; }
        .result { margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
        .malicious { background-color: #ffebee; }
        .clean { background-color: #e8f5e9; }
        table { width: 100%; border-collapse: collapse; }
        table, th, td { border: 1px solid #ddd; }
        th, td { padding: 8px; text-align: left; }
        .back-btn { background: #2196F3; color: white; padding: 10px 15px; border: none; cursor: pointer; text-decoration: none; display: inline-block; margin-top: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Scan Results</h1>
        
        <div class="result {% if result.final_verdict %}malicious{% else %}clean{% endif %}">
            <h2>File: {{ result.file_name }}</h2>
            <p><strong>Verdict:</strong> 
                {% if result.final_verdict %}
                    <span style="color: red;">Malicious</span>
                {% else %}
                    <span style="color: green;">Clean</span>
                {% endif %}
            </p>
            <p><strong>Detection Method:</strong> {{ result.detection_method }}</p>
            
            <h3>File Details:</h3>
            <table>
                <tr>
                    <th>Property</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>File Size</td>
                    <td>{{ result.file_size }} bytes</td>
                </tr>
                <tr>
                    <td>Entropy</td>
                    <td>{{ result.entropy }}</td>
                </tr>
            </table>
            
            <h3>YARA Results:</h3>
            <p>Detected by YARA: {% if result.yara_detected %}Yes{% else %}No{% endif %}</p>
            
            {% if result.yara_matches %}
                <h4>Matching Rules:</h4>
                <ul>
                {% for rule in result.yara_matches %}
                    <li>{{ rule }}</li>
                {% endfor %}
                </ul>
            {% endif %}
            
            {% if result.ml_scan_performed %}
                <h3>Machine Learning Results:</h3>
                <table>
                    <tr>
                        <th>Property</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td>ML Verdict</td>
                        <td>{% if result.ml_result.is_malicious %}Malicious{% else %}Clean{% endif %}</td>
                    </tr>
                    <tr>
                        <td>Confidence</td>
                        <td>{{ result.ml_result.confidence }}</td>
                    </tr>
                </table>
            {% endif %}
        </div>
        
        <a href="/" class="back-btn">Scan Another File</a>
    </div>
</body>
</html>
            """)
    
    batch_template = os.path.join(templates_dir, 'batch.html')
    if not os.path.exists(batch_template):
        with open(batch_template, 'w') as f:
            f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>Batch Scanning - Enhanced YARA ML</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
        .container { max-width: 800px; margin: 0 auto; }
        h1 { color: #333; }
        form { margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
        input[type="file"] { margin: 10px 0; }
        button { background: #4CAF50; color: white; padding: 10px 15px; border: none; cursor: pointer; }
        .back-btn { background: #2196F3; color: white; padding: 10px 15px; border: none; cursor: pointer; text-decoration: none; display: inline-block; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Batch Scanning</h1>
        
        <p>This feature is coming soon! You will be able to upload multiple files or a directory for scanning.</p>
        
        <a href="/" class="back-btn">Back to Single File Scan</a>
    </div>
</body>
</html>
            """)
    
    print(f"Starting Enhanced YARA ML web application at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()