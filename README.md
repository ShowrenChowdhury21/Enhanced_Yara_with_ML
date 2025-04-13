Enhanced YARA with Machine Learning
This project enhances YARA, a rule-based open-source malware detection tool, by integrating machine learning to improve its ability to detect zero-day threats, polymorphic malware, and obfuscated binaries. The hybrid approach combines YARA's traditional rule-based scanning with a Random Forest classifier trained on static analysis features.

Project Structure

enhanced-yara-ml/
├── data/
│   ├── processed/
│   ├── raw/
│   ├── test_samples/
│   └── yara_rules/
├── models/
│   ├── evaluation/
│   └── trained/
├── notebooks/
├── src/
│   ├── data/
│   ├── models/
│   ├── utils/
│   └── yara_integration/
├── tests/
├── README.md
├── requirements.txt
├── setup.py
└── various utility scripts


1. Installation
1.1 Clone the repository:

git clone https://github.com/yourusername/enhanced-yara-ml.git
cd enhanced-yara-mll

1.2 Create and activate a virtual environment:

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

1.3 Install dependencies:

pip install -r requirements.txt

1.4 Install YARA-python:

# On Linux
pip install yara-python

# On Windows
# You may need Visual C++ Build Tools
pip install yara-python


2. Dataset
This project uses the Kaggle Microsoft Malware Classification dataset for training and evaluation.

2.1 Download the dataset from Kaggle:

# First install kaggle CLI if you haven't already
pip install kaggle

# Set up your Kaggle API credentials
# Place kaggle.json in ~/.kaggle/

# Download the dataset
kaggle competitions download -c malware-classification
unzip malware-classification.zip -d data/raw


3. Workflow
3.1 Preprocessing
Preprocess the raw data to extract features for machine learning:

python src/data/preprocessing.py --input_dir data/raw --output_dir data/processed

This will:

Process the binary and assembly files
Extract basic metadata
Perform necessary cleanup and provide a CSV file.

3.2  Feature Extraction
Extract meaningful features from the preprocessed files:
python src/data/feature_extraction.py --input data/processed/labeled_dataset.csv --output data/processed/extracted_features.csv --kaggle 

This generates features such as:

File size and entropy
API call frequencies
String patterns
Op-code statistics
And more

3.3 Training
Train the machine learning model:
python src/models/model_training.py --features data/processed/features/features.csv --output models/trained/random_forest_model.joblib

The training script:

Splits the data into training and validation sets
Trains a Random Forest classifier
Performs hyperparameter tuning
Evaluates the model on the validation set
Saves the trained model

5. Evaluation
For getting 50 sample dataset for testing, run:
New-Item -ItemType Directory -Force -Path data\test_samples

then run
Get-ChildItem -Path data\raw\test\test -Filter *.bytes | Get-Random -Count 50 | ForEach-Object {Copy-Item $_.FullName -Destination data\test_samples}

Evaluate the enhanced YARA system against traditional YARA:
python evaluation.py --dataset data/test_samples --rules data/yara_rules --model models/trained/random_forest_model.joblib --output-dir evaluation_results

This will:

Evaluate traditional YARA rule-based detection
Evaluate ML-only detection
Evaluate the enhanced YARA+ML approach
Generate comparison metrics and visualizations

Results
The evaluation compares three approaches:

Traditional YARA rule-based detection
Machine learning-based detection
Enhanced YARA with ML integration

Our results show:

YARA Only: Limited detection rate (typically 30-40%)
ML Only: High detection rate but potentially higher false positives
Enhanced YARA+ML: Maintains high detection rate while leveraging YARA's precision

The hybrid approach demonstrates a significant improvement in detection capabilities, especially for obfuscated and previously unseen malware samples.
Usage
To scan a file or directory:
bashpython enhanced_yara.py --file path/to/file.exe --rules data/yara_rules --model models/trained/random_forest_model.joblib
Or for batch scanning:
bashpython enhanced_yara.py --dir path/to/directory --rules data/yara_rules --model models/trained/random_forest_model.joblib --output results.json
Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
License
This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgments

YARA Project (https://virustotal.github.io/yara/)
Microsoft for providing the Malware Classification dataset
scikit-learn for the machine learning framework