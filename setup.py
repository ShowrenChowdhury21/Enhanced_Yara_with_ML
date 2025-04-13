#!/usr/bin/env python3
from setuptools import setup, find_packages

setup(
    name="enhanced-yara-ml",
    version="0.1.0",
    description="Enhanced YARA with Machine Learning for improved malware detection",
    author="Student",
    author_email="student@example.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "jupyter>=1.0.0",
        "yara-python>=4.2.0",
        "joblib>=1.1.0",
        "tqdm>=4.62.0",
    ],
    python_requires=">=3.8",
)