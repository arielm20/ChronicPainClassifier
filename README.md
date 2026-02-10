# EEG Frequency Analysis of Chronic Pain Using Machine Learning

This project has two aims (1) Identify frequency-band based EEG markers of 3 different chronic pain conditions: Chronic Back Pain (CBP), Fibromyalgia (FM), and Nocioceptive Chronic Pain (NCCP) and (2) Use these markers to develop multiclass Machine Learning classifiers.

Data for this project is based on data from the following article "Exploring electroencephalographic chronic pain biomarkers: a mega-analysis" by Bott et al., 2025 which can be found in the repository. In addition, the EEG dataset can be found here: https://osf.io/srpbg/overview. Finally, this project is based on the the ADClassifier by itayinbarr on github (repo: https://github.com/itayinbarr/ADclassifier/tree/main)

## Overview
The project analyzes EEG data to:
    1. Identify characteristic frequencies that distinguish between FM, CBP, and NCCP patients
    2. Validate ML models for chronic pain detection
    3. Analyze the trade-offs between different classification approaches

## Repository Structure
.
├── data/                                   # EEG Data directory is not included in repo
│   ├── processed_data_rel_all_pain.csv     # frequency band power for each band and region
│   ├── feature_importance.csv              # mutual information results
│   ├── X_ml.csv
│   └── y_ml.csv
├── results/                   # Feature exploration results and model outputs
│   ├── models/
│   ├── plots/
│   └── metrics/
├── EEGpreprocessing.py       # Initial data preprocessing pipeline (filtering, ICA, and frequency power calculations)
├── feature_analysis.py       # Feature analysis and exploration
├── FeatureEngineering.py     # Feature selection and engineering
├── PainModel.py              # Model training and evaluation
├── report.md                 # Detailed analysis report
└── README.md                 # This file

## Main Findings
The analysis identified several frequency bands that distinguish FM from NCCP and CBP:
    
    1. Delta (1-3 Hz) - Global
    2. Theta (3-5 Hz) - Global

The best performing model (XGBoost) achieved the following results for multi-class classification:

    1. Balanced accuracy: 79%
    2. F1 Score: 79%
    3. ROC-AUC: 0.88

However, for classifying FM vs CBP and NCCP the following results were achieved:

    1. Precision: 96%
    2. Recall: 100%
    3. F1 Score: 98%

Considering the features for CBP and NCCP overlapped, these results make sense. A detailed report can be found in the report.md file.

## Requirements
Python 3.8+
Required packages:
    - pandas
    - numpy
    - scikit-learn
    - xgboost
    - catboost
    - optuna
    - imbalanced-learn