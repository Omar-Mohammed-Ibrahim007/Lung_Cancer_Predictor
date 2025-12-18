# Lung Cancer Risk Prediction System

A comprehensive Machine Learning–based Lung Cancer Risk Prediction System built using synthetic yet realistic healthcare data, advanced preprocessing, multiple classification models, and rigorous hyperparameter tuning with GridSearchCV.
The project includes a fully interactive Streamlit web application for real-time prediction using pretrained models.

## Project Overview

Lung cancer remains one of the leading causes of cancer-related deaths worldwide. Early risk assessment can significantly improve patient outcomes.

This project aims to:

Simulate realistic clinical and lifestyle data

Train and compare multiple ML models

Reduce overfitting through proper feature engineering and tuning

Deploy a user-friendly Streamlit web app for prediction

## Objectives

Generate a balanced, realistic dataset with missing values and outliers

Apply robust preprocessing (encoding, scaling, handling noise)

Train multiple ML classifiers

Perform mandatory hyperparameter tuning using GridSearchCV

Compare models before and after tuning

Deploy predictions via an interactive web interface

## Machine Learning Models Used

Logistic Regression

K-Nearest Neighbors (KNN)

Decision Tree

Random Forest

Extra Trees

XGBoost

LightGBM

Naive Bayes variants

Each model was trained using:

Default hyperparameters

Tuned hyperparameters (GridSearchCV)

## Dataset Description

The dataset is synthetically generated to resemble real-world lung cancer data while avoiding privacy issues.

Features Include:

Demographics (Age, Gender, Country)

Lifestyle factors (Smoking, Tobacco Use, Indoor Smoking)

Environmental exposure (Air Pollution, Occupational Exposure)

Clinical indicators (Cancer Stage, Mutation Type, Treatment)

Risk indicators (Mortality Risk, 5-Year Survival Probability)

Target Variable:

Final_Prediction → Yes / No (lung cancer risk)

✔ Balanced class distribution
✔ Missing values (~10%)
✔ Outliers (~2%)
✔ Label noise for realism

## Data Preprocessing Pipeline

Missing Values Handling

Numerical: Median imputation

Categorical: Mode / probability-based fill

Outlier Treatment

IQR-based filtering for numerical features

Encoding Strategy

OrdinalEncoder → ordered features (e.g., cancer stage)

One-Hot Encoding → nominal features

Feature Scaling

StandardScaler (trained on training set only)

All preprocessing artifacts are saved and reused during inference.

## Hyperparameter Tuning (Mandatory)

Hyperparameter tuning was performed using GridSearchCV for every model (default parameters were not allowed).

Why GridSearchCV?

Prevents overfitting

Ensures fair model comparison

Selects best model based on validation performance, not training accuracy

Tuned Hyperparameters Include:

Regularization strength (C)

Tree depth (max_depth)

Number of estimators (n_estimators)

Learning rate (learning_rate)

Number of neighbors (k)

Feature randomness (max_features)

## Model Selection Criteria

Final model versions were selected based on:

Validation accuracy

Reduced overfitting

Stability between training and validation scores

Cross-validation consistency

## Performance Comparison
Model	Default Accuracy	Tuned Accuracy
Logistic Regression	~0.81	~0.85
Random Forest	~0.86	~0.90
XGBoost	~0.88	~0.92
LightGBM	~0.87	~0.91

✔ All models showed improvement after tuning
✔ Ensemble models performed best

## Feature Importance & Interpretability

Logistic Regression coefficients analyzed

Odds Ratios calculated for interpretability

Feature importance visualizations included

This ensures model transparency, especially critical for healthcare-related predictions.

## Streamlit Web Application

An interactive web app allows users to:

Select a pretrained model

Input patient data via sidebar

Perform real-time prediction

View risk probability and final decision

## Features:

Cached models and encoders

Proper feature alignment

Safe handling of unseen inputs

Clean UI and responsive layout

📁 Project Structure
├── data/
│   └── synthetic_cancer_data.csv
├── models/
│   ├── lr.pkl
│   ├── rf.pkl
│   ├── xgb.pkl
│   └── lgb.pkl
├── encoders/
│   ├── ordinal_encoder.pkl
│   └── label_encoder.pkl
├── scaler.pkl
├── features.txt
├── app.py
├── training_notebook.ipynb
└── README.md

 Disclaimer

This project is for academic and research purposes only.
It is not a medical diagnostic tool and should not be used for real clinical decisions.

 Author

Omar Mohammad
Computer Science Student 

 Final Notes

This project demonstrates:

Strong understanding of ML pipelines

Proper evaluation methodology

Realistic data simulation

Professional deployment practices

