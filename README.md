# Heart Disease Dataset — Data Preprocessing & Feature Selection

## Overview

This project focuses on cleaning, preprocessing, exploring, and selecting features from a heart disease dataset.

The objective is to transform the raw dataset into a cleaner and more suitable format for further machine learning analysis.

The project covers data inspection, categorical encoding, handling invalid values, standardization, correlation analysis, variance analysis, and ANOVA-based feature selection.

## Objectives

- Inspect and understand the raw dataset
- Identify missing, null, duplicate, and invalid values
- Convert categorical variables into numerical representations
- Handle invalid zero values in relevant numerical features
- Standardize numerical features
- Analyze feature relationships using correlation
- Examine feature variance
- Perform ANOVA-based feature selection
- Generate a cleaned dataset for further machine learning tasks

## Dataset

The dataset contains patient-related attributes and a target variable indicating the presence of heart disease.

The main target variable is:

- `HeartDisease` — target indicating whether heart disease is present

The dataset includes features related to:

- Age
- Sex
- Chest pain type
- Resting blood pressure
- Cholesterol
- Fasting blood sugar
- Resting ECG
- Maximum heart rate
- Exercise-induced angina
- ST depression
- ST slope

## Preprocessing Performed

### 1. Initial Data Inspection

The dataset was examined using:

- `head()`
- `info()`
- `isnull()`
- `isna()`
- `describe()`
- `shape`
- duplicate checking

### 2. Categorical Encoding

Categorical variables were converted into numerical representations.

- `Sex` was converted into a binary variable
- `ExerciseAngina` was converted into a binary variable
- `ChestPainType` was converted using one-hot encoding
- `RestingECG` was converted using one-hot encoding
- `ST_Slope` was converted using one-hot encoding

### 3. Handling Invalid Values

Zero values were investigated in:

- `Cholesterol`
- `RestingBP`

Since a person cannot have zero "RestingBP" nor "Cholestrerol", added mean values instead of zero

### 4. Data Visualization

The project uses visualizations to examine:

- Feature distributions
- Correlations between numerical features
- Cholesterol and resting blood pressure distributions
- Relationships between selected numerical features

Libraries used for visualization include Matplotlib and Seaborn.

### 5. Standardization

Numerical features were standardized using `StandardScaler` from Scikit-learn.

### 6. Feature Analysis

Several techniques were used to investigate the usefulness of the features:

- Pearson correlation analysis
- Variance analysis
- ANOVA F-test

`SelectKBest` with `f_classif` was used to calculate F-scores and p-values for the features.

### 7. Final Dataset

After preprocessing and feature analysis, the cleaned dataset was exported as:

`heart_cleaned.csv`

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Google Collab/Jupyter Notebook

## Project Structure

```text
heart-disease-preprocessing/
│
├── Heart_PreProcessing.ipynb
├── heart_raw.csv
├── heart_clean.csv
├── README.md
└── requirements.txt
