# Heart Disease Prediction ML App

An end-to-end machine learning project that analyzes a heart disease dataset, performs data preprocessing and feature selection, compares multiple classification models, and deploys the selected model through an interactive Streamlit web application.

## Project Overview

The goal of this project is to build a complete machine learning workflow for predicting the presence of heart disease from patient-related clinical attributes.

The project covers the process from raw data preprocessing to model evaluation and finally to an interactive prediction interface.

### Workflow

**Raw Dataset → Data Cleaning → Encoding → Feature Analysis → Feature Selection → Scaling → Model Training → Model Evaluation → Model Selection → Streamlit Application**

## Objectives

* Understand and inspect the heart disease dataset
* Clean and preprocess the raw data
* Handle invalid values and duplicate records
* Convert categorical variables into numerical representations
* Analyze feature relationships and distributions
* Perform feature selection using statistical techniques
* Compare scaled and unscaled machine learning models
* Evaluate multiple classification algorithms
* Select the most suitable model based on evaluation results
* Build an interactive interface for making predictions

## Dataset

The project uses a heart disease dataset containing clinical and demographic attributes.

### Main Features

* `Age` — Age of the patient
* `Sex` — Sex of the patient
* `ChestPainType` — Type of chest pain
* `RestingBP` — Resting blood pressure
* `Cholesterol` — Cholesterol level
* `FastingBS` — Fasting blood sugar
* `RestingECG` — Resting electrocardiogram result
* `MaxHR` — Maximum heart rate achieved
* `ExerciseAngina` — Exercise-induced angina
* `Oldpeak` — ST depression
* `ST_Slope` — Slope of the peak exercise ST segment

### Target

* `HeartDisease` — Indicates whether heart disease is present

`HeartDisease = 0` represents no heart disease and `HeartDisease = 1` represents the presence of heart disease.

## Data Preprocessing

The raw dataset was inspected and prepared for machine learning through several preprocessing steps.

### Data Inspection

The dataset was examined using:

* Dataset shape
* Data types
* Summary statistics
* Missing/null value checks
* Duplicate checks
* Feature distributions

### Categorical Encoding

Categorical features were converted into numerical representations.

* Binary categorical variables were encoded numerically.
* Multi-class categorical variables were converted using one-hot encoding.

### Invalid Value Handling

Invalid zero values were investigated in numerical features where zero is not a meaningful physiological measurement, including:

* `RestingBP`
* `Cholesterol`

These values were handled using appropriate replacement techniques based on the available non-zero observations.

### Feature Analysis

The project includes exploratory and statistical analysis using:

* Correlation analysis
* Distribution plots
* Variance analysis
* ANOVA F-test
* `SelectKBest` feature selection

These techniques were used to understand the relationship between the input features and the target variable and to identify potentially useful features.

## Machine Learning Models

Multiple classification algorithms were trained and evaluated.

The project compares both **scaled and unscaled** feature versions.

Models evaluated include:

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Decision Tree
* Random Forest
* Support Vector Machine (SVM)

Model performance was compared using multiple evaluation metrics rather than relying only on accuracy.

### Evaluation Metrics

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

The classification notebook contains the detailed evaluation results and explains the reasoning behind the final model selection.

## Model Selection

After comparing the different models and preprocessing configurations, **Scaled Logistic Regression** was selected as the final model for deployment based on its overall evaluation performance in the experiments.

The final trained model is saved and used by the Streamlit application for prediction.

## Streamlit Application

The project includes an interactive Streamlit interface that allows users to enter patient information and receive a model prediction.

### Application Workflow

1. User enters patient-related information.
2. Input values are converted into the required feature format.
3. The same preprocessing/scaling configuration used during model training is applied.
4. The saved Logistic Regression model processes the input.
5. The application displays the prediction result.

The interface also provides sensible pre-filled example values so that users can test the application even when they do not know every input value.

> **Important:** This application is intended for educational and demonstration purposes only. It is not a medical diagnostic tool and should not be used to make real medical decisions.

## Technologies Used

* **Python**
* **Pandas** — Data manipulation and preprocessing
* **NumPy** — Numerical computation
* **Matplotlib** — Data visualization
* **Seaborn** — Statistical visualization
* **Scikit-learn** — Preprocessing, feature selection, model training and evaluation
* **Joblib** — Saving and loading trained model objects
* **Streamlit** — Interactive web application

## Repository Structure

```text
Heart-Disease-Prediction-ML-App/
│
├── app.py
├── Heart_PreProcessing.ipynb
├── Classification.ipynb
├── heart_raw.csv
├── heart_cleaned.csv
├── logistic_heart.pkl
├── scaler.pkl
├── expected_columns.pkl
├── requirements.txt
└── README.md
```

## Project Files

### `Heart_PreProcessing.ipynb`

Contains the data inspection, cleaning, encoding, visualization, feature analysis, scaling and feature selection workflow.

### `Classification.ipynb`

Contains the machine learning experiments, comparison of scaled and unscaled models, evaluation metrics and final model selection.

### `app.py`

Streamlit application that loads the trained model and allows users to make predictions through an interactive interface.

### `heart_raw.csv`

Original dataset before preprocessing.

### `heart_cleaned.csv`

Processed dataset generated after preprocessing.

### `logistic_heart.pkl`

Saved trained Logistic Regression model used by the application.

### `scaler.pkl`

Saved feature scaler used to apply the same scaling procedure during prediction.

### `expected_columns.pkl`

Saved feature-column structure used to ensure user input matches the format expected by the trained model.

## Installation

Clone the repository:

```bash
git clone https://github.com/Rudz018/Heart-Disease-Prediction-ML-App.git
cd Heart-Disease-Prediction-ML-App
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Run the Application

Start the Streamlit application using:

```bash
streamlit run app.py
```

The application will open in your browser.

## Machine Learning Workflow

The complete project follows this pipeline:

```text
Raw Heart Disease Dataset
          ↓
Data Inspection
          ↓
Data Cleaning
          ↓
Categorical Encoding
          ↓
Invalid Value Handling
          ↓
Exploratory Data Analysis
          ↓
Correlation & Variance Analysis
          ↓
ANOVA / Feature Selection
          ↓
Feature Scaling
          ↓
Multiple Classification Models
          ↓
Model Evaluation
          ↓
Model Comparison
          ↓
Scaled Logistic Regression
          ↓
Saved Model
          ↓
Streamlit Prediction Application
```

## Future Improvements

Possible future improvements include:

* Hyperparameter tuning
* Cross-validation
* Additional model comparison
* Improved input validation
* Model explainability
* Probability-based prediction display
* Deployment to a public cloud platform
* Adding a prediction history feature
* Improving UI and visualization

## Disclaimer

This project is created for educational and machine learning demonstration purposes.

The predictions produced by the application should not be considered medical advice, diagnosis, or treatment recommendations.

## Author

**Rudra Wagh**

GitHub: [Rudz018](https://github.com/Rudz018)
