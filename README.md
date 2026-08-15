# Heart Disease Prediction ML App

An end-to-end machine learning project that preprocesses a heart disease dataset, performs feature analysis and selection, compares multiple classification models on scaled and unscaled data, and deploys the selected model through an interactive Streamlit web application.

## 🚀 Live Demo

[Try the Heart Disease Prediction App](https://rw18-heart-disease-prediction-ml-app-nkwjurecjtwj2pr8ursmxa.streamlit.app/)

## Project Overview

The goal of this project is to build a complete machine learning workflow for predicting the presence of heart disease from patient-related clinical attributes.

The project progresses from raw data preprocessing and feature analysis to model training, evaluation, model selection, and deployment.

### Workflow

**Raw Dataset → Data Cleaning → Encoding → Feature Analysis → Feature Selection → Scaling → Model Training → Evaluation → Model Comparison → Model Selection → Streamlit Application**

## Objectives

* Inspect and understand the heart disease dataset
* Clean and preprocess the raw data
* Handle invalid values and duplicate records
* Convert categorical variables into numerical representations
* Analyze feature distributions and relationships
* Perform statistical feature selection
* Compare scaled and unscaled data for multiple classification models
* Evaluate models using multiple performance metrics
* Select a suitable model based on the experimental results
* Deploy the selected model through an interactive Streamlit application

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

### Data Inspection

The raw dataset was examined for:

* Dataset shape
* Data types
* Summary statistics
* Missing and null values
* Duplicate records
* Feature distributions

### Categorical Encoding

Categorical variables were converted into numerical representations.

* Binary categorical variables were encoded numerically.
* Multi-class categorical variables were converted using one-hot encoding.

### Invalid Value Handling

Zero values were investigated in features where zero is not a meaningful physiological measurement, including:

* `RestingBP`
* `Cholesterol`

These values were handled using the mean of the corresponding non-zero observations.

### Exploratory and Statistical Analysis

The project includes:

* Distribution analysis
* Correlation analysis
* Variance analysis
* ANOVA F-test
* `SelectKBest` feature selection

These methods were used to understand feature relationships and investigate the statistical relevance of the available features.

## Machine Learning Models

Five classification algorithms were trained and evaluated:

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Naive Bayes
* Decision Tree
* Support Vector Machine (SVM)

Each model was evaluated using both **unscaled and scaled** feature data.

## Model Evaluation

The models were evaluated using:

* Accuracy
* F1-score
* Recall

### Results — Not Scaled

| Model               | Accuracy | F1-score | Recall |
| ------------------- | -------: | -------: | -----: |
| Logistic Regression |   84.78% |   86.54% | 84.11% |
| KNN                 |   67.39% |   71.70% | 71.03% |
| Naive Bayes         |   84.24% |   85.99% | 83.18% |
| Decision Tree       |   79.35% |   81.19% | 76.64% |
| SVM                 |   66.30% |   69.61% | 66.36% |

### Results — Scaled

| Model               |   Accuracy |   F1-score |     Recall |
| ------------------- | ---------: | ---------: | ---------: |
| Logistic Regression | **85.33%** | **87.20%** |     85.98% |
| KNN                 |     84.78% |     86.79% |     85.98% |
| Naive Bayes         |     84.24% |     85.99% |     83.18% |
| Decision Tree       |     78.26% |     80.95% |     79.44% |
| SVM                 |     84.24% |     86.51% | **86.92%** |

## Model Comparison and Selection

The experiments show that feature scaling affected different algorithms differently.

* **Logistic Regression** improved after scaling, increasing accuracy from **84.78% to 85.33%** and F1-score from **86.54% to 87.20%**.
* **KNN** showed a major improvement after scaling, increasing accuracy from **67.39% to 84.78%**.
* **SVM** also showed a major improvement, increasing accuracy from **66.30% to 84.24%**.
* **Naive Bayes** produced the same results with and without scaling in this experiment.
* **Decision Tree** performed slightly better without scaling.

Based on the overall results, **Scaled Logistic Regression** was selected as the final model because it achieved the highest accuracy and F1-score among the tested configurations.

Although scaled SVM achieved the highest recall, Logistic Regression provided the strongest overall balance among the metrics used for model selection.

## Streamlit Application

The selected Logistic Regression model was integrated into an interactive Streamlit application.

### Application Workflow

1. User enters patient-related information.
2. Input values are converted into the required feature format.
3. The saved feature-column structure is used to align the input.
4. The saved scaler transforms the input using the same scaling process used during training.
5. The saved Logistic Regression model generates the prediction.
6. The result is displayed through the Streamlit interface.

The application also provides pre-filled example values so that users can test the interface even when they do not know every input value.

> **Important:** This application is intended for educational and demonstration purposes only. It is not a medical diagnostic tool and should not be used for real medical decisions.

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
Heart-Disease-Prediction-ML-APP/
│
├── Heart_PreProcessing.ipynb
├── ML_Classification.ipynb
├── app.py
├── heart_raw.csv
├── heart_cleaned.csv
├── logistic_heart.pkl
├── scaler.pkl
├── columns.pkl
├── requirements.txt
└── README.md
```

## Project Files

### `Heart_PreProcessing.ipynb`

Contains the data inspection, cleaning, encoding, visualization, feature analysis, scaling and feature selection workflow.

### `ML_Classification.ipynb`

Contains the machine learning experiments, comparison of scaled and unscaled models, evaluation metrics, model comparison and final model selection.

### `app.py`

Contains the Streamlit application used to generate predictions from user-provided inputs.

### `heart_raw.csv`

Original dataset before preprocessing.

### `heart_cleaned.csv`

Processed dataset generated after preprocessing.

### `logistic_heart.pkl`

Saved trained Logistic Regression model used by the Streamlit application.

### `scaler.pkl`

Saved feature scaler used to apply the same scaling procedure during prediction.

### `columns.pkl`

Saved feature-column structure used to ensure the input data matches the format expected by the trained model.

## Installation

Clone the repository:

```bash
git clone https://github.com/Rudz018/Heart-Disease-Prediction-ML-APP.git
cd Heart-Disease-Prediction-ML-APP
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

## Machine Learning Workflow

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
Scaled & Unscaled Data
          ↓
Multiple Classification Models
          ↓
Model Evaluation
          ↓
Model Comparison
          ↓
Scaled Logistic Regression Selected
          ↓
Saved Model
          ↓
Streamlit Prediction Application
```

## Future Improvements

Possible future improvements include:

* Cross-validation
* Hyperparameter tuning
* Model explainability
* Probability/confidence display
* Improved input validation
* Prediction history
* Additional UI and visualization improvements

## Disclaimer

This project is created for educational and machine learning demonstration purposes.

The predictions produced by the application should not be considered medical advice, diagnosis, or treatment recommendations.

## Author

**Rudra Wagh**

GitHub: [Rudz018](https://github.com/Rudz018)
