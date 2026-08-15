import streamlit as st
import pandas as pd
import joblib

# Load saved model and preprocessing objects
model = joblib.load("logistic_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

# App title
st.title("Heart Disease Prediction")
st.markdown("Enter your details below:")

# User inputs
age = st.slider("Age", 15, 100, 43)

sex = st.selectbox(
    "Sex",
    ["M", "F"]
)

chest_pain = st.selectbox(
    "Chest Pain Type",
    [
        "Asymptomatic(ASY)",
        "Non-Anginal Pain(NAP)",
        "Atypical Angina(ATA)",
        "Typical Angina(TA)"
    ]
)

resting_bp = st.number_input(
    "Resting Blood Pressure (mm Hg)",
    75,
    300,
    113
)

cholesterol = st.number_input(
    "Cholesterol (mg/dL)",
    120,
    1000,
    440
)

fasting_bs = st.selectbox(
    "Fasting Blood Sugar >120 mg/dL",
    ["Yes", "No"]
)

resting_ecg = st.selectbox(
    "Resting Electrocardiogram",
    [
        "Normal",
        "ST-T Wave Abnormality",
        "Left Ventricular Hypertrophy"
    ]
)

max_hr = st.slider(
    "Maximum Heart Rate",
    60,
    220,
    80
)

exercise_angina = st.selectbox(
    "Exercise Induced Angina",
    ["Yes", "No"]
)

oldpeak = st.selectbox(
    "OldPeak (ST Depression)",
    [0.0, 6.0, 3.0]
)

st_slope = st.selectbox(
    "ST Slope",
    ["Up", "Down", "Flat"]
)

st.caption(
    "Default values are provided as example inputs. "
    "Replace them with known values when available."
)

# Prediction
if st.button("Predict"):

    raw_input = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": 1 if fasting_bs == "Yes" else 0,
        "OldPeak": oldpeak,
        "MaxHR": max_hr,
        "Sex_" + sex: 1,
        "ChestPain_" + chest_pain: 1,
        "RestingECG_" + resting_ecg: 1,
        "ExerciseAngina_" + ("Y" if exercise_angina == "Yes" else "N"): 1,
        "ST_Slope_" + st_slope: 1
    }

    input_df = pd.DataFrame([raw_input])

    # Add missing columns required by the model
    for col in expected_columns:
        if col not in input_df:
            input_df[col] = 0

    # Arrange columns in the exact order used during training
    input_df = input_df[expected_columns]

    # Apply saved scaler
    scaled_input = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(scaled_input)[0]

    if prediction == 1:
        st.error("🚨 High Risk of Heart Disease 🚨")
    else:
        st.success("🙌 Low Risk of Heart Disease 🙌")
