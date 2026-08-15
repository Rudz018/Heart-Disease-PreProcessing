import streamlit as st
import pandas as pd
import joblib 

model=joblib.load("logistic_heart.pkl")
scaler=joblib.load("scaler.pkl")
expected_columns=joblib.load("columns.pkl")

st.title("Heart Disease Prediciton")
st.markdown("Enter your Details below:")
age=st.slider("Age",15,100,43)
sex=st.selectbox("Sex",["M","F"])
chest_pain=st.selectbox("Chest Pain Type",["Asymptomatic(ASY)","Non-Anginal Pain(NAP)",
                                           "Atypical Angina(ATA)"," Typical Angina(TA)"])

resting_bp=st.number_input("Resting Blood Pressure (mm Hg)",75,300,113)
cholestrol=st.number_input("Cholesterol (mg/dL)",120,1000,440)
fasting_bs=st.selectbox("Fasting Blood Sugar >120 mg/dL",["1","0"])
resting_ecg=st.selectbox("Resting Electrocardiogram",["Normal","ST-T Wave Abnormality","Left Ventricular Hypertrophy"])
max_hr=st.slider("Maximum Heart Rate",60,220,80)
excer_angina=st.selectbox("Exercise Induced Angina",["Y","N"])
oldpeak=st.selectbox("OldPeak(ST Depression)",[0.0,6.0,3.0])
st_slope=st.selectbox("ST Slope",["Up","Down","Flat"])

if st.button("Predict"):
    raw_input={
        "Age":age,
        "RestingBP":resting_bp,
        "Cholesterol": cholestrol,
        "FastingBS":fasting_bs,
        "OldPeak":oldpeak,
        "MaxHR":max_hr,
        "Sex_"+ sex : 1,
        "ChestPain_"+chest_pain:1,
        "RestingECG_"+resting_ecg:1,
        "ExerciseAngina_"+excer_angina:1,
        "ST_Slope_"+st_slope:1
        
    }

input_df=pd.DataFrame([raw_input])

for col in expected_columns:
    if col not in input_df:
        input_df[col]=0

input_df=input_df[expected_columns]
scaled_input=scaler.transform(input_df)
prediction=model.predict(scaled_input)[0]

if prediction==1:
    st.error("🚨High Risk of Heart Disease🚨")
else:
    st.success("🙌Low Risk of Heart Disease🙌")

