import streamlit as st
import pandas as pd
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "LogisticRegressio_heart.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler_heart.pkl"))
expected_columns = joblib.load(os.path.join(BASE_DIR, "columns.pkl"))

st.title("Heart Issue Prediction")
st.markdown("Provide The Following Details")


age = st.slider("Age" , 18 , 100 , 40)#(VariableName , low , max , DispalysetValue)

sex = st.selectbox("SEX", ["M" , "F"])

chest_pain = st.selectbox("Chest Pain Type",["ATA" , "NAP" , "TA" , "ASY"] )
resting_BP = st.number_input("Resting Blood Pressure (mm Hg)" , 80 , 200 ,120)
cholesterol = st.number_input("Cholesterol (mg/dL)" , 100 , 600 ,200)
fasting_Bs = st.selectbox("Fasting Blood Sugar  > 120 mg/dL" , [0,1])
resting_ecg = st.selectbox("Resting ECG" , ["Normal" , "ST" , "LVH"])
max_hr = st.slider("Max Heart Rate" , 60 , 220 , 150)
exercise_angina = st.selectbox("Exercise-Induced Angina" , ["Y", "N"])
old_peak = st.slider("OldPeak (ST Depression)" , 0.0 , 6.0 ,1.0)
st_slop = st.selectbox("ST Slope" , ["Up" ,"Flat" , "Down"])



if st.button("Predict"):
    raw_input = {
        "Age":age,
        "Resting_BP":resting_BP,
        "Cholesterol":cholesterol,
        "FastingBs":fasting_Bs,
        "MaxHR":max_hr,
        "OldPeak":old_peak,
        "Sex_" + sex: 1,
        "ChestPainType_" + chest_pain : 1,
        "RestingECG_" + resting_ecg : 1,
        "ExerciseAngina_"+ exercise_angina : 1 ,
        "ST_Slope_"+st_slop:1

    }

    input_df = pd.DataFrame([raw_input])

    for cols in expected_columns:
        if cols not in input_df:
            input_df[cols] = 0
    input_df = input_df[expected_columns]

    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]

    if prediction ==1:
        st.error("⚠️ High Risk of Heart Issue")
    else:
        st.success("✅ Low Risk of Heart Issue")

# input_df = pd.DataFrame(0, index=[0], columns=expected_columns)

# input_df["Age"] = age
# input_df["Resting_BP"] = resting_BP
# input_df["Cholesterol"] = cholesterol
# input_df["FastingBs"] = fasting_Bs
# input_df["MaxHR"] = max_hr
# input_df["OldPeak"] = old_peak

# input_df["Sex_" + sex] = 1
# input_df["ChestPainType_" + chest_pain] = 1
# input_df["RestingECG_" + resting_ecg] = 1
# input_df["ExerciseAngina_" + exercise_angina] = 1
# input_df["ST_Slope_" + st_slop] = 1
