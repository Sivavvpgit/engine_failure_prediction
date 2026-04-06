import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the trained model
model_path = hf_hub_download(repo_id="Sivavvp/engine_failure_predictive_model", filename="best_engine_failure_prediction_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI
st.title("Engine failure Prediction App")
st.write("""
This application predicts the vehciles engine is faulty or not based on RPM, Fuel pressure, Coolant pressure, Lub Oil Tempratire etc. 
Please enter the app details below to get a prediction.
""")

# User input

Engine_rpm = st.number_input("Engine rpm", min_value=1, max_value=3000, value=1, step=1)
Lub_oil_pressure = st.number_input("Lub oil pressure", min_value=1, max_value=8, value=1, step=1)
Fuel_pressure = st.number_input("Fuel pressure", min_value=0, max_value=22, value=1, step=1)
Coolant_pressure = st.number_input("Coolant pressure", min_value=0, max_value=7, step=1)
lub_oil_temp = st.number_input("lub oil temp", min_value=70, max_value=90, value=1, step=1)
Coolant_temp = st.number_input("Coolant temp", min_value=60, max_value=195, value=1, step=1)


# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Engine rpm': Engine_rpm,
    'Lub oil pressure': Lub_oil_pressure,
    'Fuel pressure': Fuel_pressure,
    'Coolant pressure': Coolant_pressure,
    'lub oil temp': lub_oil_temp,
    'Coolant temp': Coolant_temp      
}])

# Predict button
if st.button("Predict Engine"):
    proba = model.predict_proba(input_data)[0][1]
    prediction = 1 if proba >= 0.4 else 0
    result = "Engine might fail" if prediction == 1 else "Engine is good"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
