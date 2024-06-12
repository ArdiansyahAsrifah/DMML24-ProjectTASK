import streamlit as st
import numpy as np
import joblib

# Load the model and scaler
model = joblib.load('src/logistic_regression_model.pkl')
scaler = joblib.load('src/scaler.pkl')

# Streamlit app
st.title("Cancer Risk Prediction")

st.write("This app predicts the likelihood of cancer based on various health metrics.")

# Input features
age = st.slider("Age", 0, 100, 50)
gender = st.selectbox("Gender", ["Male", "Female"])
bmi = st.slider("BMI", 10.0, 60.0, 20.0)
smoking = st.selectbox("Smoking", ["Non-smoker", "Smoker"])
genetic_risk = st.selectbox("Genetic Risk", ["Low", "Medium", "High"])
physical_activity = st.slider("Physical Activity (hours/week)", 0, 20, 5)
alcohol_intake = st.slider("Alcohol Intake (drinks/week)", 0, 20, 5)
cancer_history = st.selectbox("Cancer History", ["No", "Yes"])

# Convert input features to appropriate format
input_data = {
    "Age": age,
    "Gender": 1 if gender == "Male" else 0,
    "BMI": bmi,
    "Smoking": 1 if smoking == "Smoker" else 0,
    "GeneticRisk": {"Low": 0, "Medium": 1, "High": 2}[genetic_risk],
    "PhysicalActivity": physical_activity,
    "AlcoholIntake": alcohol_intake,
    "CancerHistory": 1 if cancer_history == "Yes" else 0,
}

input_features = np.array([input_data[feature] for feature in [
    "Age", 
    "Gender", 
    "BMI", 
    "Smoking", 
    "GeneticRisk", 
    "PhysicalActivity", 
    "AlcoholIntake", 
    "CancerHistory"]]).reshape(1, -1)

# Predict
if st.button("Predict"):
    features_scaled = scaler.transform(input_features)
    prediction = model.predict(features_scaled)
    diagnosis = "High Risk" if prediction[0] == 1 else "Low Risk"
    st.write(f"The predicted diagnosis is: {diagnosis}")
