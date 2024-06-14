import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

# Load the model, scaler, and dataset
model = joblib.load('models/gradient_boosting_model.pkl')
scaler = joblib.load('models/scaler.pkl')
data = pd.read_csv('datasets/cancer_risk_data.csv')

# Load header image
header_image = 'assets\cancer.jpg' 

# Streamlit app
st.image(header_image, use_column_width=True)
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

input_features = np.array([input_data[feature] for feature in ["Age", "Gender", "BMI", "Smoking", "GeneticRisk", "PhysicalActivity", "AlcoholIntake", "CancerHistory"]]).reshape(1, -1)

# Predict
if st.button("Predict"):
    features_scaled = scaler.transform(input_features)
    prediction = model.predict(features_scaled)
    diagnosis = "High Risk" if prediction[0] == 1 else "Low Risk"
    st.write(f"The predicted diagnosis is: {diagnosis}")
    
    # Visualize the features and their contributions
    st.write("### Feature Contributions")
    
    # Create a bar plot
    feature_names = ["Age", "Gender", "BMI", "Smoking", "GeneticRisk", "PhysicalActivity", "AlcoholIntake", "CancerHistory"]
    feature_values = input_features[0]
    
    fig, ax = plt.subplots()
    ax.barh(feature_names, feature_values, color='skyblue')
    ax.set_xlabel('Feature Value')
    ax.set_title('Feature Contributions to Prediction')
    st.pyplot(fig)
    
    # Show additional information
    if diagnosis == "High Risk":
        st.write("#### Tips for Reducing Cancer Risk:")
        st.write("""
        - Quit smoking.
        - Maintain a healthy weight.
        - Eat a diet rich in fruits and vegetables.
        - Exercise regularly.
        - Limit alcohol consumption.
        - Get regular medical care.
        """)
    else:
        st.write("#### Keep up the good work maintaining a healthy lifestyle!")
    
    # Generate downloadable report
    report = f"""
    Cancer Risk Prediction Report

    Predicted Diagnosis: {diagnosis}

    Input Features:
    Age: {age}
    Gender: {gender}
    BMI: {bmi}
    Smoking: {smoking}
    Genetic Risk: {genetic_risk}
    Physical Activity: {physical_activity} hours/week
    Alcohol Intake: {alcohol_intake} drinks/week
    Cancer History: {cancer_history}
    """
    b64 = base64.b64encode(report.encode()).decode()
    href = f'<a href="data:text/plain;base64,{b64}" download="cancer_risk_report.txt">Download Report</a>'
    st.markdown(href, unsafe_allow_html=True)

# Additional features

# Histogram of original data
st.write("### Distribution of Data")
fig, ax = plt.subplots(figsize=(12, 8))
data.hist(ax=ax, bins=30, layout=(3, 3), edgecolor='black', alpha=0.7)
plt.tight_layout(pad=2.0)
st.pyplot(fig)

# Correlation heatmap
st.write("### Feature Correlation Heatmap")
correlation_matrix = data.corr()
fig, ax = plt.subplots()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Educational content
st.write("### Educational Content")
st.write("""
Here are some resources to learn more about cancer prevention and healthy living:
- [American Cancer Society](https://www.cancer.org/)
- [World Health Organization](https://www.who.int/cancer/en/)
- [National Cancer Institute](https://www.cancer.gov/)
""")

# Consult To Doctor
st.write("### Consult To Your Doctor")
st.write("""
If you have another question about your sickness, let the doctor help you:
[Check Our Doctor](https://www.halodoc.com/)
         """)

# User feedback form
st.write("### User Feedback")
feedback = st.text_area("Please provide your feedback here:")
if st.button("Submit Feedback"):
    with open('user_feedback.txt', 'a') as f:
        f.write(feedback + '\n')
    st.write("Thank you for your feedback! It has been saved.")

