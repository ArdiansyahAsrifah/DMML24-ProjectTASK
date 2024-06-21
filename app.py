import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

# Load the model, scaler, and dataset
model = joblib.load('models/random_forest_model.pkl')
scaler = joblib.load('models/scaler.pkl')
data = pd.read_csv('datasets/cancer_risk_data.csv')

# Load header image
header_image = 'assets/cancer.jpg' 

# Streamlit app
st.image(header_image, use_column_width=True)
st.title("Cancer Risk Prediction")

st.write("This app predicts the likelihood of cancer based on various health metrics.")

# Sidebar with tabs
st.sidebar.title('Menu')
tabs = st.sidebar.radio('Navigation', ('Home', 'Predict', 'Visualization', 'Resources', 'Feedback'))

# Home tab
if tabs == 'Home':
    st.write("## Home")
    st.write("Welcome to Cancer Risk Prediction App.")
    st.write("Welcome to the Cancer Risk Prediction App, designed to assess the likelihood of cancer based on various health metrics.")
    st.write("Whether you're curious about your personal risk factors or seeking insights for preventive care, our app provides a straightforward prediction based on advanced statistical models. Simply input your age, gender, BMI, lifestyle choices, and medical history, and receive an instant assessment of your cancer risk.")
    st.write("Explore insightful visualizations in the Exploratory Data Analysis (EDA) section to understand the distribution and correlations within our dataset. Additionally, discover valuable resources for cancer prevention and healthy living in the Resources tab.")
    st.write("Start your journey towards informed health decisions today with the Cancer Risk Prediction App.")

# Prediction tab
elif tabs == 'Predict':
    st.write("## Predict")
    st.write("This section allows you to make predictions based on input features.")

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

# Visualization tab
elif tabs == 'Visualization':
    st.write("## Exploratory Data Analysis")
    
    # Histograms for numeric columns
    st.write("### Distribution of Numeric Columns")
    numeric_columns = ["Age", "BMI", "PhysicalActivity", "AlcoholIntake"]
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
    axes = axes.flatten()
    for i, col in enumerate(numeric_columns):
        sns.histplot(data[col], ax=axes[i], kde=True, bins=20, color='skyblue')
        axes[i].set_title(f'Distribution of {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')
    plt.tight_layout()
    st.pyplot(fig)

     # Density plots
    st.write("### Density Plot of Numeric Columns")
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
    axes = axes.flatten()
    for i, col in enumerate(numeric_columns):
        sns.kdeplot(data[col], ax=axes[i], shade=True, color='skyblue')
        axes[i].set_title(f'Density Plot of {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Density')
    plt.tight_layout()
    st.pyplot(fig)

     # Pair plot for numeric columns
    st.write("### Pair Plot of Numeric Columns")
    pair_plot = sns.pairplot(data[numeric_columns], corner=True, diag_kind='kde', plot_kws={'color': 'skyblue'})
    st.pyplot(pair_plot.fig) 

   # Heatmap for correlation
    st.write("### Correlation Heatmap")
    corr_matrix = data[numeric_columns].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    plt.figure(figsize=(10, 8))
    heatmap = sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', mask=mask, linewidths=0.5)
    st.pyplot(heatmap.get_figure())

     # Box plot for numeric columns
    st.write("### Box Plot of Numeric Columns")
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
    axes = axes.flatten()
    for i, col in enumerate(numeric_columns):
        sns.boxplot(x=data[col], ax=axes[i], color='skyblue')
        axes[i].set_title(f'Box Plot of {col}')
        axes[i].set_xlabel(col)
    plt.tight_layout()
    st.pyplot(fig)

# Resources tab
elif tabs == 'Resources':
    st.write("## Resources")
    st.write("""
    Here are some resources to learn more about cancer prevention and healthy living:
    - [American Cancer Society](https://www.cancer.org/)
    - [World Health Organization](https://www.who.int/cancer/en/)
    - [National Cancer Institute](https://www.cancer.gov/)
    """)

# Feedback tab
elif tabs == 'Feedback':
    st.write("## Feedback")
    feedback = st.text_area("Please provide your feedback here:")
    if st.button("Submit Feedback"):
        with open('user_feedback.txt', 'a') as f:
            f.write(feedback + '\n')
        st.write("Thank you for your feedback! It has been saved.")
