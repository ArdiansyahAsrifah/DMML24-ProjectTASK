# app.py
import streamlit as st
from bmi_calculating import calculate_bmi, interpret_bmi

def main():
    st.title("BMI Calculator")

    weight = st.number_input("Enter your weight in kilograms", min_value=0.0, step=0.1)
    height = st.number_input("Enter your height in meters", min_value=0.0, step=0.01)

    if st.button("Calculate BMI"):
        bmi = calculate_bmi(weight, height)
        interpretation = interpret_bmi(bmi)

        st.write(f"Your BMI: {bmi:.2f}")
        st.write(f"Interpretation: {interpretation}")

if __name__ == "__main__":
    main()
