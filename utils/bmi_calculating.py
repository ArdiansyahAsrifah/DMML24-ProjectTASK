# bmi_calculator.py

def calculate_bmi(weight, height):
    """
    Function to calculate Body Mass Index (BMI).
    Formula: BMI = weight / (height * height)
    weight: weight in kilograms (float)
    height: height in meters (float)
    returns: BMI value (float)
    """
    bmi = weight / (height * height)
    return bmi

def interpret_bmi(bmi):
    """
    Function to interpret BMI value.
    bmi: BMI value (float)
    returns: interpretation (string)
    """
    if bmi < 18.5:
        return "Underweight"
    elif bmi >= 18.5 and bmi < 25:
        return "Normal weight"
    elif bmi >= 25 and bmi < 30:
        return "Overweight"
    else:
        return "Obesity"

if __name__ == "__main__":
    weight = float(input("Enter your weight in kilograms: "))
    height = float(input("Enter your height in meters: "))

    bmi = calculate_bmi(weight, height)
    interpretation = interpret_bmi(bmi)

    print(f"Your BMI is: {bmi:.2f}")
    print(f"Interpretation: {interpretation}")
