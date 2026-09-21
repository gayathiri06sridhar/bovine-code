import joblib
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def get_float_input(prompt):
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Invalid input. Please enter a number.")

def get_int_input(prompt):
    while True:
        try:
            return int(input(prompt))
        except ValueError:
            print("Invalid input. Please enter an integer.")

def main():
    print("========================================")
    print("Bovine Mastitis ML Prediction System")
    print("========================================\n")
    
    try:
        model = joblib.load("models/mastitis_model_pipeline.joblib")
    except FileNotFoundError:
        print("Error: Trained model not found. Please run ml_pipeline.py first.")
        return

    print("Please enter the following cow parameters:")
    
    day = get_int_input("Day (integer): ")
    temp = get_float_input("Milk Temperature (°C): ")
    ph = get_float_input("Milk pH: ")
    cond = get_float_input("Milk Conductivity: ")
    scc = get_int_input("Somatic Cell Count: ")
    yield_val = get_float_input("Milk Yield (liters): ")
    clotting = get_int_input("Clotting (0 or 1): ")
    
    feature_cols = [
        "Day", "Milk_Temperature", "Milk_pH", "Milk_Conductivity",
        "Somatic_Cell_Count", "Milk_Yield", "Clotting"
    ]
    
    input_data = pd.DataFrame([[
        day, temp, ph, cond, scc, yield_val, clotting
    ]], columns=feature_cols)
    
    print("\nProcessing...")
    
    prob = model.predict_proba(input_data)[0, 1]
    prediction = model.predict(input_data)[0]
    
    risk_level = "HIGH RISK (Mastitis Detected)" if prediction == 1 else "LOW RISK (Healthy)"
    
    print("\n========================================")
    print("Prediction Results")
    print("========================================")
    print(f"Probability of Mastitis : {prob * 100:.2f}%")
    print(f"Risk Assessment         : {risk_level}")
    print("========================================\n")

if __name__ == "__main__":
    main()
