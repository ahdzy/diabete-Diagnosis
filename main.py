
import tkinter as tk
from tkinter import messagebox
import joblib
import numpy as np
import pandas as pd


Model = joblib.load('model.joblib')


# Create and arrange labels, Entry widgets, and their respective labels using grid
gender = "Female"
age = 51
hypertension = 1
heart_disease = 1
smoking_history = "never"
bmi = 25.19
HbA1c_level = 6.6
blood_glucose_level = 140
final_out = 0


loaded_model = joblib.load('model.joblib')


X_test = pd.DataFrame({
    'gender': [gender],
    'age': [age],
    'hypertension': [hypertension],  # Change to 1 if it's more likely to result in a prediction of 1
    'heart_disease': [heart_disease],  # Change to 1 if it's more likely to result in a prediction of 1
    'smoking_history': [smoking_history],  # Change to 'ever' if it's more likely to result in a prediction of 1
    'bmi': [bmi],  # Change to a value that is more likely to result in a prediction of 1
    'HbA1c_level': [HbA1c_level],
    'blood_glucose_level': [blood_glucose_level]
})


# Make predictions
final_out = loaded_model.predict(X_test)
predictions_prop = loaded_model.predict_proba(X_test)

# Print or use the predictions as needed
print(final_out, predictions_prop)
