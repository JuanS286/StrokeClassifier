import streamlit as st
import pickle
import numpy as np

# Load the model and the scaler
model = pickle.load(open('../stroke-prediction-SVM_model-SMOTE-95accuracy.pkl', 'rb'))
scaler = pickle.load(open('../scaler.pkl', 'rb'))  # Load the scaler

# Define the predict function
def predict(model, scaler, input_data):
    # Scale the input data using the loaded scaler
    scaled_input = scaler.transform(input_data)
    return model.predict(scaled_input)

# Define the app
st.title('Stroke Risk Predictor')
st.image('logo.png', width=256)

# Input fields
age = st.number_input('Age', min_value=0, max_value=120, step=1)
hypertension = st.checkbox('Hypertension')
heart_disease = st.checkbox('Heart Disease')
avg_glucose_level = st.number_input('Average Glucose Level', min_value=50.0, max_value=300.0, step=0.1)
bmi = st.number_input('BMI', min_value=10.0, max_value=100.0, step=0.1)

# Categorical inputs
gender = st.radio('Gender', ['Male', 'Female', 'Other'])
work_type = st.radio('Work Type', ['Govt job', 'Never worked', 'Private', 'Self-employed', 'Children'])
smoking_status = st.radio('Smoking Status', ['Unknown', 'Formerly smoked', 'Never smoked', 'Smokes'])

# Button to predict
if st.button('Predict'):
    # Create input array in the order the model expects
    input_data = np.array([
        age,
        1 if hypertension else 0,
        1 if heart_disease else 0,
        avg_glucose_level,
        bmi,
        1 if gender == 'Female' else 0,
        1 if gender == 'Male' else 0,
        1 if gender == 'Other' else 0,
        1 if work_type == 'Govt job' else 0,
        1 if work_type == 'Never worked' else 0,
        1 if work_type == 'Private' else 0,
        1 if work_type == 'Self-employed' else 0,
        1 if work_type == 'Children' else 0,
        1 if smoking_status == 'Unknown' else 0,
        1 if smoking_status == 'Formerly smoked' else 0,
        1 if smoking_status == 'Never smoked' else 0,
        1 if smoking_status == 'Smokes' else 0,
    ]).reshape(1, -1)  # Reshape for a single sample

    # Make prediction
    prediction = predict(model, scaler, input_data)

    # Display result
    if prediction[0] == 0:
        st.write('Based on the provided data, the model predicts a **Low Stroke Probability**.')
    else:
        st.write('Based on the provided data, the model predicts a **High Stroke Probability**.')
