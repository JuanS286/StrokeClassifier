import streamlit as st
import pickle
import numpy as np

# Load the model
model = pickle.load(open('../stroke-prediction-SVM_model-SMOTE-95accuracy.pkl', 'rb'))

# Define the predict function
def predict(model, input):
    return model.predict(input)

# Define the app
st.title('easyPredictor')
st.image('logo.png', width=256)

# Define inputs
age = st.number_input('Age', min_value=0, max_value=120, step=1)
hypertension = st.checkbox('Hypertension')
heart_disease = st.checkbox('Heart Disease')
avg_glucose_level = st.number_input('AVG Glucose Level', min_value=50.0, max_value=300.0, step=0.1)
bmi = st.number_input('BMI', min_value=0.0, max_value=100.0, step=0.1)
gender = st.radio('Gender', ('Female', 'Male', 'Other'))
work_type = st.radio('Work Type', ('Govt job', 'Never worked', 'Private', 'Self-employed', 'Children'))
smoking_status = st.radio('Smoking Status', ('Unknown', 'Formerly smoked', 'Never smoked', 'Smokes'))

# When 'Predict' is clicked, make the prediction and store it
if st.button('Predict'):
    input = np.array([age, hypertension, heart_disease, avg_glucose_level, bmi,  
                      gender == 'Female', gender == 'Male', gender == 'Other', 
                      work_type == 'Govt job', work_type == 'Never worked', work_type == 'Private', work_type == 'Self-employed', work_type == 'Children', 
                      smoking_status == 'Unknown', smoking_status == 'Formerly smoked', smoking_status == 'Never smoked', smoking_status == 'Smokes']).reshape(1, -1)
    prediction = predict(model, input)
    
    # Display a user-friendly message based on the prediction
    if prediction[0] == 0:
        st.write('Based on the provided data, the model predicts a **No Stroke Probability**.')
    else:
        st.write('Based on the provided data, the model predicts a **High Stroke Probability**.')
