# Import necessary libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import numpy as np

# Load the TensorFlow model
@st.cache_data
def load_model(model_path):
    try:
        return tf.keras.models.load_model(model_path)
    except FileNotFoundError:
        st.error("Model file not found. Please check the file path.")
        return None

# Load the data (ensure pandas is installed)
@st.cache_data
def load_data(data_path):
    try:
        return pd.read_csv(data_path)
    except FileNotFoundError:
        st.error("Data file not found. Please check the file path.")
        return None

# Preprocess user input to match model's training data
def preprocess_input(gender, age, hypertension, heart_disease, smoking_history, bmi, hba1c, blood_glucose):
    # Convert and encode inputs as per the training data format
    gender = 1 if gender == 'Male' else 0
    hypertension = 1 if hypertension == 'Yes' else 0
    heart_disease = 1 if heart_disease == 'Yes' else 0
    smoking_history_map = {'Never': 1, 'Formerly': 2, 'Currently': 3, 'Other': 4}
    smoking_history = smoking_history_map.get(smoking_history, 0)

    # Normalize the continuous values based on the training data ranges
    age = (age - MIN_AGE) / (MAX_AGE - MIN_AGE)
    bmi = (bmi - MIN_BMI) / (MAX_BMI - MIN_BMI)
    hba1c = (hba1c - MIN_HBA1C) / (MAX_HBA1C - MIN_HBA1C)
    blood_glucose = (blood_glucose - MIN_GLUCOSE) / (MAX_GLUCOSE - MIN_GLUCOSE)

    # Create an array with the expected number of features (15)
    processed_input = [gender, age, hypertension, heart_disease, smoking_history, bmi, hba1c, blood_glucose]

    return processed_input

# Function to display prediction result
def display_prediction_result(prediction_proba):
    threshold = 0.5
    category = "Diabetic" if prediction_proba >= threshold else "Non-Diabetic"
    st.write(f"Prediction Probability of Diabetes: {prediction_proba:.2f}")
    st.write(f"Based on the prediction model, the individual is categorized as: {category}")

    # Optional: Adding color or emphasis based on the result
    if category == "Diabetic":
        st.markdown("<span style='color:red'>High Risk of Diabetes</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span style='color:green'>Low Risk of Diabetes</span>", unsafe_allow_html=True)

# Constants for normalization (replace with actual values from your training data)
MIN_AGE, MAX_AGE = 0, 100
MIN_BMI, MAX_BMI = 10, 50
MIN_HBA1C, MAX_HBA1C = 3.5, 9.0
MIN_GLUCOSE, MAX_GLUCOSE = 80, 300

# Main app
def main():
    # Custom styles
    st.markdown("""
        <style>
        /* Global font changes */
        html, body, [class*="css"] {
            font-family: 'Gill Sans', 'Gill Sans MT', Calibri, 'Trebuchet MS', sans-serif;
            color: #4f8bf9;  /* Change text color */
        }
        
        /* Background color for the entire app */
        body {
            background-color: #f0f2f6;
        }

        /* Style for titles and headers */
        .css-2trqyj, .css-hi6a2p {
            color: #ff6347;  /* Tomato color for titles */
        }

        /* Styling buttons */
        .stButton>button {
            color: white;
            background-color: #4f8bf9; /* Blue button background */
            border-radius: 10px; /* Rounded corners */
            border: 2px solid #4f8bf9; /* Blue border */
        }

        /* Customizing sliders */
        .stSlider .css-1cpxqw2 {
            background-color: #4f8bf9; /* Blue slider track */
        }

        /* Styling checkbox */
        .stCheckbox {
            color: #ff6347; /* Tomato color for checkbox text */
        }
        </style>
        """, unsafe_allow_html=True)
    st.title('Diabetes Prediction Application')

    # Load model and data
    model_path = 'diabetes_prediction_model.keras'
    data_path = 'diabetes_training_data.csv'
    model = load_model(model_path)
    data = load_data(data_path)

    if model is None or data is None:
        st.warning("Unable to load model or data.")
        return

    # Data Overview
    st.subheader('Dataset Overview')
    if st.checkbox('Show Data'):
        st.write(data.head())

    # Data Visualization
    st.subheader('Data Visualization')
    if st.checkbox('Show Correlation Heatmap'):
        plt.figure(figsize=(10, 6))
        sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
        st.pyplot(plt)

    # User Input
    st.subheader('Enter Your Information')
    gender = st.selectbox('Gender', ['Male', 'Female'])
    age = st.slider('Age', MIN_AGE, MAX_AGE, MIN_AGE)
    hypertension = st.selectbox('Hypertension', ['No', 'Yes'])
    heart_disease = st.selectbox('Heart Disease', ['No', 'Yes'])
    smoking_history = st.selectbox('Smoking History', ['Never', 'Formerly', 'Currently', 'Other'])
    bmi = st.slider('Body Mass Index (BMI)', MIN_BMI, MAX_BMI, MIN_BMI)
    hba1c = st.slider('HbA1c Level', MIN_HBA1C, MAX_HBA1C, MIN_HBA1C)
    blood_glucose = st.slider('Blood Glucose Level', MIN_GLUCOSE, MAX_GLUCOSE, MIN_GLUCOSE)

    # Prediction and result display
    if st.button('Predict'):
        try:
            processed_input = preprocess_input(gender, age, hypertension, heart_disease, smoking_history, bmi, hba1c, blood_glucose)
            processed_input = np.array([processed_input])  # Convert to NumPy array for the model

            prediction_proba = model.predict(processed_input)[0][0]

            display_prediction_result(prediction_proba)
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

    # Footer
    st.write('Developed by Mohamed Habib Agrebi')  # Replace with your name or organization

# Run main
if __name__ == '__main__':
    main()
