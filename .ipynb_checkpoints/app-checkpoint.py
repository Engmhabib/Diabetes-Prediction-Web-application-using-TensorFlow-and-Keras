
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

# Load the TensorFlow model
@st.cache(allow_output_mutation=True)
def load_model():
    try:
        return tf.keras.models.load_model('diabetes_prediction_model.h5')  # Updated model path
    except FileNotFoundError:
        st.error("Model file not found. Please check the file path.")
        return None

# Load the data (ensure pandas is installed)
@st.cache(allow_output_mutation=True)
def load_data():
    try:
        return pd.read_csv('diabetes_prediction_dataset.csv')  # Updated data path
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

    # Normalize the continuous values (assuming the same range as training data)
    # Replace these with actual min-max values used in your training dataset
    age = (age - 0) / (100 - 0)
    bmi = (bmi - 10) / (50 - 10)
    hba1c = (hba1c - 3.5) / (9.0 - 3.5)
    blood_glucose = (blood_glucose - 80) / (300 - 80)

    return [gender, age, hypertension, heart_disease, smoking_history, bmi, hba1c, blood_glucose]

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
    
    model = load_model()
    data = load_data()

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
    age = st.slider('Age', 0, 100, 25)
    hypertension = st.selectbox('Hypertension', ['No', 'Yes'])
    heart_disease = st.selectbox('Heart Disease', ['No', 'Yes'])
    smoking_history = st.selectbox('Smoking History', ['Never', 'Formerly', 'Currently', 'Other'])
    bmi = st.slider('Body Mass Index (BMI)', 10, 50, 25)
    hba1c = st.slider('HbA1c Level', 3.5, 9.0, 5.5)
    blood_glucose = st.slider('Blood Glucose Level', 80, 300, 120)


    # Prediction and result display
    if st.button('Predict'):
        processed_input = preprocess_input(user_input)  # Ensure user_input is processed correctly
        prediction_proba = model.predict_proba(processed_input)[0][1]  # Assuming binary classification
        display_prediction_result(prediction_proba)
         try:
            prediction = model.predict([input_data])
            prediction_proba = model.predict_proba([input_data])

        if prediction[0] == 1:
                st.success(f'Diabetic with a probability of {prediction_proba[0][1]:.2f}')
            else:
                st.success(f'Not Diabetic with a probability of {prediction_proba[0][0]:.2f}')
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

    # Footer
    
    st.write('Developed by Mohamed Habib Agrebi')  # Replace with your name or organization

# Run main
if __name__ == '__main__':
    main()
