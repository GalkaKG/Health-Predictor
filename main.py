import streamlit as st
import pandas as pd
import pickle
from PIL import Image

# Load the trained models with the pipelines
with open("models/outcome_model.pkl", "rb") as file:
    outcome_model = pickle.load(file)

with open("models/disease_model.pkl", "rb") as file:
    disease_model = pickle.load(file)

# Sidebar for navigation
st.sidebar.title("Navigation")
selected_page = st.sidebar.radio("Go to", ["Home", "Prediction"])

# Home Page
if selected_page == "Home":
    st.title("Disease Prediction App")
    st.markdown("""
    Welcome to the **Disease Prediction App**!  
    This app uses Machine Learning to predict whether you might have a disease based on your symptoms and other health indicators.
    """)
    # Add an image for visual enhancement
    image = Image.open("assets/healthcare.png")  # Ensure this path is correct
    st.image(image, use_column_width=True)
    st.info("Navigate to the **Prediction** page using the sidebar to start.")

# Prediction Page
elif selected_page == "Prediction":
    st.title("Prediction Form")
    st.markdown("### Please fill out the form below:")

    # Two-column layout for better arrangement
    col1, col2 = st.columns(2)

    with col1:
        fever = st.selectbox("Fever", ["Yes", "No"])
        cough = st.selectbox("Cough", ["Yes", "No"])
        fatigue = st.selectbox("Fatigue", ["Yes", "No"])
        difficulty_breathing = st.selectbox("Difficulty Breathing", ["Yes", "No"])

    with col2:
        age = st.slider("Age", min_value=0, max_value=100, value=25)
        gender = st.selectbox("Gender", ["Male", "Female"])
        blood_pressure = st.selectbox("Blood Pressure", ["Low", "Normal", "High"])
        cholesterol = st.selectbox("Cholesterol Level", ["Normal", "High"])

    # Divider for clarity
    st.markdown("---")
    if st.button("Predict Disease"):
        # Prepare input data for prediction
        input_data = pd.DataFrame({
            "Fever": [fever],
            "Cough": [cough],
            "Fatigue": [fatigue],
            "Difficulty Breathing": [difficulty_breathing],
            "Age": [age],
            "Gender": [gender],
            "Blood Pressure": [blood_pressure],
            "Cholesterol Level": [cholesterol]
        })

        # Get outcome prediction (Positive or Negative)
        outcome_prediction = outcome_model.predict(input_data)[0]

        # Visualization of the prediction
        st.markdown("### Prediction Result")
        if outcome_prediction == "Positive":
            st.success("The predicted outcome is: **Positive**")
            disease_prediction = disease_model.predict(input_data)[0]
            st.write(f"The predicted disease is: **{disease_prediction}**")
            st.warning("We recommend consulting with a healthcare professional.")
        else:
            st.error("The predicted outcome is: **Negative**")
            st.write("No disease predicted based on the provided inputs.")

# Footer
st.markdown("---")
st.caption("Disease Prediction App © 2024")

