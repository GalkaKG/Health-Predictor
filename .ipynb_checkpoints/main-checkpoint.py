# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import pickle

# # Load the trained model with the pipeline
# with open("model_with_pipeline.pkl", "rb") as file:
#     model = pickle.load(file)

# # Title for the app
# st.title("Disease Prediction App")

# # Input fields for each feature
# disease = st.selectbox("Disease Type", ["Influenza", "Common Cold", "Eczema", "Asthma", "Stroke"])
# fever = st.selectbox("Fever", ["Yes", "No"])
# cough = st.selectbox("Cough", ["Yes", "No"])
# fatigue = st.selectbox("Fatigue", ["Yes", "No"])
# difficulty_breathing = st.selectbox("Difficulty Breathing", ["Yes", "No"])
# age = st.slider("Age", min_value=0, max_value=100, value=25)
# gender = st.selectbox("Gender", ["Male", "Female"])
# blood_pressure = st.selectbox("Blood Pressure", ["Low", "Normal", "High"])
# cholesterol = st.selectbox("Cholesterol Level", ["Normal", "High"])

# # Define button for prediction
# if st.button("Predict Disease Outcome"):
#     # Prepare input data for the model (convert to dataframe)
#     input_data = pd.DataFrame({
#         "Disease": [disease],
#         "Fever": [fever],
#         "Cough": [cough],
#         "Fatigue": [fatigue],
#         "Difficulty Breathing": [difficulty_breathing],
#         "Age": [age],
#         "Gender": [gender],
#         "Blood Pressure": [blood_pressure],
#         "Cholesterol Level": [cholesterol]
#     })

#     # Display input data for debugging
#     st.write("Input Data for Prediction:")
#     st.write(input_data)

#     # Get prediction probabilities using the model's pipeline (this automatically handles encoding)
#     proba = model.predict_proba(input_data)[0]
    
#     # Show the predicted probabilities for each class
#     st.write(f"Probability of Negative: {proba[0]}")
#     st.write(f"Probability of Positive: {proba[1]}")

#     # Make prediction based on the higher probability
#     prediction = model.predict(input_data)[0]
    
#     # Show result
#     st.write("The predicted outcome is:", "Positive" if prediction == 1 else "Negative")

    
    

#     # Example: Plotting the prediction probabilities as a bar chart
#     labels = ['Negative', 'Positive']
#     probabilities = [proba[0], proba[1]]

#     plt.bar(labels, probabilities)
#     plt.title("Prediction Probabilities")
#     plt.xlabel("Class")
#     plt.ylabel("Probability")

#     # Display the plot in Streamlit
#     st.pyplot(plt)
    
#     import seaborn as sns

#     # Example: Seaborn bar plot for probabilities
#     sns.barplot(x=labels, y=probabilities)
#     plt.title("Prediction Probabilities")
#     plt.xlabel("Class")
#     plt.ylabel("Probability")

#     # Display the plot in Streamlit
#     st.pyplot(plt)
    
    
#     import plotly.graph_objects as go

#     # Example: Plotting the prediction probabilities as an interactive bar chart
#     fig = go.Figure(data=[go.Bar(x=labels, y=probabilities)])

#     # Update the layout of the plot
#     fig.update_layout(title="Prediction Probabilities",
#                       xaxis_title="Class",
#                       yaxis_title="Probability")

#     # Display the interactive plot in Streamlit
#     st.plotly_chart(fig)
    

    
import streamlit as st
import pandas as pd
import pickle

# Load the trained models with the pipelines
with open("models/outcome_model.pkl", "rb") as file:
    outcome_model = pickle.load(file)

with open("models/disease_model.pkl", "rb") as file:
    disease_model = pickle.load(file)

# Title for the app
st.title("Disease Prediction App")

# Input fields for each feature
fever = st.selectbox("Fever", ["Yes", "No"])
cough = st.selectbox("Cough", ["Yes", "No"])
fatigue = st.selectbox("Fatigue", ["Yes", "No"])
difficulty_breathing = st.selectbox("Difficulty Breathing", ["Yes", "No"])
age = st.slider("Age", min_value=0, max_value=100, value=25)
gender = st.selectbox("Gender", ["Male", "Female"])
blood_pressure = st.selectbox("Blood Pressure", ["Low", "Normal", "High"])
cholesterol = st.selectbox("Cholesterol Level", ["Normal", "High"])

# Define button for prediction
if st.button("Predict Disease"):
    # Prepare input data for outcome prediction
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
    
    # Show the outcome prediction
    st.write(f"The predicted outcome is: {outcome_prediction}")

    # If the outcome is Positive, predict the Disease
    if outcome_prediction == "Positive":
        # Get disease prediction based on the same input data
        disease_prediction = disease_model.predict(input_data)[0]
        st.write(f"The predicted disease is: {disease_prediction}")
    else:
        st.write("No disease predicted due to negative outcome")
