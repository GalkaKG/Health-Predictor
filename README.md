# Health Predictor

### Deployed App

The application is also deployed and accessible online. You can try the model and make predictions directly by visiting the following link:
[Access the Deployed App](https://health-predictor-3uopoqgwgsyjbmporfpjew.streamlit.app/)

## Overview

The **Health Predictor** is a machine learning-based model designed to assist in predicting health outcomes and diagnosing potential diseases based on a patient's symptoms and profile. This tool leverages data-driven insights to identify correlations between various health indicators such as age, gender, symptoms, and medical history, enabling users to make informed decisions about potential health issues.

## Features

- **Disease Prediction**: The model predicts a wide range of diseases based on input symptoms, age, gender, blood pressure, and cholesterol levels.
- **Comprehensive Dataset**: Built using a dataset that includes various diseases, their symptoms, and patient profile features such as age, gender, and lifestyle factors.
- **User-Friendly Interface**: Input patient details and symptoms, and receive a predicted diagnosis.
- **Model Training**: The model is trained using various machine learning algorithms to provide accurate disease predictions.

## Dataset

The model is built using a dataset containing various diseases and associated symptoms along with patient demographic and health data. You can download the dataset here:

[Download Dataset](https://www.kaggle.com/datasets/uom190346a/disease-symptoms-and-patient-profile-dataset/data)

The dataset includes the following columns:

- **Disease**: The target variable, representing the disease or condition diagnosed.
- **Symptoms**: A set of symptoms (such as Fever, Cough, Fatigue, Difficulty Breathing) associated with each disease.
- **Patient Profile**: Demographic data such as Age, Gender, Blood Pressure, and Cholesterol Level.
- **Outcome Variable**: The final diagnosis indicating if a disease is predicted to be positive or negative based on the symptoms and patient profile.

## Installation

### Prerequisites

Before running the model, make sure you have the following libraries installed:

- **Python** (>= 3.7)
- **pandas** (for data manipulation)
- **scikit-learn** (for machine learning model implementation)
- **numpy** (for numerical operations)
- **matplotlib** (for visualizations)

You can install the required libraries using `pip`:

```bash
pip install pandas scikit-learn numpy matplotlib
```

### Install Dependencies
To install the necessary libraries, simply run:

```bash
pip install -r requirements.txt
```

This will install all the dependencies required for the app to run, including Streamlit and scikit-learn.

### Running the App Locally
Once the dependencies are installed, you can run the app locally by executing the following command:

```bash
streamlit run main.py
```

This will start the Streamlit app, and you can view it in your browser.
