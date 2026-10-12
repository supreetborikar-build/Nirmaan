import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the model and scaler
@st.cache_resource
def load_artifacts():
    model = joblib.load('improved_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_artifacts()
    st.title("University Cutoff Predictor App")
    st.write("Enter details to predict the capped cutoff figure.")

    # Inputs
    year = st.number_input("Year", min_value=2024, max_value=2030, value=2025)
    university_encoded = st.number_input("University Encoded Value", value=1000.0)
    category_encoded = st.number_input("Category Encoded Value", value=1000.0)
    seat_type_encoded = st.number_input("Seat Type Encoded Value", value=1000.0)

    if st.button("Predict"):
        features = np.array([[year, university_encoded, category_encoded, seat_type_encoded]])
        scaled_features = scaler.transform(features)
        prediction = model.predict(scaled_features)
        st.success(f"Predicted Cutoff Figure: {prediction[0]:.2f}")
except Exception as e:
    st.error(f"Error loading model components: {e}")