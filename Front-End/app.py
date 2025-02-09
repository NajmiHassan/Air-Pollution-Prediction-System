import joblib
 
# Load trained models
regression_model = joblib.load("regression_model.pkl")
classification_model = joblib.load("classification_model.pkl")

import streamlit as st
import numpy as np
import joblib

# Load trained models
regression_model = joblib.load("regression_model.pkl")
classification_model = joblib.load("classification_model.pkl")

# Streamlit UI
st.title("🌍 Air Pollution Prediction System")

st.sidebar.header("Enter Environmental Factors:")
temperature = st.sidebar.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0, value=25.0)
humidity = st.sidebar.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0)
pm10 = st.sidebar.number_input("PM10 Level", min_value=0.0, max_value=500.0, value=20.0)
no2 = st.sidebar.number_input("NO2 Level", min_value=0.0, max_value=500.0, value=15.0)
so2 = st.sidebar.number_input("SO2 Level", min_value=0.0, max_value=500.0, value=10.0)
co = st.sidebar.number_input("CO Level", min_value=0.0, max_value=10.0, value=1.0)
industrial_proximity = st.sidebar.slider("Proximity to Industrial Areas", 0.0, 10.0, 5.0)
population_density = st.sidebar.number_input("Population Density", min_value=0, max_value=10000, value=500)

# Collect input into an array
features = np.array([[temperature, humidity, pm10, no2, so2, co, industrial_proximity, population_density]])

# Predict PM2.5 Level (Regression)
if st.sidebar.button("Predict PM2.5 Level"):
    pm25_pred = regression_model.predict(features)
    st.write(f"**Predicted PM2.5 Level:** {pm25_pred[0]:.2f}")

# Predict Air Quality (Classification)
if st.sidebar.button("Predict Air Quality Level"):
    air_quality_pred = classification_model.predict(features)
    quality_map = {0: "Good", 1: "Moderate", 2: "Unhealthy", 3: "Hazardous"}
    st.write(f"**Predicted Air Quality Level:** {quality_map[air_quality_pred[0]]}")
