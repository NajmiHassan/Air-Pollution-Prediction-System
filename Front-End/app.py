import streamlit as st
import pandas as pd
from air_pollution_predictor import AirPollutionPredictor

# Configure page
st.set_page_config(
    page_title="🌍 Air Pollution Prediction System",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize the predictor
@st.cache_resource
def load_predictor():
    """Load the air pollution predictor with caching"""
    try:
        return AirPollutionPredictor()
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.stop()

predictor = load_predictor()

# Main title and description
st.title("🌍 Air Pollution Prediction System")
st.markdown("""
This system predicts **PM2.5 levels** and **Air Quality** based on various environmental factors.
Enter the environmental parameters below to get predictions.
""")

# Create columns for better layout
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📊 Environmental Parameters")
    
    # Create sub-columns for input fields
    input_col1, input_col2 = st.columns(2)
    
    with input_col1:
        st.subheader("🌡️ Weather Conditions")
        temperature = st.number_input(
            "Temperature (°C)", 
            min_value=-10.0, 
            max_value=50.0, 
            value=25.0,
            help="Ambient temperature in Celsius"
        )
        
        humidity = st.number_input(
            "Humidity (%)", 
            min_value=0.0, 
            max_value=100.0, 
            value=50.0,
            help="Relative humidity percentage"
        )
        
        st.subheader("🏭 Pollutant Levels")
        pm10 = st.number_input(
            "PM10 Level (μg/m³)", 
            min_value=0.0, 
            max_value=500.0, 
            value=20.0,
            help="Particulate Matter 10 micrometers"
        )
        
        no2 = st.number_input(
            "NO2 Level (μg/m³)", 
            min_value=0.0, 
            max_value=500.0, 
            value=15.0,
            help="Nitrogen Dioxide level"
        )
    
    with input_col2:
        st.subheader("💨 Gas Concentrations")
        so2 = st.number_input(
            "SO2 Level (μg/m³)", 
            min_value=0.0, 
            max_value=500.0, 
            value=10.0,
            help="Sulfur Dioxide level"
        )
        
        co = st.number_input(
            "CO Level (mg/m³)", 
            min_value=0.0, 
            max_value=10.0, 
            value=1.0,
            help="Carbon Monoxide level"
        )
        
        st.subheader("🏘️ Location Factors")
        industrial_proximity = st.slider(
            "Proximity to Industrial Areas", 
            0.0, 
            10.0, 
            5.0,
            help="Scale: 0 (far) to 10 (very close)"
        )
        
        population_density = st.number_input(
            "Population Density (people/km²)", 
            min_value=0, 
            max_value=10000, 
            value=500,
            help="Number of people per square kilometer"
        )
    
    # Prediction buttons
    st.header("🔮 Predictions")
    
    prediction_col1, prediction_col2, prediction_col3 = st.columns(3)
    
    with prediction_col1:
        predict_pm25 = st.button("🔍 Predict PM2.5 Level", type="primary", use_container_width=True)
    
    with prediction_col2:
        predict_quality = st.button("🌬️ Predict Air Quality", type="primary", use_container_width=True)
    
    with prediction_col3:
        predict_both = st.button("📈 Predict Both", type="secondary", use_container_width=True)

with col2:
    st.header("📈 Results")
    
    # Handle predictions
    if predict_pm25 or predict_both:
        try:
            features = predictor.prepare_features(
                temperature, humidity, pm10, no2, so2, co, 
                industrial_proximity, population_density
            )
            pm25_result = predictor.predict_pm25(features)
            
            st.success("PM2.5 Prediction Complete!")
            st.metric(
                label="Predicted PM2.5 Level",
                value=f"{pm25_result:.2f} μg/m³",
                delta=None
            )
            
            # PM2.5 health information
            if pm25_result <= 12:
                st.success("🟢 Good air quality")
            elif pm25_result <= 35:
                st.warning("🟡 Moderate air quality")
            elif pm25_result <= 55:
                st.warning("🟠 Unhealthy for sensitive groups")
            else:
                st.error("🔴 Unhealthy air quality")
                
        except Exception as e:
            st.error(f"Error in PM2.5 prediction: {str(e)}")
    
    if predict_quality or predict_both:
        try:
            features = predictor.prepare_features(
                temperature, humidity, pm10, no2, so2, co, 
                industrial_proximity, population_density
            )
            quality_index, quality_label = predictor.predict_air_quality(features)
            
            st.success("Air Quality Prediction Complete!")
            st.metric(
                label="Predicted Air Quality",
                value=quality_label,
                delta=None
            )
            
            # Get and display description
            description = predictor.get_quality_description(quality_label)
            st.info(description)
            
            # Color coding for quality levels
            if quality_label == "Good":
                st.success("🟢 " + quality_label)
            elif quality_label == "Moderate":
                st.warning("🟡 " + quality_label)
            elif quality_label == "Unhealthy":
                st.warning("🟠 " + quality_label)
            else:
                st.error("🔴 " + quality_label)
                
        except Exception as e:
            st.error(f"Error in air quality prediction: {str(e)}")

# Additional information section
st.markdown("---")
st.header("ℹ️ About the System")

info_col1, info_col2 = st.columns(2)

with info_col1:
    st.subheader("📋 Input Parameters")
    st.markdown("""
    - **Temperature**: Ambient air temperature
    - **Humidity**: Relative humidity percentage
    - **PM10**: Particulate matter ≤ 10μm
    - **NO2**: Nitrogen dioxide concentration
    - **SO2**: Sulfur dioxide concentration
    - **CO**: Carbon monoxide concentration
    - **Industrial Proximity**: Distance to industrial areas
    - **Population Density**: People per square kilometer
    """)

with info_col2:
    st.subheader("🎯 Prediction Outputs")
    st.markdown("""
    - **PM2.5 Level**: Predicted particulate matter ≤ 2.5μm concentration
    - **Air Quality**: Classification into Good, Moderate, Unhealthy, or Hazardous
    
    **Air Quality Standards:**
    - 🟢 Good: Minimal health impact
    - 🟡 Moderate: Acceptable for most people
    - 🟠 Unhealthy: Risk for sensitive groups
    - 🔴 Hazardous: Health risk for everyone
    """)

# Display input summary
with st.expander("📊 Current Input Summary"):
    input_data = {
        'Parameter': ['Temperature', 'Humidity', 'PM10', 'NO2', 'SO2', 'CO', 'Industrial Proximity', 'Population Density'],
        'Value': [f"{temperature}°C", f"{humidity}%", f"{pm10} μg/m³", f"{no2} μg/m³", 
                 f"{so2} μg/m³", f"{co} mg/m³", f"{industrial_proximity}/10", f"{population_density} people/km²"],
        'Unit': ['Celsius', 'Percentage', 'μg/m³', 'μg/m³', 'μg/m³', 'mg/m³', 'Scale 0-10', 'people/km²']
    }
    
    df = pd.DataFrame(input_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
