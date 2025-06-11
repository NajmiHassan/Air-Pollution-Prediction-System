#  Air Pollution Prediction System

A machine learning-based web application that predicts PM2.5 levels and air quality using environmental and demographic factors.

## Overview

This system uses trained regression and classification models to predict:
- **PM2.5 Concentration** (µg/m³) - Fine particulate matter levels
- **Air Quality Level** - Classification into Good, Moderate, Poor, or Hazardous

##  Features

- Real-time air quality predictions
- User-friendly web interface built with Streamlit
- Dual prediction capabilities (PM2.5 levels and quality classification)
- Interactive input controls with validation
- Color-coded results with health recommendations

## 📊 Dataset Information

**Source**: Air Quality and Pollution Assessment Dataset
- **Size**: 5,000 samples
- **Features**: 8 environmental and demographic variables
- **Target**: Air quality levels (Good, Moderate, Poor, Hazardous)

### Input Parameters:
- **Temperature** (°C): Average regional temperature
- **Humidity** (%): Relative humidity
- **PM10** (µg/m³): Coarse particulate matter
- **NO2** (ppb): Nitrogen dioxide levels
- **SO2** (ppb): Sulfur dioxide levels
- **CO** (ppm): Carbon monoxide levels
- **Industrial Proximity** (km): Distance to nearest industrial zone
- **Population Density** (people/km²): Regional population density

## 🚀 Quick Start

### Prerequisites
```bash
pip install streamlit numpy pandas pickle
```

### File Structure
```
air-pollution-predictor/
├── app.py                      # Streamlit web interface
├── air_pollution_predictor.py  # Main prediction logic
├── regression_model.pkl        # Trained PM2.5 prediction model
├── classification_model.pkl    # Trained air quality classification model
└── README.md
```

### Running the Application
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## 🎮 How to Use

1. **Enter Environmental Data**: Input temperature, humidity, and pollutant levels
2. **Set Location Factors**: Specify industrial proximity and population density
3. **Get Predictions**: Choose to predict PM2.5 levels, air quality, or both
4. **View Results**: See color-coded predictions with health recommendations

## 🏥 Air Quality Standards

- 🟢 **Good**: Safe air quality with minimal health impact
- 🟡 **Moderate**: Acceptable for most people
- 🟠 **Poor**: May affect sensitive individuals
- 🔴 **Hazardous**: Health risk for everyone

## 🔧 Technical Details

- **Framework**: Streamlit for web interface
- **Models**: Scikit-learn based regression and classification
- **Input Validation**: Built-in range checking and error handling
- **Caching**: Optimized model loading for better performance

## 📁 Project Structure

- `air_pollution_predictor.py`: Core prediction logic and model handling
- `app.py`: Streamlit web application interface
- Model files: Pre-trained pickle files for predictions

## Made by Najmi Hassan
