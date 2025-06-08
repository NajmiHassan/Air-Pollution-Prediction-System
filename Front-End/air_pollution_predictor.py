import pickle
import numpy as np
import os

class AirPollutionPredictor:
    def __init__(self, regression_model_path="regression_model.pkl", classification_model_path="classification_model.pkl"):
        """
        Initialize the Air Pollution Predictor with trained models
        
        Args:
            regression_model_path (str): Path to the regression model pickle file
            classification_model_path (str): Path to the classification model pickle file
        """
        self.regression_model = None
        self.classification_model = None
        self.quality_map = {0: "Good", 1: "Moderate", 2: "Unhealthy", 3: "Hazardous"}
        
        # Load models
        self.load_models(regression_model_path, classification_model_path)
    
    def load_models(self, regression_path, classification_path):
        """Load the trained models from pickle files"""
        try:
            # Load regression model
            if os.path.exists(regression_path):
                with open(regression_path, "rb") as f:
                    self.regression_model = pickle.load(f)
                print("Regression model loaded successfully!")
            else:
                raise FileNotFoundError(f"Regression model file not found: {regression_path}")
            
            # Load classification model
            if os.path.exists(classification_path):
                with open(classification_path, "rb") as f:
                    self.classification_model = pickle.load(f)
                print("Classification model loaded successfully!")
            else:
                raise FileNotFoundError(f"Classification model file not found: {classification_path}")
                
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            raise
    
    def prepare_features(self, temperature, humidity, pm10, no2, so2, co, industrial_proximity, population_density):
        """
        Prepare input features for prediction
        
        Args:
            temperature (float): Temperature in Celsius
            humidity (float): Humidity percentage
            pm10 (float): PM10 level
            no2 (float): NO2 level
            so2 (float): SO2 level
            co (float): CO level
            industrial_proximity (float): Proximity to industrial areas (0-10)
            population_density (int): Population density
            
        Returns:
            numpy.array: Feature array ready for prediction
        """
        return np.array([[temperature, humidity, pm10, no2, so2, co, industrial_proximity, population_density]])
    
    def predict_pm25(self, features):
        """
        Predict PM2.5 level using regression model
        
        Args:
            features (numpy.array): Input features
            
        Returns:
            float: Predicted PM2.5 level
        """
        if self.regression_model is None:
            raise ValueError("Regression model not loaded")
        
        try:
            prediction = self.regression_model.predict(features)
            return round(prediction[0], 2)
        except Exception as e:
            print(f"Error in PM2.5 prediction: {str(e)}")
            raise
    
    def predict_air_quality(self, features):
        """
        Predict air quality level using classification model
        
        Args:
            features (numpy.array): Input features
            
        Returns:
            tuple: (quality_index, quality_label)
        """
        if self.classification_model is None:
            raise ValueError("Classification model not loaded")
        
        try:
            prediction = self.classification_model.predict(features)
            quality_index = prediction[0]
            quality_label = self.quality_map.get(quality_index, "Unknown")
            return quality_index, quality_label
        except Exception as e:
            print(f"Error in air quality prediction: {str(e)}")
            raise
    
    def predict_both(self, temperature, humidity, pm10, no2, so2, co, industrial_proximity, population_density):
        """
        Predict both PM2.5 level and air quality
        
        Returns:
            dict: Dictionary containing both predictions
        """
        features = self.prepare_features(temperature, humidity, pm10, no2, so2, co, industrial_proximity, population_density)
        
        pm25_level = self.predict_pm25(features)
        quality_index, quality_label = self.predict_air_quality(features)
        
        return {
            'pm25_level': pm25_level,
            'air_quality_index': quality_index,
            'air_quality_label': quality_label,
            'features_used': {
                'temperature': temperature,
                'humidity': humidity,
                'pm10': pm10,
                'no2': no2,
                'so2': so2,
                'co': co,
                'industrial_proximity': industrial_proximity,
                'population_density': population_density
            }
        }
    
    def get_quality_description(self, quality_label):
        """Get description for air quality levels"""
        descriptions = {
            "Good": "Air quality is satisfactory, and air pollution poses little or no risk.",
            "Moderate": "Air quality is acceptable. However, sensitive individuals may experience minor issues.",
            "Unhealthy": "Members of sensitive groups may experience health effects. The general public is less likely to be affected.",
            "Hazardous": "Health alert: The risk of health effects is increased for everyone."
        }
        return descriptions.get(quality_label, "No description available.")
