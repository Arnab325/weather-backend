import requests
import joblib
import numpy as np
import os
from tensorflow.keras.models import load_model

# -------------------------------
# LOAD MODELS
# -------------------------------
rain_model = load_model("training/rain_modelV1.keras")
storm_model = joblib.load("training/thunderstorm_xgb_model.pkl")

# LOAD FEATURES (change if different files exist)
rain_features = joblib.load("training/features.pkl")
storm_features = joblib.load("training/features.pkl")

print("Rain model loaded")
print("Storm model loaded")

# -------------------------------
# API CONFIG
# -------------------------------
API_KEY = os.getenv("OPENWEATHER_API_KEY")  # safer
if API_KEY is None:
    API_KEY = "79ee1288346960b196a8c7d9ab8b3e3e"  # fallback (replace)

LAT = 22.7   # Kolkata (change if needed)
LON = 88.4

# -------------------------------
# FETCH WEATHER DATA
# -------------------------------
url = f"https://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric"

response = requests.get(url)

if response.status_code != 200:
    print("Error fetching data:", response.text)
    exit()

data = response.json()

print(data)
