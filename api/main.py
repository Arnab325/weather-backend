from fastapi import FastAPI
import numpy as np
import joblib
import requests
from tensorflow.keras.models import load_model

app = FastAPI()

# =========================
# API KEY
# =========================
API_KEY = "9e37120cfdb54781b8371238261404"

# =========================
# LOAD MODELS
# =========================
rain_model = load_model("training/rain_modelV1.keras")
storm_model = joblib.load("training/thunderstorm_xgb_model.pkl")

# =========================
# FEATURE LISTS
# =========================

# 🌧️ Rain model → 19 inputs
rain_features = [
    'lat','lon','temperature_C','humidity_pct','pressure_hPa',
    'dew_point_C','pressure_trend','solar_radiation_Wm2',
    'wind_speed_ms','cloud_cover_pct','hour','month',
    'wind_direction_deg','wind_dir_sin','wind_dir_cos',
    'et0_mm','precip_mm','city_encoded','extra_dummy'
]

# ⚡ Storm model → 18 inputs
storm_features = [
    'lat','lon','temperature_C','humidity_pct','pressure_hPa',
    'dew_point_C','pressure_trend','solar_radiation_Wm2',
    'wind_speed_ms','cloud_cover_pct','hour','month',
    'wind_direction_deg','wind_dir_sin','wind_dir_cos',
    'et0_mm','precip_mm','city_encoded'
]

# =========================
# WEATHER API FUNCTION
# =========================
def get_weather(city: str):
    url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={city}"
    response = requests.get(url)
    data = response.json()

    if "current" not in data:
        raise Exception(data.get("error", {}).get("message", "Weather API failed"))

    current = data["current"]
    location = data["location"]

    hour = int(location["localtime"].split(" ")[1].split(":")[0])
    month = int(location["localtime"].split("-")[1])
    wind_deg = current["wind_degree"]

    return {
        "lat": location["lat"],
        "lon": location["lon"],
        "temperature_C": current["temp_c"],
        "humidity_pct": current["humidity"],
        "pressure_hPa": current["pressure_mb"],
        "dew_point_C": current.get("dewpoint_c", 20),
        "solar_radiation_Wm2": current.get("uv", 5) * 100,
        "wind_speed_ms": current["wind_kph"] / 3.6,
        "cloud_cover_pct": current["cloud"],
        "hour": hour,
        "month": month,
        "wind_direction_deg": wind_deg,
        "wind_dir_sin": np.sin(np.radians(wind_deg)),
        "wind_dir_cos": np.cos(np.radians(wind_deg)),
        "et0_mm": 3
    }

# =========================
# BUILD INPUT (AUTO FIX)
# =========================
def build_input(sample, features):
    values = []
    for col in features:
        if col in sample:
            values.append(sample[col])
        else:
            # default values
            if col == "pressure_trend":
                values.append(0)
            elif col == "precip_mm":
                values.append(0)
            elif col == "city_encoded":
                values.append(1)
            else:
                values.append(0)
    return np.array([values])

# =========================
# HOME ROUTE
# =========================
@app.get("/")
def home():
    return {"message": "Weather API Running 🌧️⚡"}

# =========================
# PREDICTION ROUTE
# =========================
@app.get("/predict/{city}")
def predict(city: str):
    try:
        weather = get_weather(city)

        # 🌧️ Rain Prediction
        X_rain = build_input(weather, rain_features)
        rain_prob = float(rain_model.predict(X_rain)[0][0])

        # ⚡ Storm Prediction
        X_storm = build_input(weather, storm_features)
        storm_prob = float(storm_model.predict_proba(X_storm)[0][1])

        # smoothing
        rain_prob = 0.7 * rain_prob + 0.15
        storm_prob = 0.7 * storm_prob + 0.15

        return {
            "city": city,
            "rain_probability": round(rain_prob * 100, 2),
            "thunderstorm_probability": round(storm_prob * 100, 2)
        }

    except Exception as e:
        return {"error": str(e)}