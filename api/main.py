from fastapi import FastAPI
import numpy as np
import joblib
import requests
import os

app = FastAPI()

# =========================
# ENV
# =========================
API_KEY = os.getenv("API_KEY")

# =========================
# LOAD MODELS
# =========================
rain_model = joblib.load("training/rain_model_final.pkl")
storm_model = joblib.load("training/thunderstorm_xgb_model.pkl")

rain_threshold = 0.72

# =========================
# FEATURE LISTS
# =========================
rain_features = [
    'lat','lon','temperature_C','humidity_pct','pressure_hPa',
    'dew_point_C','pressure_trend','solar_radiation_Wm2',
    'wind_speed_ms','cloud_cover_pct',
    'wind_dir_sin','wind_dir_cos',
    'et0_mm','temp_dew_diff','humidity_pressure'
]

storm_features = [
    'lat','lon','temperature_C','humidity_pct','pressure_hPa',
    'dew_point_C','pressure_trend','solar_radiation_Wm2',
    'wind_speed_ms','cloud_cover_pct','hour','month',
    'wind_direction_deg','wind_dir_sin','wind_dir_cos',
    'et0_mm','precip_mm','city_encoded'
]

# =========================
# WEATHER FETCH
# =========================
def get_weather(city: str):
    url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={city}"
    res = requests.get(url)
    data = res.json()

    if "current" not in data:
        raise Exception(data.get("error", {}).get("message", "Weather API failed"))

    current = data["current"]
    location = data["location"]

    wind_deg = current["wind_degree"]
    hour = int(location["localtime"].split(" ")[1].split(":")[0])
    month = int(location["localtime"].split("-")[1])

    temp = current["temp_c"]
    humidity = current["humidity"]
    pressure = current["pressure_mb"]
    dew = current.get("dewpoint_c", temp)

    sample = {
        "lat": location["lat"],
        "lon": location["lon"],
        "temperature_C": temp,
        "humidity_pct": humidity,
        "pressure_hPa": pressure,
        "dew_point_C": dew,
        "solar_radiation_Wm2": current.get("uv", 5) * 100,
        "wind_speed_ms": current["wind_kph"] / 3.6,
        "cloud_cover_pct": current["cloud"],
        "wind_direction_deg": wind_deg,
        "wind_dir_sin": np.sin(np.radians(wind_deg)),
        "wind_dir_cos": np.cos(np.radians(wind_deg)),
        "et0_mm": 3,
        "pressure_trend": 0,
        "hour": hour,
        "month": month,
        "precip_mm": 0,
        "city_encoded": 1
    }

    # feature engineering
    sample["temp_dew_diff"] = temp - dew
    sample["humidity_pressure"] = humidity * pressure

    return sample

# =========================
# BUILD INPUT
# =========================
def build_input(sample, features):
    return np.array([[sample.get(col, 0) for col in features]])

# =========================
# HOME
# =========================
@app.get("/")
def home():
    return {"message": "Weather API Running 🌧️⚡"}

# =========================
# CURRENT WEATHER
# =========================
@app.get("/weather/{city}")
def weather(city: str):
    try:
        w = get_weather(city)
        return {
            "city": city,
            "temperature_C": w["temperature_C"],
            "humidity_pct": w["humidity_pct"],
            "pressure_hPa": w["pressure_hPa"],
            "cloud_cover_pct": w["cloud_cover_pct"],
            "wind_speed_ms": w["wind_speed_ms"],
            "lat": w["lat"],
            "lon": w["lon"]
        }
    except Exception as e:
        return {"error": str(e)}

# =========================
# PREDICTION
# =========================
@app.get("/predict/{city}")
def predict(city: str):
    try:
        weather = get_weather(city)

        # 🌧️ Rain
        X_rain = build_input(weather, rain_features)
        rain_prob = float(rain_model.predict_proba(X_rain)[0][1])
        rain_pred = int(rain_prob > rain_threshold)

        # ⚡ Storm
        X_storm = build_input(weather, storm_features)
        storm_prob = float(storm_model.predict_proba(X_storm)[0][1])

        return {
            "city": city,
            "current_weather": {
                "temperature_C": weather["temperature_C"],
                "humidity_pct": weather["humidity_pct"],
                "pressure_hPa": weather["pressure_hPa"],
                "cloud_cover_pct": weather["cloud_cover_pct"]
            },
            "rain_probability": round(rain_prob * 100, 2),
            "rain_prediction": rain_pred,
            "thunderstorm_probability": round(storm_prob * 100, 2)
        }

    except Exception as e:
        return {"error": str(e)}
