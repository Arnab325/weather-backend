from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import joblib
import requests
import os

app = FastAPI()

# =========================
# CORS
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# API KEY
# =========================
API_KEY = "9e37120cfdb54781b8371238261404"

# =========================
# LOAD MODELS
# =========================
rain_model = joblib.load("training/rain_model_final.pkl")
storm_model = joblib.load("training/thunderstorm_xgb_model.pkl")
heat_model = joblib.load("training/heat_model.pkl")
pollution_model = joblib.load("training/pollution_model.pkl")

rain_threshold = 0.72

# =========================
# FEATURE POOL
# =========================
base_features = [
    'lat','lon','temperature_C','humidity_pct','pressure_hPa',
    'dew_point_C','pressure_trend','solar_radiation_Wm2',
    'wind_speed_ms','cloud_cover_pct','hour','month',
    'wind_direction_deg','wind_dir_sin','wind_dir_cos',
    'et0_mm','precip_mm','city_encoded',
    'temp_dew_diff','humidity_pressure'
]

# =========================
# WIND DIRECTION TEXT
# =========================
def get_wind_direction(deg):
    directions = ["N","NE","E","SE","S","SW","W","NW"]
    return directions[round(deg / 45) % 8]

# =========================
# WEATHER FETCH (UPDATED)
# =========================
def get_weather(query: str):
    # query can be "Kolkata" OR "22.57,88.36"
    url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={query}&aqi=yes"
    res = requests.get(url, timeout=10)
    data = res.json()

    if "current" not in data:
        raise Exception(data.get("error", {}).get("message", "Weather API failed"))

    current = data["current"]
    location = data["location"]
    air = current.get("air_quality", {})

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
        "wind_speed_kph": current["wind_kph"],
        "wind_direction_deg": wind_deg,

        "cloud_cover_pct": current["cloud"],
        "visibility_km": current.get("vis_km", 0),
        "uv_index": current.get("uv", 0),
        "precip_mm": current.get("precip_mm", 0),
        "feels_like_C": current.get("feelslike_c", temp),

        "wind_dir_sin": np.sin(np.radians(wind_deg)),
        "wind_dir_cos": np.cos(np.radians(wind_deg)),

        "et0_mm": 3,
        "pressure_trend": 0,
        "hour": hour,
        "month": month,
        "city_encoded": 1,

        # 🌫️ GAS DATA
        "co": air.get("co", 0),
        "no2": air.get("no2", 0),
        "o3": air.get("o3", 0),
        "so2": air.get("so2", 0),
        "pm2_5": air.get("pm2_5", 0),
        "pm10": air.get("pm10", 0)
    }

    sample["temp_dew_diff"] = temp - dew
    sample["humidity_pressure"] = humidity * pressure

    return sample, location

# =========================
# BUILD INPUT
# =========================
def build_input_dynamic(sample, model):
    if hasattr(model, "feature_names_in_"):
        features = list(model.feature_names_in_)
    else:
        features = base_features

    values = [sample.get(col, 0) for col in features]

    expected = getattr(model, "n_features_in_", len(values))

    if len(values) > expected:
        values = values[:expected]
    elif len(values) < expected:
        values += [0] * (expected - len(values))

    return np.array([values])

# =========================
# INTERPRETATION
# =========================
def categorize_heat(score):
    if score == 0:
        return "Low"
    elif score == 1:
        return "Moderate"
    else:
        return "High"

def categorize_pollution(score):
    if score < 50:
        return "Good"
    elif score < 100:
        return "Moderate"
    elif score < 150:
        return "Unhealthy for Sensitive Groups"
    else:
        return "Unhealthy"

# =========================
# ROUTES
# =========================
@app.get("/")
def home():
    return {"message": "Climate Intelligence API Running 🌍"}

@app.get("/predict/{query}")
def predict(query: str):
    try:
        weather, location = get_weather(query)

        # 🌧️ Rain
        X_rain = build_input_dynamic(weather, rain_model)
        rain_prob = float(rain_model.predict_proba(X_rain)[0][1])
        rain_pred = int(rain_prob > rain_threshold)

        # ⚡ Storm
        X_storm = build_input_dynamic(weather, storm_model)
        storm_prob = float(storm_model.predict_proba(X_storm)[0][1])

        # 🌡️ Heat
        X_heat = build_input_dynamic(weather, heat_model)
        heat_score = float(heat_model.predict(X_heat)[0])

        # 🌫️ Pollution
        X_pollution = build_input_dynamic(weather, pollution_model)
        pollution_score = float(pollution_model.predict(X_pollution)[0])

        return {
            "query": query,

            # 🌍 RESOLVED LOCATION
            "resolved_location": {
                "name": location["name"],
                "region": location["region"],
                "country": location["country"]
            },

            "location": {
                "lat": weather["lat"],
                "lon": weather["lon"]
            },

            "current_weather": {
                "temperature_C": weather["temperature_C"],
                "feels_like_C": weather["feels_like_C"],
                "humidity_pct": weather["humidity_pct"],
                "pressure_hPa": weather["pressure_hPa"],
                "cloud_cover_pct": weather["cloud_cover_pct"],

                "wind": {
                    "speed_ms": round(weather["wind_speed_ms"], 2),
                    "speed_kph": round(weather["wind_speed_kph"], 2),
                    "direction_deg": weather["wind_direction_deg"],
                    "direction": get_wind_direction(weather["wind_direction_deg"])
                },

                "visibility_km": weather["visibility_km"],
                "uv_index": weather["uv_index"],
                "precip_mm": weather["precip_mm"],
                "solar_radiation_Wm2": weather["solar_radiation_Wm2"],
                "dew_point_C": weather["dew_point_C"]
            },

            "rain": {
                "probability_%": round(rain_prob * 100, 2),
                "prediction": rain_pred
            },

            "thunderstorm": {
                "probability_%": round(storm_prob * 100, 2)
            },

            "heat_risk": {
                "score": round(heat_score, 2),
                "level": categorize_heat(heat_score)
            },

            "air_pollution": {
                "score": round(pollution_score, 2),
                "category": categorize_pollution(pollution_score)
            },

            "air_gases": {
                "co": round(weather["co"], 2),
                "no2": round(weather["no2"], 2),
                "o3": round(weather["o3"], 2),
                "so2": round(weather["so2"], 2),
                "pm2_5": round(weather["pm2_5"], 2),
                "pm10": round(weather["pm10"], 2)
            }
        }

    except Exception as e:
        return {"error": str(e)}
