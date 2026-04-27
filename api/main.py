from fastapi import FastAPI
import numpy as np
import joblib
import requests

app = FastAPI()

# =========================
# API KEY (HARDCODED)
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
# BASE FEATURE POOL
# (we'll dynamically slice what each model needs)
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
# WEATHER FETCH
# =========================
def get_weather(city: str):
    url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={city}"
    res = requests.get(url, timeout=10)
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

    # Feature engineering
    sample["temp_dew_diff"] = temp - dew
    sample["humidity_pressure"] = humidity * pressure

    return sample

# =========================
# BUILD INPUT (SMART)
# =========================
def build_input_dynamic(sample, model, fallback_features):
    """
    Automatically matches feature size required by model
    """
    # Try to use exact feature names if available
    if hasattr(model, "feature_names_in_"):
        features = list(model.feature_names_in_)
    else:
        features = fallback_features

    values = [sample.get(col, 0) for col in features]

    # Adjust size if needed
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
    if score < 30:
        return "Low"
    elif score < 40:
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

@app.get("/predict/{city}")
def predict(city: str):
    try:
        weather = get_weather(city)

        # 🌧️ Rain
        X_rain = build_input_dynamic(weather, rain_model, base_features)
        rain_prob = float(rain_model.predict_proba(X_rain)[0][1])
        rain_pred = int(rain_prob > rain_threshold)

        # ⚡ Storm
        X_storm = build_input_dynamic(weather, storm_model, base_features)
        storm_prob = float(storm_model.predict_proba(X_storm)[0][1])

        # 🌡️ Heat
        X_heat = build_input_dynamic(weather, heat_model, base_features)
        heat_score = float(heat_model.predict(X_heat)[0])

        # 🌫️ Pollution
        X_pollution = build_input_dynamic(weather, pollution_model, base_features)
        pollution_score = float(pollution_model.predict(X_pollution)[0])

        return {
            "city": city,

            "current_weather": {
                "temperature_C": weather["temperature_C"],
                "humidity_pct": weather["humidity_pct"],
                "pressure_hPa": weather["pressure_hPa"],
                "cloud_cover_pct": weather["cloud_cover_pct"]
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
            }
        }

    except Exception as e:
        return {"error": str(e)}
