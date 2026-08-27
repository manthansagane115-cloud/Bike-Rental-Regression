import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import date
import google.generativeai as genai

st.set_page_config(
    page_title="Bike Rental Demand Prediction",
    page_icon="🚲",
    layout="centered"
)

MODEL_FILE = "model.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


# Feature columns used when the Random Forest was trained.
FEATURE_COLUMNS = [
    "yr", "mnth", "hr", "weekday", "temp", "atemp", "hum", "windspeed",
    "day", "weekofyear", "is_weekend", "rush_hour",
    "hr_sin", "hr_cos", "month_sin", "month_cos",
    "weekday_sin", "weekday_cos", "temp_difference", "comfort_index",
    "season_fall", "season_springer", "season_summer", "season_winter",
    "holiday_No", "holiday_Yes",
    "workingday_No work", "workingday_Working Day",
    "weathersit_Clear", "weathersit_Heavy Rain",
    "weathersit_Light Snow", "weathersit_Mist",
    "time_of_day_Afternoon", "time_of_day_Evening",
    "time_of_day_Morning", "time_of_day_Night"
]


def time_of_day(hour):
    if hour < 6:
        return "Night"
    elif hour < 12:
        return "Morning"
    elif hour < 17:
        return "Afternoon"
    elif hour < 21:
        return "Evening"
    else:
        return "Night"


def normalize_season(value):
    mapping = {
        "spring": "springer",
        "springer": "springer",
        "summer": "summer",
        "fall": "fall",
        "autumn": "fall",
        "winter": "winter"
    }
    return mapping[str(value).lower()]


def normalize_weather(value):
    mapping = {
        "clear": "Clear",
        "mist": "Mist",
        "light snow": "Light Snow",
        "heavy rain": "Heavy Rain"
    }
    return mapping[str(value).lower()]


def build_features(year, month, hour, weekday, season, holiday,
                    workingday, weather, temp, atemp, hum, windspeed,
                    selected_date):
    row = {}

    # Numerical features exactly as used during training.
    row["yr"] = float(year)
    row["mnth"] = int(month)
    row["hr"] = int(hour)
    row["weekday"] = int(weekday)
    row["temp"] = float(temp)
    row["atemp"] = float(atemp)
    row["hum"] = float(hum)
    row["windspeed"] = float(windspeed)

    # Date-based features.
    dt = pd.Timestamp(selected_date)
    row["day"] = int(dt.day)
    row["weekofyear"] = int(dt.isocalendar().week)

    # Engineered features.
    row["is_weekend"] = int(weekday in [0, 6])
    row["rush_hour"] = int(hour in [7, 8, 9, 17, 18, 19])
    row["hr_sin"] = np.sin(2 * np.pi * hour / 24)
    row["hr_cos"] = np.cos(2 * np.pi * hour / 24)
    row["month_sin"] = np.sin(2 * np.pi * month / 12)
    row["month_cos"] = np.cos(2 * np.pi * month / 12)
    row["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    row["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
    row["temp_difference"] = atemp - temp
    row["comfort_index"] = temp * (1 - hum)

    # One-hot encoded categorical features.
    season = normalize_season(season)
    weather = normalize_weather(weather)
    holiday = str(holiday)
    workingday = str(workingday)
    tod = time_of_day(hour)

    row["season_fall"] = int(season == "fall")
    row["season_springer"] = int(season == "springer")
    row["season_summer"] = int(season == "summer")
    row["season_winter"] = int(season == "winter")

    row["holiday_No"] = int(holiday == "No")
    row["holiday_Yes"] = int(holiday == "Yes")

    row["workingday_No work"] = int(workingday == "No work")
    row["workingday_Working Day"] = int(workingday == "Working Day")

    row["weathersit_Clear"] = int(weather == "Clear")
    row["weathersit_Heavy Rain"] = int(weather == "Heavy Rain")
    row["weathersit_Light Snow"] = int(weather == "Light Snow")
    row["weathersit_Mist"] = int(weather == "Mist")

    row["time_of_day_Afternoon"] = int(tod == "Afternoon")
    row["time_of_day_Evening"] = int(tod == "Evening")
    row["time_of_day_Morning"] = int(tod == "Morning")
    row["time_of_day_Night"] = int(tod == "Night")

    # Guarantee exact training-column order.
    return pd.DataFrame([row]).reindex(columns=FEATURE_COLUMNS, fill_value=0)


# ---------------------------------------------------------------------
# Gemini setup
# ---------------------------------------------------------------------
GEMINI_ENABLED = False
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    GEMINI_ENABLED = True
except Exception:
    # No key configured yet (e.g. running locally without secrets.toml).
    GEMINI_ENABLED = False


def get_gemini_suggestion(prediction, inputs: dict) -> str:
    """Ask Gemini for a short, practical explanation/recommendation
    based on the model's predicted bike demand and the input conditions."""
    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = f"""
You are a bike-sharing operations analyst.

Predicted bike rental demand: {prediction} bikes.
Conditions: {inputs}

In 3-4 short sentences:
1. Explain the likely reason for this demand level given the conditions.
2. Give one practical, actionable recommendation for the bike-sharing
   operator (e.g. rebalancing bikes between stations, staffing levels,
   maintenance timing, promotions during low demand).

Keep the tone concise and operational, not generic.
"""
    response = model.generate_content(prompt)
    return response.text


# Load model.
try:
    model = load_model()
except FileNotFoundError:
    st.error("model.pkl was not found. Put model.pkl in the same GitHub repository as app.py.")
    st.stop()
except Exception as e:
    st.error(f"Could not load model.pkl: {e}")
    st.stop()

st.title("🚲 Bike Rental Demand Prediction")
st.write(
    "Predict the expected number of bike rentals using time, weather, "
    "season and working-day information."
)

st.header("Prediction Inputs")

col1, col2 = st.columns(2)

with col1:
    selected_date = st.date_input(
        "Date",
        value=date(2012, 6, 15),
        min_value=date(2011, 1, 1),
        max_value=date(2012, 12, 31)
    )
    year = selected_date.year

    month = st.slider("Month", 1, 12, selected_date.month)
    hour = st.slider("Hour", 0, 23, 17)
    weekday = st.slider(
        "Weekday (0=Sunday, 6=Saturday)",
        0, 6, 2
    )
    season = st.selectbox(
        "Season",
        ["springer", "summer", "fall", "winter"]
    )
    holiday = st.selectbox(
        "Holiday",
        ["No", "Yes"]
    )

with col2:
    workingday = st.selectbox(
        "Working Day",
        ["Working Day", "No work"]
    )
    weather = st.selectbox(
        "Weather Situation",
        ["Clear", "Mist", "Light Snow", "Heavy Rain"]
    )
    temp = st.number_input(
        "Temperature (normalized)",
        min_value=0.0,
        max_value=1.0,
        value=0.60,
        step=0.01
    )
    atemp = st.number_input(
        "Feeling Temperature (normalized)",
        min_value=0.0,
        max_value=1.0,
        value=0.60,
        step=0.01
    )
    hum = st.number_input(
        "Humidity (normalized)",
        min_value=0.0,
        max_value=1.0,
        value=0.50,
        step=0.01
    )
    windspeed = st.number_input(
        "Wind Speed (normalized)",
        min_value=0.0,
        max_value=1.0,
        value=0.20,
        step=0.01
    )

is_weekend = int(weekday in [0, 6])
rush_hour = int(hour in [7, 8, 9, 17, 18, 19])
tod = time_of_day(hour)

st.subheader("Engineered Features")
st.write(
    f"**Time of day:** {tod} | "
    f"**Weekend:** {is_weekend} | "
    f"**Rush hour:** {rush_hour}"
)

if not GEMINI_ENABLED:
    st.caption("ℹ️ Gemini suggestions are disabled — add GEMINI_API_KEY to secrets to enable them.")

if st.button("Predict Bike Demand", type="primary"):
    try:
        input_df = build_features(
            year=year,
            month=month,
            hour=hour,
            weekday=weekday,
            season=season,
            holiday=holiday,
            workingday=workingday,
            weather=weather,
            temp=temp,
            atemp=atemp,
            hum=hum,
            windspeed=windspeed,
            selected_date=selected_date
        )

        # The notebook trained the model on log1p(cnt).
        prediction_log = model.predict(input_df)[0]
        prediction = max(0, round(float(np.expm1(prediction_log))))

        st.success(f"🚲 Predicted Bike Demand: **{prediction:,} bikes**")
        st.info(
            "Prediction generated using the tuned Random Forest regression "
            "model from the Bike Rental project."
        )

        if GEMINI_ENABLED:
            with st.spinner("Getting AI insight from Gemini..."):
                try:
                    suggestion = get_gemini_suggestion(
                        prediction,
                        {
                            "date": str(selected_date),
                            "month": month,
                            "hour": hour,
                            "weekday": weekday,
                            "season": season,
                            "holiday": holiday,
                            "workingday": workingday,
                            "weather": weather,
                            "temperature": temp,
                            "feels_like": atemp,
                            "humidity": hum,
                            "windspeed": windspeed,
                            "time_of_day": tod,
                            "is_weekend": bool(is_weekend),
                            "rush_hour": bool(rush_hour),
                        }
                    )
                    st.subheader("💡 Gemini Suggestion")
                    st.write(suggestion)
                except Exception as e:
                    st.warning(f"Gemini suggestion unavailable: {e}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.exception(e)
