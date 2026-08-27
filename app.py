import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import date
import google.generativeai as genai
import plotly.graph_objects as go

st.set_page_config(
    page_title="Bike Rental Demand Prediction",
    page_icon="🏍️",
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

# Friendlier display names for the feature-importance chart.
FEATURE_LABELS = {
    "yr": "Year", "mnth": "Month", "hr": "Hour", "weekday": "Weekday",
    "temp": "Temperature", "atemp": "Feels-like Temp", "hum": "Humidity",
    "windspeed": "Wind Speed", "day": "Day of Month", "weekofyear": "Week of Year",
    "is_weekend": "Weekend", "rush_hour": "Rush Hour",
    "hr_sin": "Hour (cyclical)", "hr_cos": "Hour (cyclical)",
    "month_sin": "Month (cyclical)", "month_cos": "Month (cyclical)",
    "weekday_sin": "Weekday (cyclical)", "weekday_cos": "Weekday (cyclical)",
    "temp_difference": "Temp Difference", "comfort_index": "Comfort Index",
    "season_fall": "Season: Fall", "season_springer": "Season: Spring",
    "season_summer": "Season: Summer", "season_winter": "Season: Winter",
    "holiday_No": "Not a Holiday", "holiday_Yes": "Holiday",
    "workingday_No work": "Non-Working Day", "workingday_Working Day": "Working Day",
    "weathersit_Clear": "Weather: Clear", "weathersit_Heavy Rain": "Weather: Heavy Rain",
    "weathersit_Light Snow": "Weather: Light Snow", "weathersit_Mist": "Weather: Mist",
    "time_of_day_Afternoon": "Afternoon", "time_of_day_Evening": "Evening",
    "time_of_day_Morning": "Morning", "time_of_day_Night": "Night",
}


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
    GEMINI_ENABLED = False


def get_gemini_suggestion(prediction, inputs: dict) -> str:
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


# ---------------------------------------------------------------------
# Theme / styling — interactive gradient background + bike branding
# ---------------------------------------------------------------------
st.markdown(
    """
    <style>
    @keyframes gradientShift {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .stApp {
        background: linear-gradient(-45deg, #0f2027, #2c5364, #1a936f, #114b5f);
        background-size: 400% 400%;
        animation: gradientShift 18s ease infinite;
    }

    .block-container {
        background: rgba(255, 255, 255, 0.94);
        border-radius: 20px;
        padding: 2.2rem 2.2rem 2.5rem 2.2rem;
        margin-top: 1.2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
        color: #111111 !important;
    }

    .block-container p,
    .block-container span,
    .block-container label,
    .block-container li,
    .block-container div,
    .block-container h1,
    .block-container h2,
    .block-container h3,
    .block-container h4,
    .block-container .stMarkdown,
    .block-container .stCaption,
    .block-container .stSlider label,
    .block-container .stSelectbox label,
    .block-container .stDateInput label,
    .block-container [data-testid="stMetricLabel"],
    .block-container [data-testid="stMetricValue"] {
        color: #111111 !important;
    }

    /* Keep the gradient bike title readable — it uses its own gradient fill */
    .block-container .bike-title {
        -webkit-text-fill-color: transparent !important;
    }

    /* Keep button text white since buttons have a dark gradient background */
    .block-container div.stButton > button,
    .block-container div.stButton > button p {
        color: #ffffff !important;
    } 
    }

    .bike-header {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 14px;
        margin-bottom: 0.2rem;
    }

    .bike-icon {
        font-size: 3rem;
        animation: pedal 2.5s ease-in-out infinite;
        display: inline-block;
    }

    @keyframes pedal {
        0%   { transform: rotate(0deg) translateY(0px); }
        50%  { transform: rotate(-8deg) translateY(-4px); }
        100% { transform: rotate(0deg) translateY(0px); }
    }

    .bike-title {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(90deg, #114b5f, #1a936f, #45c4b0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }

    .bike-subtitle {
        text-align: center;
        color: #4a4a4a;
        font-size: 1rem;
        margin-top: -6px;
        margin-bottom: 1.4rem;
    }

    div.stButton > button {
        background: linear-gradient(90deg, #1a936f, #114b5f);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.6rem 1.2rem;
        font-weight: 700;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        width: 100%;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(17, 75, 95, 0.4);
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #ecfdf5, #e0f2fe);
        border-radius: 14px;
        padding: 0.8rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="bike-header">
        <span class="bike-icon">🚲</span>
        <span class="bike-title">Bike Rental Demand Prediction</span>
    </div>
    <div class="bike-subtitle">
        Predict hourly bike rental demand from time, weather, and seasonal conditions
    </div>
    """,
    unsafe_allow_html=True
)

# Load model.
try:
    model = load_model()
except FileNotFoundError:
    st.error("model.pkl was not found. Put model.pkl in the same GitHub repository as app.py.")
    st.stop()
except Exception as e:
    st.error(f"Could not load model.pkl: {e}")
    st.stop()

st.header("📋 Prediction Inputs")

col1, col2 = st.columns(2)

with col1:
    selected_date = st.date_input(
        "📅 Date",
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
        "🍂 Season",
        ["springer", "summer", "fall", "winter"]
    )
    holiday = st.selectbox(
        "🎉 Holiday",
        ["No", "Yes"]
    )

with col2:
    workingday = st.selectbox(
        "💼 Working Day",
        ["Working Day", "No work"]
    )
    weather = st.selectbox(
        "🌦️ Weather Situation",
        ["Clear", "Mist", "Light Snow", "Heavy Rain"]
    )

st.subheader("🌡️ Weather Factors")
st.caption("Adjust the sliders to match current or forecasted conditions (0 = lowest, 1 = highest, normalized).")

w1, w2 = st.columns(2)
with w1:
    temp = st.slider(
        "Temperature (normalized)",
        min_value=0.0, max_value=1.0, value=0.60, step=0.01,
        help="Normalized air temperature — 0 is coldest, 1 is hottest in the dataset's range."
    )
    hum = st.slider(
        "Humidity (normalized)",
        min_value=0.0, max_value=1.0, value=0.50, step=0.01,
        help="Normalized relative humidity."
    )
with w2:
    atemp = st.slider(
        "Feeling Temperature (normalized)",
        min_value=0.0, max_value=1.0, value=0.60, step=0.01,
        help="Normalized 'feels-like' temperature."
    )
    windspeed = st.slider(
        "Wind Speed (normalized)",
        min_value=0.0, max_value=1.0, value=0.20, step=0.01,
        help="Normalized wind speed."
    )

is_weekend = int(weekday in [0, 6])
rush_hour = int(hour in [7, 8, 9, 17, 18, 19])
tod = time_of_day(hour)

st.subheader("⚙️ Engineered Features")
c1, c2, c3 = st.columns(3)
c1.metric("Time of Day", tod)
c2.metric("Weekend?", "Yes" if is_weekend else "No")
c3.metric("Rush Hour?", "Yes" if rush_hour else "No")

if not GEMINI_ENABLED:
    st.caption("ℹ️ Gemini suggestions are disabled — add GEMINI_API_KEY to secrets to enable them.")

predict_clicked = st.button("🚲 Predict Bike Demand", type="primary")

if predict_clicked:
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

        # -----------------------------------------------------------------
        # Charts: factors affecting demand
        # -----------------------------------------------------------------
        st.header("📊 What's Driving This Prediction")

        chart_tab1, chart_tab2 = st.tabs(["🔑 Top Influencing Factors", "🌦️ Current Conditions"])

        with chart_tab1:
            if hasattr(model, "feature_importances_"):
                importances = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS)
                importances = importances.sort_values(ascending=False).head(10)
                labels = [FEATURE_LABELS.get(f, f) for f in importances.index]

                fig_imp = go.Figure(go.Bar(
                    x=importances.values[::-1],
                    y=labels[::-1],
                    orientation="h",
                    marker=dict(
                        color=importances.values[::-1],
                        colorscale=[[0, "#a7f3d0"], [1, "#114b5f"]],
                    ),
                ))
                fig_imp.update_layout(
                    title="Top 10 factors the model relies on most (overall)",
                    xaxis_title="Relative importance",
                    yaxis_title="",
                    height=420,
                    margin=dict(l=10, r=10, t=50, b=10),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_imp, use_container_width=True)
                st.caption(
                    "These are the features the Random Forest model leans on most heavily "
                    "across all predictions — not just this one."
                )
            else:
                st.warning("This model doesn't expose feature importances.")

        with chart_tab2:
            cond_labels = ["Temperature", "Feels-like Temp", "Humidity", "Wind Speed"]
            cond_values = [temp, atemp, hum, windspeed]

            fig_cond = go.Figure(go.Bar(
                x=cond_labels,
                y=cond_values,
                marker_color=["#f97316", "#fb923c", "#38bdf8", "#94a3b8"],
                text=[f"{v:.2f}" for v in cond_values],
                textposition="outside",
            ))
            fig_cond.update_layout(
                title="Your current weather inputs (normalized 0–1)",
                yaxis=dict(range=[0, 1.1]),
                height=380,
                margin=dict(l=10, r=10, t=50, b=10),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_cond, use_container_width=True)

            fig_context = go.Figure(go.Bar(
                x=["Rush Hour", "Weekend", "Holiday", "Working Day"],
                y=[rush_hour, is_weekend, int(holiday == "Yes"), int(workingday == "Working Day")],
                marker_color="#1a936f",
            ))
            fig_context.update_layout(
                title="Context flags for this prediction (1 = Yes, 0 = No)",
                yaxis=dict(range=[0, 1.3], dtick=1),
                height=320,
                margin=dict(l=10, r=10, t=50, b=10),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_context, use_container_width=True)

        # -----------------------------------------------------------------
        # Gemini suggestion
        # -----------------------------------------------------------------
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
