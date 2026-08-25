import streamlit as st
import pandas as pd
import numpy as np
import joblib

from preprocessing import create_features


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Bike Rental Demand Prediction",
    page_icon="🚲",
    layout="wide"
)


# ============================================================
# LOAD MODEL AND ARTIFACTS
# ============================================================

@st.cache_resource
def load_artifacts():

    model = joblib.load(
        "bike_rental_rf_model.pkl"
    )

    encoder = joblib.load(
        "bike_rental_encoder.pkl"
    )

    feature_names = joblib.load(
        "bike_rental_feature_names.pkl"
    )

    return model, encoder, feature_names


model, encoder, feature_names = load_artifacts()


# ============================================================
# TITLE
# ============================================================

st.title("🚲 Bike Rental Demand Prediction")

st.markdown(
    """
    ### Predict Bike Rental Demand

    This application predicts the expected number of bike rentals
    based on date, time, weather and working-day conditions.
    """
)


st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("📋 Input Parameters")


# ============================================================
# DATE
# ============================================================

selected_date = st.sidebar.date_input(
    "Select Date",
    value=pd.Timestamp("2012-06-15")
)


# ============================================================
# YEAR
# ============================================================

year = selected_date.year

if year <= 2011:
    yr = 0
else:
    yr = 1


st.sidebar.info(
    f"Year: {year}"
)


# ============================================================
# MONTH
# ============================================================

mnth = selected_date.month


st.sidebar.info(
    f"Month: {mnth}"
)


# ============================================================
# HOUR
# ============================================================

hr = st.sidebar.slider(
    "Hour of Day",
    min_value=0,
    max_value=23,
    value=12
)


# ============================================================
# WEEKDAY
# ============================================================

weekday = selected_date.weekday()

# Python:
# Monday = 0
# Sunday = 6

# Bike dataset convention:
# Sunday = 0
# Monday = 1
# ...
# Saturday = 6

weekday = (weekday + 1) % 7


weekday_names = {
    0: "Sunday",
    1: "Monday",
    2: "Tuesday",
    3: "Wednesday",
    4: "Thursday",
    5: "Friday",
    6: "Saturday"
}


st.sidebar.info(
    f"Weekday: {weekday_names[weekday]}"
)


# ============================================================
# TEMPERATURE
# ============================================================

temp = st.sidebar.number_input(
    "Temperature (normalized)",
    min_value=0.0,
    max_value=1.0,
    value=0.50,
    step=0.01
)


# ============================================================
# FEELING TEMPERATURE
# ============================================================

atemp = st.sidebar.number_input(
    "Feeling Temperature (normalized)",
    min_value=0.0,
    max_value=1.0,
    value=0.50,
    step=0.01
)


# ============================================================
# HUMIDITY
# ============================================================

hum = st.sidebar.number_input(
    "Humidity (normalized)",
    min_value=0.0,
    max_value=1.0,
    value=0.60,
    step=0.01
)


# ============================================================
# WINDSPEED
# ============================================================

windspeed = st.sidebar.number_input(
    "Windspeed (normalized)",
    min_value=0.0,
    max_value=1.0,
    value=0.20,
    step=0.01
)


# ============================================================
# SEASON
# ============================================================

season = st.sidebar.selectbox(
    "Season",
    [
        "springer",
        "summer",
        "fall",
        "winter"
    ]
)


# ============================================================
# HOLIDAY
# ============================================================

holiday = st.sidebar.selectbox(
    "Holiday",
    [
        "No",
        "Yes"
    ]
)


# ============================================================
# WORKING DAY
# ============================================================

workingday = st.sidebar.selectbox(
    "Working Day",
    [
        "No work",
        "Working Day"
    ]
)


# ============================================================
# WEATHER
# ============================================================

weathersit = st.sidebar.selectbox(
    "Weather Situation",
    [
        "Clear",
        "Mist",
        "Light Snow",
        "Heavy Rain"
    ]
)


# ============================================================
# PREDICTION
# ============================================================

predict_button = st.button(
    "🚲 Predict Bike Rentals",
    type="primary",
    use_container_width=True
)


if predict_button:

    try:

        # ----------------------------------------------------
        # CREATE ENGINEERED FEATURES
        # ----------------------------------------------------

        df_input = create_features(
            selected_date=selected_date,
            yr=yr,
            mnth=mnth,
            hr=hr,
            weekday=weekday,
            temp=temp,
            atemp=atemp,
            hum=hum,
            windspeed=windspeed,
            season=season,
            holiday=holiday,
            workingday=workingday,
            weathersit=weathersit
        )


        # ----------------------------------------------------
        # CATEGORICAL FEATURES
        # ----------------------------------------------------

        categorical_features = [
            "season",
            "holiday",
            "workingday",
            "weathersit",
            "time_of_day"
        ]


        # ----------------------------------------------------
        # NUMERICAL FEATURES
        # ----------------------------------------------------

        numerical_features = [
            "yr",
            "mnth",
            "hr",
            "weekday",
            "temp",
            "atemp",
            "hum",
            "windspeed",
            "day",
            "weekofyear",
            "is_weekend",
            "rush_hour",
            "hr_sin",
            "hr_cos",
            "month_sin",
            "month_cos",
            "weekday_sin",
            "weekday_cos",
            "temp_difference",
            "comfort_index"
        ]


        # ----------------------------------------------------
        # NUMERICAL DATA
        # ----------------------------------------------------

        X_numerical = (
            df_input[numerical_features]
            .reset_index(drop=True)
        )


        # ----------------------------------------------------
        # ENCODE CATEGORICAL FEATURES
        # ----------------------------------------------------

        X_encoded_cat = encoder.transform(
            df_input[categorical_features]
        )


        X_encoded_cat = pd.DataFrame(
            X_encoded_cat,
            columns=encoder.get_feature_names_out(
                categorical_features
            )
        )


        # ----------------------------------------------------
        # COMBINE FEATURES
        # ----------------------------------------------------

        X_final = pd.concat(
            [
                X_numerical,
                X_encoded_cat
            ],
            axis=1
        )


        # ----------------------------------------------------
        # ENSURE EXACT FEATURE ORDER
        # ----------------------------------------------------

        X_final = X_final.reindex(
            columns=feature_names,
            fill_value=0
        )


        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        prediction_log = model.predict(
            X_final
        )[0]


        # ----------------------------------------------------
        # CONVERT LOG PREDICTION TO ACTUAL COUNT
        # ----------------------------------------------------

        prediction = np.expm1(
            prediction_log
        )


        prediction = max(
            0,
            round(prediction)
        )


        # ====================================================
        # DISPLAY RESULT
        # ====================================================

        st.success(
            "Prediction generated successfully!"
        )


        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "Predicted Rentals",
                f"{prediction:,}"
            )


        with col2:

            st.metric(
                "Hour",
                f"{hr}:00"
            )


        with col3:

            st.metric(
                "Date",
                str(selected_date)
            )


        st.divider()


        # ----------------------------------------------------
        # SHOW INPUT SUMMARY
        # ----------------------------------------------------

        st.subheader("Prediction Input Summary")


        summary = pd.DataFrame({
            "Parameter": [
                "Date",
                "Hour",
                "Weekday",
                "Temperature",
                "Feeling Temperature",
                "Humidity",
                "Windspeed",
                "Season",
                "Holiday",
                "Working Day",
                "Weather"
            ],

            "Value": [
                str(selected_date),
                hr,
                weekday_names[weekday],
                temp,
                atemp,
                hum,
                windspeed,
                season,
                holiday,
                workingday,
                weathersit
            ]
        })


        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True
        )


        # ----------------------------------------------------
        # DEBUG / FEATURE CHECK
        # ----------------------------------------------------

        with st.expander(
            "View generated model features"
        ):

            st.dataframe(
                X_final,
                use_container_width=True
            )


    except Exception as e:

        st.error(
            f"Prediction failed: {e}"
        )

        st.exception(e)
