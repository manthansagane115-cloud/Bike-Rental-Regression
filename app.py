# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import datetime
import joblib

@st.cache_resource
def load_artifacts():
    model=joblib.load("model.pkl")
    with open("encoder.pkl", "rb") as f: encoder = pickle.load(f)
    with open("feature_columns.pkl", "rb") as f: feature_columns = pickle.load(f)
    with open("categorical_features.pkl", "rb") as f: categorical_features = pickle.load(f)
    with open("numerical_features.pkl", "rb") as f: numerical_features = pickle.load(f)
    return model, encoder, feature_columns, categorical_features, numerical_features

model, encoder, feature_columns, categorical_features, numerical_features = load_artifacts()

def time_of_day(hour):
    if hour < 6: return 'Night'
    elif hour < 12: return 'Morning'
    elif hour < 17: return 'Afternoon'
    elif hour < 21: return 'Evening'
    else: return 'Night'

st.title("🚲 Bike Rental Demand Predictor")

col1, col2 = st.columns(2)
with col1:
    date = st.date_input("Date", datetime.date(2012, 6, 15))
    hr = st.slider("Hour", 0, 23, 8)
    season = st.selectbox("Season", [1,2,3,4], format_func=lambda x: {1:"Winter",2:"Spring",3:"Summer",4:"Fall"}[x])
    weathersit = st.selectbox("Weather", [1,2,3,4], format_func=lambda x: {1:"Clear",2:"Mist",3:"Light Rain/Snow",4:"Heavy Rain/Snow"}[x])
    holiday = st.checkbox("Holiday")
    workingday = st.checkbox("Working Day", value=True)

with col2:
    temp = st.slider("Temperature (normalized 0-1)", 0.0, 1.0, 0.5)
    atemp = st.slider("Feels-like Temp (normalized 0-1)", 0.0, 1.0, 0.5)
    hum = st.slider("Humidity (normalized 0-1)", 0.0, 1.0, 0.5)
    windspeed = st.slider("Windspeed (normalized 0-1)", 0.0, 1.0, 0.2)

if st.button("Predict Rentals"):
    raw_input = {
        "dteday": str(date), "season": season, "yr": date.year - 2011,
        "mnth": date.month, "hr": hr, "holiday": int(holiday),
        "weekday": date.weekday(), "workingday": int(workingday),
        "weathersit": weathersit, "temp": temp, "atemp": atemp,
        "hum": hum, "windspeed": windspeed
    }

    row = pd.DataFrame([raw_input])
    row['dteday'] = pd.to_datetime(row['dteday'])
    row['day'] = row['dteday'].dt.day
    row['weekofyear'] = row['dteday'].dt.isocalendar().week.astype(int)
    row['is_weekend'] = row['weekday'].isin([0, 6]).astype(int)
    row['rush_hour'] = row['hr'].isin([7,8,9,17,18,19]).astype(int)
    row['time_of_day'] = row['hr'].apply(time_of_day)
    row['hr_sin'] = np.sin(2*np.pi*row['hr']/24)
    row['hr_cos'] = np.cos(2*np.pi*row['hr']/24)
    row['month_sin'] = np.sin(2*np.pi*row['mnth']/12)
    row['month_cos'] = np.cos(2*np.pi*row['mnth']/12)
    row['weekday_sin'] = np.sin(2*np.pi*row['weekday']/7)
    row['weekday_cos'] = np.cos(2*np.pi*row['weekday']/7)
    row['temp_difference'] = row['atemp'] - row['temp']
    row['comfort_index'] = row['temp'] * (1 - row['hum'])


    row[categorical_features] = row[categorical_features].astype(str)
    cat = encoder.transform(row[categorical_features])
    cat = pd.DataFrame(cat, columns=encoder.get_feature_names_out(categorical_features))
    num = row[numerical_features].reset_index(drop=True)
    X_new = pd.concat([num, cat], axis=1).reindex(columns=feature_columns, fill_value=0)

    log_pred = model.predict(X_new)[0]
    st.success(f"Predicted rentals: **{int(np.expm1(log_pred))}** bikes")
