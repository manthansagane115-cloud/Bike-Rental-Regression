import pandas as pd
import numpy as np


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


def create_features(
    selected_date,
    yr,
    mnth,
    hr,
    weekday,
    temp,
    atemp,
    hum,
    windspeed,
    season,
    holiday,
    workingday,
    weathersit
):

    df = pd.DataFrame({
        "dteday": [pd.to_datetime(selected_date)],
        "yr": [yr],
        "mnth": [mnth],
        "hr": [hr],
        "weekday": [weekday],
        "temp": [temp],
        "atemp": [atemp],
        "hum": [hum],
        "windspeed": [windspeed],
        "season": [season],
        "holiday": [holiday],
        "workingday": [workingday],
        "weathersit": [weathersit]
    })

    # Date features
    df["day"] = df["dteday"].dt.day

    df["weekofyear"] = (
        df["dteday"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    # Weekend
    df["is_weekend"] = (
        df["weekday"]
        .isin([0, 6])
        .astype(int)
    )

    # Rush hour
    df["rush_hour"] = (
        df["hr"]
        .isin([7, 8, 9, 17, 18, 19])
        .astype(int)
    )

    # Time of day
    df["time_of_day"] = df["hr"].apply(time_of_day)

    # Cyclical features
    df["hr_sin"] = np.sin(
        2 * np.pi * df["hr"] / 24
    )

    df["hr_cos"] = np.cos(
        2 * np.pi * df["hr"] / 24
    )

    df["month_sin"] = np.sin(
        2 * np.pi * df["mnth"] / 12
    )

    df["month_cos"] = np.cos(
        2 * np.pi * df["mnth"] / 12
    )

    df["weekday_sin"] = np.sin(
        2 * np.pi * df["weekday"] / 7
    )

    df["weekday_cos"] = np.cos(
        2 * np.pi * df["weekday"] / 7
    )

    # Weather features
    df["temp_difference"] = (
        df["atemp"] - df["temp"]
    )

    df["comfort_index"] = (
        df["temp"] * (1 - df["hum"])
    )

    return df
