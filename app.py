"""
Bike Rental Demand Predictor — single-file Flask app
=====================================================

Replicates the EDA -> feature engineering -> Random Forest pipeline from the
ExcelR Bike Rental notebook, then serves a self-contained interactive
dashboard (HTML/CSS/JS embedded in this file, no template folder needed).

HOW TO RUN
----------
1. Put your training file next to this app.py and name it "Dataset.csv"
   (same columns as the notebook: dteday, season, yr, mnth, hr, holiday,
   weekday, workingday, weathersit, temp, atemp, hum, windspeed, casual,
   registered, cnt, [instant]).
2. pip install flask pandas numpy scikit-learn joblib
3. python app.py
4. Open http://127.0.0.1:5000

On first run the model trains and is cached to ./model_cache/ (joblib).
Delete that folder to force retraining (e.g. after changing Dataset.csv).

The dropdowns / slider ranges on the dashboard are generated dynamically
from whatever categories/ranges exist in YOUR Dataset.csv (via /api/meta),
so this works even if your category labels differ from the notebook's
sample ("spring"/"summer" vs 1/2/3/4, "Clear"/"Mist" vs 1/2/3, etc).
"""

import os
import json
import pickle
import traceback

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, Response

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import joblib

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(APP_DIR, "Dataset.csv")
CACHE_DIR = os.path.join(APP_DIR, "model_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

MODEL_PATH = os.path.join(CACHE_DIR, "model.pkl")
ENCODER_PATH = os.path.join(CACHE_DIR, "encoder.pkl")
META_PATH = os.path.join(CACHE_DIR, "meta.pkl")

RUSH_HOURS = [7, 8, 9, 17, 18, 19]
WEEKEND_DAYS = [0, 6]  # matches notebook convention (0 & 6 treated as weekend)


# ---------------------------------------------------------------------------
# Data loading / cleaning (mirrors notebook section 1)
# ---------------------------------------------------------------------------
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


def load_and_clean(path):
    df = pd.read_csv(path)
    df.replace("?", np.nan, inplace=True)

    # Parse date (try the notebook's dd-mm-yyyy format first, then fall back)
    try:
        df["dteday"] = pd.to_datetime(df["dteday"], format="%d-%m-%Y")
    except Exception:
        df["dteday"] = pd.to_datetime(df["dteday"], errors="coerce")

    numeric_cols = ["yr", "mnth", "hr", "weekday", "temp", "atemp",
                     "hum", "windspeed", "casual", "registered"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col].fillna(df[col].median(), inplace=True)

    categorical_cols = ["season", "holiday", "workingday", "weathersit"]
    for col in categorical_cols:
        if col in df.columns:
            df[col].fillna(df[col].mode()[0], inplace=True)

    df["cnt"] = pd.to_numeric(df["cnt"], errors="coerce")
    df.dropna(subset=["cnt", "dteday"], inplace=True)
    df["cnt_log"] = np.log1p(df["cnt"])
    return df


# ---------------------------------------------------------------------------
# Feature engineering (mirrors notebook section 3)
# ---------------------------------------------------------------------------
def engineer_features(df):
    df = df.copy()
    df["day"] = df["dteday"].dt.day
    df["weekofyear"] = df["dteday"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = df["weekday"].isin(WEEKEND_DAYS).astype(int)
    df["rush_hour"] = df["hr"].isin(RUSH_HOURS).astype(int)
    df["time_of_day"] = df["hr"].apply(time_of_day)

    df["hr_sin"] = np.sin(2 * np.pi * df["hr"] / 24)
    df["hr_cos"] = np.cos(2 * np.pi * df["hr"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["mnth"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["mnth"] / 12)
    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)

    df["temp_difference"] = df["atemp"] - df["temp"]
    df["comfort_index"] = df["temp"] * (1 - df["hum"])
    return df


def build_feature_matrix(df):
    y = df["cnt_log"]
    drop_cols = [c for c in ["cnt", "cnt_log", "casual", "registered", "instant"]
                 if c in df.columns]
    X = df.drop(columns=drop_cols)

    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_features = X.select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    ).columns.tolist()

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_cat = encoder.fit_transform(X[categorical_features])
    X_cat = pd.DataFrame(X_cat, columns=encoder.get_feature_names_out(categorical_features),
                          index=X.index)

    X_num = X[numerical_features].reset_index(drop=True)
    X_cat = X_cat.reset_index(drop=True)
    X_final = pd.concat([X_num, X_cat], axis=1)

    return X_final, y, encoder, categorical_features, numerical_features


# ---------------------------------------------------------------------------
# Training + metadata (baselines used by the dashboard's charts/insights)
# ---------------------------------------------------------------------------
def compute_meta(df, model, encoder, categorical_features, numerical_features, feature_columns):
    options = {feat: sorted([str(c) for c in cats])
               for feat, cats in zip(categorical_features, encoder.categories_)}

    ranges = {}
    for feat in numerical_features:
        if feat in ("day", "weekofyear", "hr_sin", "hr_cos", "month_sin", "month_cos",
                     "weekday_sin", "weekday_cos", "temp_difference", "comfort_index",
                     "is_weekend", "rush_hour"):
            continue  # derived automatically, not a user input
        ranges[feat] = {
            "min": float(df[feat].min()),
            "max": float(df[feat].max()),
            "mean": float(df[feat].mean()),
        }

    hourly_avg = df.groupby("hr")["cnt"].mean().round(1).to_dict()
    season_avg = df.groupby("season")["cnt"].mean().round(1).to_dict()
    weather_avg = df.groupby("weathersit")["cnt"].mean().round(1).to_dict()
    workingday_avg = df.groupby("workingday")["cnt"].mean().round(1).to_dict()

    casual_ratio_by_workingday = (
        df.groupby("workingday").apply(lambda g: float(g["casual"].sum() / g["cnt"].sum()))
        .to_dict()
    )
    overall_casual_ratio = float(df["casual"].sum() / df["cnt"].sum())

    quantiles = {
        "q25": float(df["cnt"].quantile(0.25)),
        "q50": float(df["cnt"].quantile(0.50)),
        "q75": float(df["cnt"].quantile(0.75)),
        "q90": float(df["cnt"].quantile(0.90)),
    }

    # Aggregate one-hot feature importances back to their original column
    raw_importances = dict(zip(feature_columns, model.feature_importances_))
    agg_importance = {}
    for feat in numerical_features:
        agg_importance[feat] = agg_importance.get(feat, 0) + raw_importances.get(feat, 0)
    for col, val in raw_importances.items():
        for cat in categorical_features:
            if col.startswith(cat + "_"):
                agg_importance[cat] = agg_importance.get(cat, 0) + val
                break
    top_importance = sorted(agg_importance.items(), key=lambda x: x[1], reverse=True)[:10]

    return {
        "options": options,
        "ranges": ranges,
        "hourly_avg": {int(k): v for k, v in hourly_avg.items()},
        "season_avg": {str(k): v for k, v in season_avg.items()},
        "weather_avg": {str(k): v for k, v in weather_avg.items()},
        "workingday_avg": {str(k): v for k, v in workingday_avg.items()},
        "casual_ratio_by_workingday": {str(k): v for k, v in casual_ratio_by_workingday.items()},
        "overall_casual_ratio": overall_casual_ratio,
        "overall_avg": float(df["cnt"].mean()),
        "quantiles": quantiles,
        "top_importance": [{"feature": f, "importance": round(float(v), 4)} for f, v in top_importance],
        "categorical_features": categorical_features,
        "numerical_features": numerical_features,
        "feature_columns": feature_columns,
        "n_rows": int(len(df)),
    }


def train_and_cache():
    df_raw = load_and_clean(DATA_PATH)
    df = engineer_features(df_raw)
    X_final, y, encoder, categorical_features, numerical_features = build_feature_matrix(df)
    feature_columns = X_final.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(
        n_estimators=200, max_depth=15, min_samples_split=2,
        min_samples_leaf=1, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    pred = np.expm1(model.predict(X_test))
    actual = np.expm1(y_test)
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(actual, pred))),
        "mae": float(mean_absolute_error(actual, pred)),
        "r2": float(r2_score(actual, pred)),
    }

    # Refit on all data for deployment, like the notebook does
    final_model = RandomForestRegressor(
        n_estimators=200, max_depth=15, min_samples_split=2,
        min_samples_leaf=1, random_state=42, n_jobs=-1
    )
    final_model.fit(X_final, y)

    meta = compute_meta(df, final_model, encoder, categorical_features,
                         numerical_features, feature_columns)
    meta["metrics"] = metrics

    joblib.dump(final_model, MODEL_PATH, compress=3)
    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(encoder, f)
    with open(META_PATH, "wb") as f:
        pickle.dump(meta, f)

    return final_model, encoder, meta


def load_or_train():
    if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH) and os.path.exists(META_PATH):
        model = joblib.load(MODEL_PATH)
        with open(ENCODER_PATH, "rb") as f:
            encoder = pickle.load(f)
        with open(META_PATH, "rb") as f:
            meta = pickle.load(f)
        return model, encoder, meta
    if os.path.exists(DATA_PATH):
        return train_and_cache()
    return None, None, None


MODEL, ENCODER, META = load_or_train()


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def build_row_from_input(payload):
    """Turn the dashboard's form payload into a single-row engineered dataframe."""
    dteday = pd.to_datetime(payload.get("dteday"))
    hr = int(payload["hr"])
    weekday = int(payload["weekday"])
    mnth = int(payload["mnth"])

    row = {
        "yr": float(payload["yr"]),
        "mnth": mnth,
        "hr": hr,
        "weekday": weekday,
        "temp": float(payload["temp"]),
        "atemp": float(payload["atemp"]),
        "hum": float(payload["hum"]),
        "windspeed": float(payload["windspeed"]),
        "season": payload["season"],
        "holiday": payload["holiday"],
        "workingday": payload["workingday"],
        "weathersit": payload["weathersit"],
        "dteday": dteday,
    }
    df = pd.DataFrame([row])
    df = engineer_features(df)
    return df


def predict_from_payload(payload):
    df_row = build_row_from_input(payload)

    numerical_features = META["numerical_features"]
    categorical_features = META["categorical_features"]

    X_num = df_row[numerical_features].reset_index(drop=True)
    X_cat = ENCODER.transform(df_row[categorical_features])
    X_cat = pd.DataFrame(X_cat, columns=ENCODER.get_feature_names_out(categorical_features))

    X_final = pd.concat([X_num, X_cat], axis=1)
    X_final = X_final.reindex(columns=META["feature_columns"], fill_value=0)

    pred_log = MODEL.predict(X_final)[0]
    predicted = float(np.expm1(pred_log))
    predicted = max(predicted, 0)

    # Demand level from training-data quantiles
    q = META["quantiles"]
    if predicted < q["q25"]:
        level = "Low"
    elif predicted < q["q50"]:
        level = "Moderate"
    elif predicted < q["q75"]:
        level = "High"
    else:
        level = "Very High"

    # Casual / registered split estimate
    ratio = META["casual_ratio_by_workingday"].get(
        str(payload["workingday"]), META["overall_casual_ratio"]
    )
    casual_est = predicted * ratio
    registered_est = predicted * (1 - ratio)

    hr = int(payload["hr"])
    hour_avg = META["hourly_avg"].get(hr, META["overall_avg"])
    season_avg = META["season_avg"].get(str(payload["season"]), META["overall_avg"])
    weather_avg = META["weather_avg"].get(str(payload["weathersit"]), META["overall_avg"])
    overall_avg = META["overall_avg"]

    # Rule-based textual insights
    insights = []
    if predicted > hour_avg * 1.15:
        insights.append(f"Predicted demand is above the typical average for {hr}:00 "
                         f"({hour_avg:.0f} rides), suggesting favorable conditions this hour.")
    elif predicted < hour_avg * 0.85:
        insights.append(f"Predicted demand is below the typical average for {hr}:00 "
                         f"({hour_avg:.0f} rides).")
    else:
        insights.append(f"Predicted demand is close to the usual average for {hr}:00 "
                         f"({hour_avg:.0f} rides).")

    if hr in RUSH_HOURS:
        insights.append("This falls within a commuter rush-hour window, historically a strong demand driver.")

    comfort = float(df_row["comfort_index"].iloc[0])
    if comfort > 0.4:
        insights.append("Weather comfort is high (warm and low humidity) — typically boosts ridership.")
    elif comfort < 0.15:
        insights.append("Weather comfort is low (cold and/or humid) — typically suppresses ridership.")

    if int(df_row["is_weekend"].iloc[0]) == 1:
        insights.append("Selected day falls on a weekend/off day, which tends to shift more usage to casual riders.")
    else:
        insights.append("Selected day is a regular weekday, which tends to favor registered/commuter riders.")

    if weather_avg < overall_avg * 0.8:
        insights.append("The chosen weather condition historically depresses demand versus the overall average.")

    hourly_pattern = [META["hourly_avg"].get(h, 0) for h in range(24)]

    return {
        "predicted_rentals": round(predicted),
        "demand_level": level,
        "casual_estimate": round(casual_est),
        "registered_estimate": round(registered_est),
        "comparison": {
            "prediction": round(predicted, 1),
            "hour_avg": round(hour_avg, 1),
            "season_avg": round(season_avg, 1),
            "weather_avg": round(weather_avg, 1),
            "overall_avg": round(overall_avg, 1),
        },
        "hourly_pattern": hourly_pattern,
        "selected_hour": hr,
        "insights": insights,
        "feature_importance": META["top_importance"],
        "quantiles": q,
    }


# ---------------------------------------------------------------------------
# Flask app + embedded frontend
# ---------------------------------------------------------------------------
app = Flask(__name__)

INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Bike Rental Demand Predictor</title>
<link rel="stylesheet" href="/static/style.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.4/chart.umd.min.js"></script>
</head>
<body>
<div class="app">
  <header class="topbar">
    <div class="brand">
      <span class="brand-icon">🚲</span>
      <div>
        <h1>Bike Rental Demand Predictor</h1>
        <p>Random Forest model &middot; trained on historical hourly rentals</p>
      </div>
    </div>
    <div id="model-badge" class="badge">loading model…</div>
  </header>

  <div id="data-warning" class="warning hidden">
    <strong>No trained model found.</strong> Place your <code>Dataset.csv</code> next to
    <code>app.py</code> and restart the app, then reload this page.
  </div>

  <main class="grid">
    <section class="card form-card">
      <h2>Trip Conditions</h2>
      <form id="predict-form">
        <div class="field">
          <label for="dteday">Date</label>
          <input type="date" id="dteday" name="dteday" required>
        </div>
        <div class="field">
          <label for="hr">Hour of day <span id="hr-val">8</span>:00</label>
          <input type="range" id="hr" name="hr" min="0" max="23" step="1" value="8">
        </div>
        <div class="field">
          <label for="weekday">Weekday (0-6)</label>
          <input type="number" id="weekday" name="weekday" min="0" max="6" value="1" required>
        </div>
        <div class="field select-field">
          <label for="season">Season</label>
          <select id="season" name="season" required></select>
        </div>
        <div class="field select-field">
          <label for="holiday">Holiday</label>
          <select id="holiday" name="holiday" required></select>
        </div>
        <div class="field select-field">
          <label for="workingday">Working day</label>
          <select id="workingday" name="workingday" required></select>
        </div>
        <div class="field select-field">
          <label for="weathersit">Weather</label>
          <select id="weathersit" name="weathersit" required></select>
        </div>
        <div class="field">
          <label for="temp">Temperature (normalized) <span id="temp-val"></span></label>
          <input type="range" id="temp" name="temp" min="0" max="1" step="0.01">
        </div>
        <div class="field">
          <label for="atemp">Feels-like temp (normalized) <span id="atemp-val"></span></label>
          <input type="range" id="atemp" name="atemp" min="0" max="1" step="0.01">
        </div>
        <div class="field">
          <label for="hum">Humidity (normalized) <span id="hum-val"></span></label>
          <input type="range" id="hum" name="hum" min="0" max="1" step="0.01">
        </div>
        <div class="field">
          <label for="windspeed">Windspeed (normalized) <span id="windspeed-val"></span></label>
          <input type="range" id="windspeed" name="windspeed" min="0" max="1" step="0.01">
        </div>
        <input type="hidden" id="yr" name="yr">
        <input type="hidden" id="mnth" name="mnth">

        <button type="submit" id="predict-btn">Predict Demand</button>
      </form>
    </section>

    <section class="card results-card">
      <div id="placeholder" class="placeholder">
        <p>Fill in the conditions on the left and click <strong>Predict Demand</strong> to see
        the forecast and insights here.</p>
      </div>

      <div id="results" class="results hidden">
        <div class="headline">
          <div class="headline-number">
            <span id="pred-value">0</span>
            <small>predicted rentals / hour</small>
          </div>
          <div id="demand-badge" class="demand-badge">—</div>
        </div>

        <div id="insights-list" class="insights"></div>

        <div class="charts-grid">
          <div class="chart-box">
            <h3>Casual vs Registered (est.)</h3>
            <canvas id="chart-split"></canvas>
          </div>
          <div class="chart-box">
            <h3>Prediction vs Historical Averages</h3>
            <canvas id="chart-compare"></canvas>
          </div>
          <div class="chart-box wide">
            <h3>Hourly Demand Pattern</h3>
            <canvas id="chart-hourly"></canvas>
          </div>
          <div class="chart-box wide">
            <h3>Top Drivers of Demand (model feature importance)</h3>
            <canvas id="chart-importance"></canvas>
          </div>
        </div>
      </div>
    </section>
  </main>

  <footer>
    <span id="metrics-line"></span>
  </footer>
</div>
<script src="/static/script.js"></script>
</body>
</html>
"""

STYLE_CSS = r"""
:root {
  --bg: #0f1420;
  --panel: #161d2e;
  --panel-2: #1c2438;
  --text: #eaf0ff;
  --muted: #93a0bd;
  --accent: #4f8cff;
  --accent-2: #35c78a;
  --warn: #ff9f43;
  --danger: #ff5d6c;
  --border: #26304a;
  --radius: 14px;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  background: radial-gradient(circle at top left, #1a2338, var(--bg));
  color: var(--text);
  min-height: 100vh;
}
.app { max-width: 1200px; margin: 0 auto; padding: 24px 20px 60px; }

.topbar {
  display: flex; align-items: center; justify-content: space-between;
  margin-bottom: 20px; flex-wrap: wrap; gap: 12px;
}
.brand { display: flex; align-items: center; gap: 14px; }
.brand-icon { font-size: 34px; }
.brand h1 { margin: 0; font-size: 22px; }
.brand p { margin: 2px 0 0; color: var(--muted); font-size: 13px; }
.badge {
  background: var(--panel-2); border: 1px solid var(--border);
  padding: 8px 14px; border-radius: 999px; font-size: 12px; color: var(--muted);
}
.badge.ok { color: var(--accent-2); border-color: var(--accent-2); }

.warning {
  background: #3a2412; border: 1px solid var(--warn); color: #ffd8a8;
  padding: 12px 16px; border-radius: var(--radius); margin-bottom: 18px; font-size: 14px;
}
.warning code { background: rgba(255,255,255,0.08); padding: 2px 6px; border-radius: 6px; }
.hidden { display: none !important; }

.grid {
  display: grid; grid-template-columns: 340px 1fr; gap: 20px;
}
@media (max-width: 880px) { .grid { grid-template-columns: 1fr; } }

.card {
  background: var(--panel); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 20px;
}
.card h2 { margin-top: 0; font-size: 16px; color: var(--text); }

.field { margin-bottom: 14px; display: flex; flex-direction: column; gap: 6px; }
.field label { font-size: 13px; color: var(--muted); }
.field input[type="range"] { accent-color: var(--accent); }
.field input[type="number"], .field input[type="date"], select {
  background: var(--panel-2); border: 1px solid var(--border); color: var(--text);
  padding: 8px 10px; border-radius: 8px; font-size: 14px;
}
select { appearance: none; }

#predict-btn {
  width: 100%; padding: 12px; margin-top: 6px; border: none; border-radius: 10px;
  background: linear-gradient(135deg, var(--accent), #7aa9ff);
  color: #0b1020; font-weight: 700; font-size: 14px; cursor: pointer;
  transition: transform 0.1s ease;
}
#predict-btn:hover { transform: translateY(-1px); }
#predict-btn:disabled { opacity: 0.6; cursor: progress; }

.results-card { min-height: 400px; }
.placeholder { color: var(--muted); font-size: 14px; padding: 40px 10px; text-align: center; }

.headline {
  display: flex; align-items: center; justify-content: space-between;
  border-bottom: 1px solid var(--border); padding-bottom: 16px; margin-bottom: 16px;
  flex-wrap: wrap; gap: 12px;
}
.headline-number span {
  font-size: 42px; font-weight: 800; color: var(--accent-2);
}
.headline-number small { display: block; color: var(--muted); font-size: 12px; }
.demand-badge {
  padding: 8px 16px; border-radius: 999px; font-weight: 700; font-size: 13px;
  background: var(--panel-2); border: 1px solid var(--border);
}
.demand-badge.Low { color: #93a0bd; }
.demand-badge.Moderate { color: var(--accent); border-color: var(--accent); }
.demand-badge.High { color: var(--warn); border-color: var(--warn); }
.demand-badge.Very-High { color: var(--danger); border-color: var(--danger); }

.insights { display: flex; flex-direction: column; gap: 8px; margin-bottom: 18px; }
.insights div {
  background: var(--panel-2); border-left: 3px solid var(--accent);
  padding: 8px 12px; border-radius: 8px; font-size: 13px; color: #d6ddf0;
}

.charts-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.chart-box { background: var(--panel-2); border: 1px solid var(--border); border-radius: 12px; padding: 14px; }
.chart-box.wide { grid-column: 1 / -1; }
.chart-box h3 { margin: 0 0 10px; font-size: 13px; color: var(--muted); font-weight: 600; }
.chart-box canvas { max-height: 260px; }
@media (max-width: 640px) { .charts-grid { grid-template-columns: 1fr; } }

footer { margin-top: 24px; text-align: center; color: var(--muted); font-size: 12px; }
"""

SCRIPT_JS = r"""
let META = null;
let charts = {};

const $ = (id) => document.getElementById(id);

function fillSelect(id, options, preferred) {
  const el = $(id);
  el.innerHTML = "";
  options.forEach((opt) => {
    const o = document.createElement("option");
    o.value = opt; o.textContent = opt;
    el.appendChild(o);
  });
  if (preferred && options.includes(preferred)) el.value = preferred;
}

function setRange(id, range) {
  const el = $(id);
  if (!range) return;
  const step = (range.max - range.min) > 5 ? 1 : 0.01;
  el.min = range.min; el.max = range.max; el.step = step;
  el.value = range.mean;
  const label = $(id + "-val");
  if (label) label.textContent = Number(range.mean).toFixed(2);
}

async function loadMeta() {
  const res = await fetch("/api/meta");
  const data = await res.json();
  if (!data.ready) {
    $("data-warning").classList.remove("hidden");
    $("model-badge").textContent = "no model loaded";
    return;
  }
  META = data;
  $("model-badge").textContent = `model ready · R2 ${data.metrics.r2.toFixed(3)} · ${data.n_rows} rows`;
  $("model-badge").classList.add("ok");
  $("metrics-line").textContent =
    `Model quality on held-out test data — RMSE: ${data.metrics.rmse.toFixed(1)}, ` +
    `MAE: ${data.metrics.mae.toFixed(1)}, R²: ${data.metrics.r2.toFixed(3)}`;

  fillSelect("season", data.options.season);
  fillSelect("holiday", data.options.holiday);
  fillSelect("workingday", data.options.workingday);
  fillSelect("weathersit", data.options.weathersit);

  setRange("temp", data.ranges.temp);
  setRange("atemp", data.ranges.atemp);
  setRange("hum", data.ranges.hum);
  setRange("windspeed", data.ranges.windspeed);

  const today = new Date();
  $("dteday").value = today.toISOString().slice(0, 10);
  $("yr").value = data.ranges.yr ? data.ranges.yr.max : today.getFullYear();
}

function bindLiveLabels() {
  ["temp", "atemp", "hum", "windspeed"].forEach((id) => {
    $(id).addEventListener("input", () => {
      $(id + "-val").textContent = Number($(id).value).toFixed(2);
    });
  });
  $("hr").addEventListener("input", () => {
    $("hr-val").textContent = $("hr").value;
  });
}

function destroyCharts() {
  Object.values(charts).forEach((c) => c && c.destroy());
  charts = {};
}

const CHART_COLORS = {
  accent: "#4f8cff", accent2: "#35c78a", warn: "#ff9f43", danger: "#ff5d6c",
  grid: "rgba(255,255,255,0.08)", text: "#93a0bd",
};
Chart.defaults.color = CHART_COLORS.text;
Chart.defaults.borderColor = CHART_COLORS.grid;

function renderCharts(result) {
  destroyCharts();

  charts.split = new Chart($("chart-split"), {
    type: "doughnut",
    data: {
      labels: ["Casual", "Registered"],
      datasets: [{
        data: [result.casual_estimate, result.registered_estimate],
        backgroundColor: [CHART_COLORS.warn, CHART_COLORS.accent],
        borderWidth: 0,
      }],
    },
    options: { plugins: { legend: { position: "bottom" } } },
  });

  const cmp = result.comparison;
  charts.compare = new Chart($("chart-compare"), {
    type: "bar",
    data: {
      labels: ["This prediction", "Hour avg", "Season avg", "Weather avg", "Overall avg"],
      datasets: [{
        data: [cmp.prediction, cmp.hour_avg, cmp.season_avg, cmp.weather_avg, cmp.overall_avg],
        backgroundColor: [CHART_COLORS.accent2, CHART_COLORS.accent, CHART_COLORS.accent,
                           CHART_COLORS.accent, CHART_COLORS.accent],
        borderRadius: 6,
      }],
    },
    options: {
      indexAxis: "y",
      plugins: { legend: { display: false } },
      scales: { x: { grid: { color: CHART_COLORS.grid } }, y: { grid: { display: false } } },
    },
  });

  charts.hourly = new Chart($("chart-hourly"), {
    type: "line",
    data: {
      labels: [...Array(24).keys()].map((h) => h + ":00"),
      datasets: [{
        label: "Avg rentals",
        data: result.hourly_pattern,
        borderColor: CHART_COLORS.accent,
        backgroundColor: "rgba(79,140,255,0.15)",
        fill: true, tension: 0.35, pointRadius: 2,
      }, {
        label: "Selected hour",
        data: [...Array(24).keys()].map((h) => h === result.selected_hour ? result.predicted_rentals : null),
        borderColor: CHART_COLORS.danger,
        backgroundColor: CHART_COLORS.danger,
        pointRadius: 7, showLine: false,
      }],
    },
    options: {
      plugins: { legend: { position: "bottom" } },
      scales: { x: { grid: { display: false } }, y: { grid: { color: CHART_COLORS.grid } } },
    },
  });

  const imp = result.feature_importance;
  charts.importance = new Chart($("chart-importance"), {
    type: "bar",
    data: {
      labels: imp.map((f) => f.feature),
      datasets: [{
        data: imp.map((f) => f.importance),
        backgroundColor: CHART_COLORS.accent2,
        borderRadius: 6,
      }],
    },
    options: {
      indexAxis: "y",
      plugins: { legend: { display: false } },
      scales: { x: { grid: { color: CHART_COLORS.grid } }, y: { grid: { display: false } } },
    },
  });
}

function renderInsights(result) {
  const list = $("insights-list");
  list.innerHTML = "";
  result.insights.forEach((text) => {
    const div = document.createElement("div");
    div.textContent = text;
    list.appendChild(div);
  });
}

async function handleSubmit(e) {
  e.preventDefault();
  const btn = $("predict-btn");
  btn.disabled = true; btn.textContent = "Predicting…";

  const payload = {
    dteday: $("dteday").value,
    hr: $("hr").value,
    weekday: $("weekday").value,
    season: $("season").value,
    holiday: $("holiday").value,
    workingday: $("workingday").value,
    weathersit: $("weathersit").value,
    temp: $("temp").value,
    atemp: $("atemp").value,
    hum: $("hum").value,
    windspeed: $("windspeed").value,
    yr: $("yr").value,
    mnth: new Date($("dteday").value).getMonth() + 1,
  };

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error((await res.json()).error || "Prediction failed");
    const result = await res.json();

    $("placeholder").classList.add("hidden");
    $("results").classList.remove("hidden");

    $("pred-value").textContent = result.predicted_rentals;
    const badge = $("demand-badge");
    badge.textContent = result.demand_level + " demand";
    badge.className = "demand-badge " + result.demand_level.replace(" ", "-");

    renderInsights(result);
    renderCharts(result);
  } catch (err) {
    alert("Prediction error: " + err.message);
  } finally {
    btn.disabled = false; btn.textContent = "Predict Demand";
  }
}

window.addEventListener("DOMContentLoaded", () => {
  bindLiveLabels();
  loadMeta();
  $("predict-form").addEventListener("submit", handleSubmit);
});
"""


@app.route("/")
def index():
    return Response(INDEX_HTML, mimetype="text/html")


@app.route("/static/style.css")
def style():
    return Response(STYLE_CSS, mimetype="text/css")


@app.route("/static/script.js")
def script():
    return Response(SCRIPT_JS, mimetype="application/javascript")


@app.route("/api/meta")
def api_meta():
    if META is None:
        return jsonify({"ready": False})
    payload = dict(META)
    payload["ready"] = True
    return jsonify(payload)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if META is None or MODEL is None or ENCODER is None:
        return jsonify({"error": "Model is not trained yet. Add Dataset.csv and restart."}), 503
    try:
        payload = request.get_json(force=True)
        result = predict_from_payload(payload)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    if MODEL is None:
        print("=" * 70)
        print("WARNING: Dataset.csv not found next to app.py — the dashboard")
        print("will load, but predictions are disabled until you add it and")
        print("restart the app.")
        print("=" * 70)
    app.run(debug=True, host="127.0.0.1", port=5000)
