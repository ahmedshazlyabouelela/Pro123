
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(
    BASE_DIR / "lightgbm_air_quality_model.joblib"
)

preprocessor = joblib.load(
    BASE_DIR / "preprocessor.joblib"
)

metadata = joblib.load(
    BASE_DIR / "metadata.joblib"
)


def prepare_input(
    dt,
    station,
    wd,
    pm25,
    pm10,
    so2,
    no2,
    co,
    o3,
    temp,
    pres,
    dewp,
    rain,
    wspm,
    pm25_lag1,
    pm25_lag2,
    pm10_lag1,
    pm10_lag2,
    o3_lag1,
    o3_lag2
):

    hour = dt.hour
    month = dt.month

    row = {
        "PM2.5": pm25,
        "PM10": pm10,
        "SO2": so2,
        "NO2": no2,
        "CO": co,
        "O3": o3,

        "TEMP": temp,
        "PRES": pres,
        "DEWP": dewp,
        "RAIN": rain,
        "WSPM": wspm,

        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),

        "month_sin": np.sin(2 * np.pi * month / 12),
        "month_cos": np.cos(2 * np.pi * month / 12),

        "PM2.5_lag_1": pm25_lag1,
        "PM2.5_lag_2": pm25_lag2,

        "PM10_lag_1": pm10_lag1,
        "PM10_lag_2": pm10_lag2,

        "O3_lag_1": o3_lag1,
        "O3_lag_2": o3_lag2,

        "wd": wd,
        "station": station
    }

    columns = (
        metadata["new_numeric_features"]
        + metadata["categorical_features"]
    )

    return pd.DataFrame([row], columns=columns)


def tuned_prediction(probabilities):

    probability_map = {
        str(cls): float(prob)
        for cls, prob in zip(
            model.classes_,
            probabilities
        )
    }

    toxic_threshold = metadata["thresholds"]["Toxic"]
    unhealthy_threshold = metadata["thresholds"]["Not Healthy"]

    if probability_map.get("Toxic", 0) >= toxic_threshold:
        prediction = "Toxic"

    elif probability_map.get("Not Healthy", 0) >= unhealthy_threshold:
        prediction = "Not Healthy"

    else:
        prediction = "Healthy"

    return prediction, probability_map


st.set_page_config(
    page_title="Air Quality Classification",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 Air Quality Classification System")

st.write(
    "Predict air quality using pollution, weather, "
    "time and previous-hour sensor readings."
)


with st.form("prediction_form"):

    col1, col2, col3 = st.columns(3)

    with col1:

        st.subheader("General Information")

        station = st.selectbox(
            "Station",
            metadata["stations"]
        )

        wd = st.selectbox(
            "Wind Direction",
            metadata["wind_directions"]
        )

        date_value = st.date_input("Date")

        time_value = st.time_input("Time")


    with col2:

        st.subheader("Pollutants")

        pm25 = st.number_input(
            "PM2.5",
            min_value=0.0,
            value=50.0
        )

        pm10 = st.number_input(
            "PM10",
            min_value=0.0,
            value=80.0
        )

        so2 = st.number_input(
            "SO2",
            min_value=0.0,
            value=10.0
        )

        no2 = st.number_input(
            "NO2",
            min_value=0.0,
            value=40.0
        )

        co = st.number_input(
            "CO",
            min_value=0.0,
            value=800.0
        )

        o3 = st.number_input(
            "O3",
            min_value=0.0,
            value=60.0
        )


    with col3:

        st.subheader("Weather")

        temp = st.number_input(
            "Temperature",
            value=20.0
        )

        pres = st.number_input(
            "Pressure",
            value=1010.0
        )

        dewp = st.number_input(
            "Dew Point",
            value=5.0
        )

        rain = st.number_input(
            "Rain",
            min_value=0.0,
            value=0.0
        )

        wspm = st.number_input(
            "Wind Speed",
            min_value=0.0,
            value=2.0
        )


    st.subheader("Previous Sensor Readings")

    c1, c2, c3 = st.columns(3)

    with c1:

        pm25_lag1 = st.number_input(
            "PM2.5 - 1 hour ago",
            min_value=0.0,
            value=50.0
        )

        pm25_lag2 = st.number_input(
            "PM2.5 - 2 hours ago",
            min_value=0.0,
            value=50.0
        )


    with c2:

        pm10_lag1 = st.number_input(
            "PM10 - 1 hour ago",
            min_value=0.0,
            value=80.0
        )

        pm10_lag2 = st.number_input(
            "PM10 - 2 hours ago",
            min_value=0.0,
            value=80.0
        )


    with c3:

        o3_lag1 = st.number_input(
            "O3 - 1 hour ago",
            min_value=0.0,
            value=60.0
        )

        o3_lag2 = st.number_input(
            "O3 - 2 hours ago",
            min_value=0.0,
            value=60.0
        )


    submit = st.form_submit_button(
        "Predict Air Quality",
        use_container_width=True
    )


if submit:

    dt = datetime.combine(
        date_value,
        time_value
    )

    input_df = prepare_input(
        dt,
        station,
        wd,

        pm25,
        pm10,
        so2,
        no2,
        co,
        o3,

        temp,
        pres,
        dewp,
        rain,
        wspm,

        pm25_lag1,
        pm25_lag2,

        pm10_lag1,
        pm10_lag2,

        o3_lag1,
        o3_lag2
    )

    X_input = preprocessor.transform(input_df)

    probabilities = model.predict_proba(
        X_input
    )[0]

    prediction, probability_map = tuned_prediction(
        probabilities
    )


    if prediction == "Healthy":

        st.success(
            f"Predicted Air Quality: {prediction}"
        )

    elif prediction == "Not Healthy":

        st.warning(
            f"Predicted Air Quality: {prediction}"
        )

    else:

        st.error(
            f"Predicted Air Quality: {prediction}"
        )


    st.subheader("Prediction Probabilities")

    prob_df = pd.DataFrame(
        {
            "Class": probability_map.keys(),
            "Probability": probability_map.values()
        }
    )

    prob_df = prob_df.set_index("Class")

    st.bar_chart(prob_df)

    st.dataframe(
        (prob_df * 100).round(2).rename(
            columns={
                "Probability":
                "Probability (%)"
            }
        ),
        use_container_width=True
    )
