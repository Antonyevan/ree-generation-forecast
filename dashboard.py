#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 18:12:43 2026

@author: antonyevanalosius
"""

import streamlit as st
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from features import load_and_engineer, time_based_split

st.set_page_config(page_title="Spain Solar Forecast", layout="wide")
st.title("☀️ Spain Solar Generation Forecast")
st.caption("Model vs REE's official day-ahead forecast")

# Load data and train (cached so it doesn't retrain every time someone interacts)
@st.cache_data
def get_data_and_model():
    df_clean = load_and_engineer()
    train, test = time_based_split(df_clean)
    features = ['hour', 'day_of_week', 'solar_lag_24h', 'wind_lag_24h', 'solar_rolling_3h']
    model = GradientBoostingRegressor(random_state=42)
    model.fit(train[features], train['generation solar'])
    test = test.copy()
    test['model_pred'] = model.predict(test[features])
    return test

test = get_data_and_model()

# Let the user pick a date to "replay"
available_dates = sorted(test['time'].dt.date.unique())
selected_date = st.select_slider("Select a date", options=available_dates, value=available_dates[0])

day_data = test[test['time'].dt.date == selected_date]

st.line_chart(
    day_data.set_index('time')[['generation solar', 'forecast solar day ahead', 'model_pred']]
)

col1, col2 = st.columns(2)
col1.metric("REE forecast error (MAE)", f"{abs(day_data['generation solar'] - day_data['forecast solar day ahead']).mean():.1f} MW")
col2.metric("Your model error (MAE)", f"{abs(day_data['generation solar'] - day_data['model_pred']).mean():.1f} MW")

st.info("📡 Live REE data is currently unavailable (outage since July 24, 2026). This dashboard replays historical test-period data.")