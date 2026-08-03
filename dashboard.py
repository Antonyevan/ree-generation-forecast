import streamlit as st
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from features import load_and_engineer, time_based_split

st.set_page_config(page_title="Spain Solar Forecast", layout="wide")
st.title("☀️ Spain Solar Generation Forecast")
st.caption("Gradient Boosting model vs REE's official day-ahead forecast")


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

# Default to a representative mid-range date rather than the very first day
default_date = available_dates[len(available_dates) // 2]

selected_date = st.select_slider("Select a date", options=available_dates, value=default_date)

day_data = test[test['time'].dt.date == selected_date]

# Rename columns for display so the chart legend reads clearly
day_data_display = day_data.rename(columns={
    'generation solar': 'Actual',
    'forecast solar day ahead': 'REE Forecast',
    'model_pred': 'Gradient Boosting Forecast'
})

st.line_chart(
    day_data_display.set_index('time')[['Actual', 'REE Forecast', 'Gradient Boosting Forecast']]
)

col1, col2 = st.columns(2)
col1.metric(
    "REE Official Forecast (MAE)",
    f"{abs(day_data['generation solar'] - day_data['forecast solar day ahead']).mean():.1f} MW"
)
col2.metric(
    "Gradient Boosting Forecast (MAE)",
    f"{abs(day_data['generation solar'] - day_data['model_pred']).mean():.1f} MW"
)

st.caption(
    "Note: single-day comparisons are noisy — some days REE wins, some days the "
    "Gradient Boosting model wins. Averaged across the full 6-month test period, "
    "the Gradient Boosting model outperforms REE's forecast by ~22% (96.6 MW vs "
    "123.6 MW MAE)."
)

st.info(
    "📡 Live REE data is currently unavailable (outage since July 24, 2026). "
    "This dashboard replays historical test-period data."
)