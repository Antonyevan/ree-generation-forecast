import json
import os
import streamlit as st
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from features import load_and_engineer, time_based_split, build_live_features

st.set_page_config(page_title="Spain Solar Forecast", layout="wide")
st.title("☀️ Spain Solar Generation Forecast")
st.caption("Gradient Boosting model vs REE's official day-ahead forecast")

FEATURES = ['hour', 'day_of_week', 'solar_lag_24h', 'wind_lag_24h', 'solar_rolling_3h']


# Train once, shared by both tabs
@st.cache_data
def get_trained_model_and_test():
    df_clean = load_and_engineer()
    train, test = time_based_split(df_clean)
    model = GradientBoostingRegressor(random_state=42)
    model.fit(train[FEATURES], train['generation solar'])
    test = test.copy()
    test['model_pred'] = model.predict(test[FEATURES])
    return model, test


model, test = get_trained_model_and_test()

tab_live, tab_historical = st.tabs(["🔴 Live (Today)", "📅 Historical Replay"])


# ── LIVE TAB ──────────────────────────────────────────────────────────
with tab_live:
    cache_path = "data/live_solar_cache.json"

    if not os.path.exists(cache_path):
        st.warning(
            "No live data cached yet. Run `python3 fetch_live_data.py` to fetch "
            "the latest data from REE's ESIOS API."
        )
    else:
        with open(cache_path) as f:
            live_data = json.load(f)

        live_features = build_live_features(live_data)

        if live_features.empty:
            st.warning(
                "Not enough live history yet to compute features (need ~24h+ of data). "
                "Run fetch_live_data.py again later once more history has accumulated."
            )
        else:
            live_features = live_features.copy()
            live_features['model_pred'] = model.predict(live_features[FEATURES])

            live_display = live_features.rename(columns={
                'generation solar': 'Actual',
                'forecast solar day ahead': 'REE Forecast',
                'model_pred': 'Gradient Boosting Forecast'
            })

            st.line_chart(
                live_display.set_index('time')[['Actual', 'REE Forecast', 'Gradient Boosting Forecast']]
            )

            col1, col2 = st.columns(2)
            col1.metric(
                "REE Official Forecast (MAE)",
                f"{abs(live_features['generation solar'] - live_features['forecast solar day ahead']).mean():.1f} MW"
            )
            col2.metric(
                "Gradient Boosting Forecast (MAE)",
                f"{abs(live_features['generation solar'] - live_features['model_pred']).mean():.1f} MW"
            )

            fetched_at = pd.to_datetime(live_data["fetched_at"])
            st.caption(f"📡 Live data from REE's ESIOS API — last fetched {fetched_at.strftime('%Y-%m-%d %H:%M')}")
            st.caption(
                "Note: per ESIOS API terms of use, this data is fetched and cached "
                "periodically rather than queried live per visitor. Live features are "
                "resampled to hourly to match the resolution the model was trained on."
            )


# ── HISTORICAL TAB ────────────────────────────────────────────────────
with tab_historical:
    available_dates = sorted(test['time'].dt.date.unique())
    default_date = available_dates[len(available_dates) // 2]

    selected_date = st.select_slider("Select a date", options=available_dates, value=default_date)

    day_data = test[test['time'].dt.date == selected_date]

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