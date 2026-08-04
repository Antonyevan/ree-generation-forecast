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


# ── Historical model: trained on 2015-2018 Kaggle data ──────────────────
@st.cache_data
def get_historical_model_and_test():
    df_clean = load_and_engineer()
    train, test = time_based_split(df_clean)
    model = GradientBoostingRegressor(random_state=42)
    model.fit(train[FEATURES], train['generation solar'])
    test = test.copy()
    test['model_pred'] = model.predict(test[FEATURES])
    return test


# ── Recent model: trained on the last ~1 year of ESIOS data ────────────
@st.cache_data
def get_recent_model():
    with open("data/recent_solar_data.json") as f:
        recent_data = json.load(f)
    df_clean = build_live_features(recent_data)
    train, test = time_based_split(df_clean, test_days=60)
    model = GradientBoostingRegressor(random_state=42)
    model.fit(train[FEATURES], train['generation solar'])
    return model


tab_live, tab_historical = st.tabs(["🔴 Live (Today)", "📅 Historical Replay"])


# ── LIVE TAB — uses the recent-data model ───────────────────────────────
with tab_live:
    cache_path = "data/live_solar_cache.json"
    recent_model_path = "data/recent_solar_data.json"

    if not os.path.exists(cache_path):
        st.warning(
            "No live data cached yet. Run `python3 fetch_live_data.py` to fetch "
            "the latest data from REE's ESIOS API."
        )
    elif not os.path.exists(recent_model_path):
        st.warning(
            "Recent training data not found. Run `python3 pull_recent_data.py` first."
        )
    else:
        recent_model = get_recent_model()

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
            live_features['model_pred'] = recent_model.predict(live_features[FEATURES])

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
                "Note: the MAE above reflects only the current ~24-48h window shown — "
                "a small, noisy sample. On the full 60-day held-out test set, this "
                "model beats REE's forecast by 36.3% on average (638.5 MW vs 1001.7 "
                "MW MAE). Single-day results will vary."
            )
            st.caption(
                "This model is trained on the last ~1 year of ESIOS data — see "
                "'Project Journey' in the README for why this differs from the "
                "Historical Replay tab's model."
            )


# ── HISTORICAL TAB — uses the original 2015-2018 model ──────────────────
with tab_historical:
    test = get_historical_model_and_test()

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
        "This model is trained on 2015-2018 historical data and outperforms REE's "
        "2018 forecast by ~22% on average. It is shown here for comparison — see "
        "'Project Journey' in the README for why a separate, more recent model "
        "powers the Live tab."
    )