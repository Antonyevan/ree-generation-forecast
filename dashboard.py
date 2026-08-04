import json
import os
import streamlit as st
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from features import load_and_engineer, time_based_split

st.set_page_config(page_title="Spain Solar Forecast", layout="wide")
st.title("☀️ Spain Solar Generation Forecast")
st.caption("Gradient Boosting model vs REE's official day-ahead forecast")

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

        actual_df = pd.DataFrame(live_data["actual_solar"])
        forecast_df = pd.DataFrame(live_data["forecast_solar"])

        actual_df["datetime"] = pd.to_datetime(actual_df["datetime"])
        forecast_df["datetime"] = pd.to_datetime(forecast_df["datetime"])

        actual_df = actual_df[["datetime", "value"]].rename(columns={"value": "Actual"}).set_index("datetime")
        forecast_df = forecast_df[["datetime", "value"]].rename(columns={"value": "REE Forecast"}).set_index("datetime")

        # Forecast is hourly, actual is 5-min — reindex forecast onto actual's
        # timestamps and forward-fill so it renders as a continuous line
        forecast_aligned = forecast_df.reindex(actual_df.index.union(forecast_df.index)).sort_index()
        forecast_aligned = forecast_aligned.ffill()
        forecast_aligned = forecast_aligned.reindex(actual_df.index)

        live_chart_df = actual_df.join(forecast_aligned).sort_index()

        st.line_chart(live_chart_df[["Actual", "REE Forecast"]])

        fetched_at = pd.to_datetime(live_data["fetched_at"])
        st.caption(f"📡 Live data from REE's ESIOS API — last fetched {fetched_at.strftime('%Y-%m-%d %H:%M')}")
        st.caption(
            "Note: per ESIOS API terms of use, this data is fetched and cached "
            "periodically rather than queried live per visitor."
        )


# ── HISTORICAL TAB ────────────────────────────────────────────────────
with tab_historical:

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