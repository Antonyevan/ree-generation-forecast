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
@st.cache_data(ttl=3600)  # rebuild at most once per hour, so weekly data refreshes actually get picked up
def get_recent_model_and_test():
    with open("data/recent_solar_data.json") as f:
        recent_data = json.load(f)
    df_clean = build_live_features(recent_data)
    train, test = time_based_split(df_clean, test_days=60)
    model = GradientBoostingRegressor(random_state=42)
    model.fit(train[FEATURES], train['generation solar'])
    test = test.copy()
    test['model_pred'] = model.predict(test[FEATURES])
    return model, test


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
        recent_model, recent_test = get_recent_model_and_test()

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
                live_display.set_index('time')[['Actual', 'REE Forecast', 'Gradient Boosting Forecast']],
                color=['#9ca3af', '#ef4444', '#3b82f6']
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
            st.caption(f"📡 Live data from REE's ESIOS API — last fetched {fetched_at.strftime('%Y-%m-%d %H:%M')} UTC")
            st.caption(
                "Note: the MAE above reflects only the current ~24-48h window shown — "
                "a small, noisy sample."
            )
            st.caption(
                "This model is trained on the last ~1 year of ESIOS data — see "
                "'Project Journey' in the README for why this differs from the "
                "Historical Replay tab's model."
            )

            # ── Full test-period daily breakdown ────────────────────
            full_daily = recent_test.copy()
            full_daily['date'] = full_daily['time'].dt.date
            full_daily['model_error'] = abs(full_daily['generation solar'] - full_daily['model_pred'])
            full_daily['baseline_error'] = abs(full_daily['generation solar'] - full_daily['forecast solar day ahead'])

            full_daily_summary = full_daily.groupby('date')[['model_error', 'baseline_error']].mean()
            total_days = len(full_daily_summary)
            model_wins = (full_daily_summary['model_error'] < full_daily_summary['baseline_error']).sum()
            win_rate = model_wins / total_days * 100

            st.markdown("### 🏆 Back-tested performance")
            st.metric(
                "Model win rate over full test period",
                f"{model_wins} of {total_days} days ({win_rate:.0f}%)"
            )

            st.markdown("#### Daily error over the full test period")
            chart_df = full_daily_summary.rename(columns={
                'model_error': 'Gradient Boosting Forecast',
                'baseline_error': 'REE Forecast'
            })
            st.line_chart(
                chart_df[['Gradient Boosting Forecast', 'REE Forecast']],
                color=['#3b82f6', '#ef4444']
            )

            # Find the actual weakest stretch dynamically, rather than hardcoding a date
            full_daily_summary_sorted = full_daily_summary.sort_index()
            worst_week_start = full_daily_summary_sorted['model_error'].rolling(7).mean().idxmax()

            if pd.notna(worst_week_start):
                st.caption(
                    f"The model wins most days, but performance dips in some stretches "
                    f"(currently weakest around {worst_week_start.strftime('%b %d')}) — "
                    f"a realistic reminder that no single trained model performs "
                    f"uniformly forever. See 'Project Journey' in the README."
                )
            else:
                st.caption(
                    "Performance varies day to day — see 'Project Journey' in the README."
                )

            st.markdown("### 📅 Most recent week")
            st.caption("Shown for transparency — a single week is a small, noisy sample, not a representative average.")

            daily_summary = full_daily_summary.tail(7).round(1)
            daily_summary['Winner'] = daily_summary.apply(
                lambda row: '🟦 Model' if row['model_error'] < row['baseline_error'] else '🟥 REE',
                axis=1
            )
            daily_summary = daily_summary.rename(columns={
                'model_error': 'Model MAE (MW)',
                'baseline_error': 'REE MAE (MW)'
            })

            st.dataframe(daily_summary, use_container_width=True)


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
        day_data_display.set_index('time')[['Actual', 'REE Forecast', 'Gradient Boosting Forecast']],
        color=['#9ca3af', '#ef4444', '#3b82f6']
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

    overall_model_mae = abs(test["generation solar"] - test["model_pred"]).mean()
    overall_baseline_mae = abs(test["generation solar"] - test["forecast solar day ahead"]).mean()
    diff_pct = (overall_baseline_mae - overall_model_mae) / overall_baseline_mae * 100

    if diff_pct > 0:
        comparison_text = f"outperforms REE's 2018 forecast by {diff_pct:.0f}% on average"
    else:
        comparison_text = f"currently underperforms REE's 2018 forecast by {abs(diff_pct):.0f}% on average"

    st.caption(
        f"This model is trained on 2015-2018 historical data and {comparison_text} "
        f"({overall_model_mae:.1f} MW vs {overall_baseline_mae:.1f} MW MAE, full test period). "
        "See 'Project Journey' in the README for why a separate, more recent model "
        "powers the Live tab."
    )