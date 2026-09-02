"""Computes and persists the live dashboard's current metrics.

Runs on a schedule via .github/workflows/refresh_metrics.yml, writing
latest_metrics.json — a durable snapshot of what the live dashboard is
currently showing, including how stale the underlying live data is.

This exists specifically so external consumers (e.g. the ree-assistant
project) can read an honest, persisted answer to "what is the model doing
right now" without needing to run Streamlit or retrain anything themselves.
"""

import json
import joblib
import pandas as pd

from features import build_live_features, time_based_split
from anomaly_detection import detect_anomalies

FEATURES = ['hour', 'day_of_week', 'solar_lag_24h', 'wind_lag_24h', 'solar_rolling_3h']


def main():
    with open("data/live_solar_cache.json") as f:
        live_data = json.load(f)
    live_data_fetched_at = pd.to_datetime(live_data["fetched_at"], utc=True)

    with open("data/recent_solar_data.json") as f:
        recent_data = json.load(f)

    df_clean = build_live_features(recent_data)
    train, test = time_based_split(df_clean, test_days=60)

    model = joblib.load("recent_model.pkl")
    test = test.copy()
    test['model_pred'] = model.predict(test[FEATURES])

    model_mae = float((test['generation solar'] - test['model_pred']).abs().mean())
    baseline_mae = float((test['generation solar'] - test['forecast solar day ahead']).abs().mean())
    improvement_pct = round((baseline_mae - model_mae) / baseline_mae * 100, 1)

    test['date'] = test['time'].dt.date
    test['model_error'] = (test['generation solar'] - test['model_pred']).abs()
    test['baseline_error'] = (test['generation solar'] - test['forecast solar day ahead']).abs()
    daily = test.groupby('date')[['model_error', 'baseline_error']].mean()

    win_rate_pct = round((daily['model_error'] < daily['baseline_error']).mean() * 100, 1)

    anomalies, _ = detect_anomalies(daily)
    anomaly_list = [
        {"date": str(date), "model_error": round(float(row["model_error"]), 1)}
        for date, row in anomalies.iterrows()
    ]

    computed_at = pd.Timestamp.now(tz="UTC")
    hours_since_live_fetch = round(
        (computed_at - live_data_fetched_at).total_seconds() / 3600, 1
    )

    output = {
        "computed_at": computed_at.isoformat(),
        "live_data_fetched_at": live_data_fetched_at.isoformat(),
        "hours_since_live_fetch": hours_since_live_fetch,
        "model_mae": round(model_mae, 1),
        "baseline_mae": round(baseline_mae, 1),
        "improvement_pct": improvement_pct,
        "win_rate_pct": win_rate_pct,
        "anomaly_count": len(anomaly_list),
        "anomalies": anomaly_list,
    }

    with open("latest_metrics.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"Wrote latest_metrics.json — MAE {model_mae:.1f}, win rate {win_rate_pct}%, "
          f"{len(anomaly_list)} anomalies, live data {hours_since_live_fetch}h old.")


if __name__ == "__main__":
    main()
