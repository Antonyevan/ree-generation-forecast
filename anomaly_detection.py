import json
import os
import pandas as pd


def detect_anomalies(daily_summary, std_threshold=2):
    """
    Flags days where model error is statistically unusual —
    mean + (std_threshold * standard deviation) over the given period.

    Returns a DataFrame of only the anomalous days, for further
    investigation (e.g. correlating with weather or grid events
    to inform future feature engineering).
    """
    error_mean = daily_summary['model_error'].mean()
    error_std = daily_summary['model_error'].std()
    threshold = error_mean + std_threshold * error_std

    anomalies = daily_summary[daily_summary['model_error'] > threshold]
    return anomalies, threshold


def record_anomalies(anomalies, log_path="anomaly_log.json"):
    """
    Appends newly detected anomalies to a persistent log, avoiding duplicates.

    Each entry records the date, the model error on that date, and when
    it was first recorded — building a durable history of flagged days
    over time, independent of any single dashboard session or test window.

    Returns the list of newly added entries (empty if nothing new).
    """
    if os.path.exists(log_path):
        with open(log_path) as f:
            log = json.load(f)
    else:
        log = []

    existing_dates = {entry["date"] for entry in log}
    recorded_at = pd.Timestamp.now(tz="UTC").isoformat()

    new_entries = [
        {"date": str(date), "model_error": float(row["model_error"]), "recorded_at": recorded_at}
        for date, row in anomalies.iterrows()
        if str(date) not in existing_dates
    ]

    log.extend(new_entries)

    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    return new_entries