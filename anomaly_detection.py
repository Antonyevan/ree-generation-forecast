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
