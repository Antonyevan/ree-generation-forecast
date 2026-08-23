import pandas as pd
from anomaly_detection import detect_anomalies


def test_detect_anomalies_flags_only_true_outliers():
    """A clear outlier should be flagged against a realistic-sized sample;
    normal day-to-day variation should not."""
    normal_days = [100, 102, 98, 101, 99, 103, 97, 100, 102, 99] * 3  # 30 normal days
    df = pd.DataFrame({'model_error': normal_days + [500]})  # one clear outlier
    anomalies, threshold = detect_anomalies(df)

    assert len(anomalies) == 1
    assert anomalies['model_error'].iloc[0] == 500


def test_detect_anomalies_returns_empty_when_no_outliers():
    """Uniform, low-variance data should produce zero flagged anomalies."""
    df = pd.DataFrame({'model_error': [100, 101, 99, 100, 102]})
    anomalies, threshold = detect_anomalies(df)

    assert len(anomalies) == 0
