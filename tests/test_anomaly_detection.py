import pandas as pd
from anomaly_detection import detect_anomalies, record_anomalies


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


def test_record_anomalies_writes_new_entries(tmp_path):
    """Recording anomalies should create a log file with the correct entries."""
    log_path = tmp_path / "test_anomaly_log.json"
    df = pd.DataFrame({'model_error': [777.0]}, index=pd.to_datetime(['2026-01-15']))

    new_entries = record_anomalies(df, log_path=str(log_path))

    assert len(new_entries) == 1
    assert new_entries[0]['model_error'] == 777.0
    assert log_path.exists()


def test_record_anomalies_avoids_duplicates(tmp_path):
    """Recording the same anomaly date twice should not create a duplicate entry."""
    log_path = tmp_path / "test_anomaly_log.json"
    df = pd.DataFrame({'model_error': [777.0]}, index=pd.to_datetime(['2026-01-15']))

    record_anomalies(df, log_path=str(log_path))
    second_call_entries = record_anomalies(df, log_path=str(log_path))

    assert len(second_call_entries) == 0
