import pandas as pd
from features import load_and_engineer, build_live_features


def test_solar_rolling_3h_excludes_current_row(tmp_path):
    """Regression test for the leak: solar_rolling_3h must only average
    PAST hours, never the current hour's own value."""
    csv_path = tmp_path / "test_energy.csv"
    pd.DataFrame({
        "time": pd.date_range("2018-08-27", periods=30, freq="h", tz="UTC"),
        "generation solar": [i * 10 for i in range(30)],       # 0, 10, 20, ...
        "generation wind onshore": [5] * 30,
    }).to_csv(csv_path, index=False)

    df = load_and_engineer(path=str(csv_path))

    row = df[df["time"] == "2018-08-28 05:00:00+00:00"].iloc[0]

    # Row 29 (value 290). shift(1) -> uses row 28 (280) as "current" for rolling.
    # rolling(3) on shifted series at row29 = mean of original rows 26,27,28 = 260,270,280
    assert row["solar_rolling_3h"] == 270.0


def test_build_live_features_solar_rolling_3h_excludes_current_row():
    """Regression test: build_live_features had the same leak as
    load_and_engineer. This locks in the shift(1) fix for the live path.

    Needs 24+ hours of 5-min data, since solar_lag_24h (shift 24 on the
    hourly-resampled series) gets dropped by dropna() otherwise."""
    timestamps = pd.date_range("2026-08-01", periods=30 * 12, freq="5min", tz="UTC")  # 30 hours

    live_data = {
        "actual_solar": [
            {"datetime": str(ts), "value": i * 10} for i, ts in enumerate(timestamps)
        ],
        "actual_wind": [
            {"datetime": str(ts), "value": 5} for ts in timestamps
        ],
        "forecast_solar": [
            {"datetime": str(ts), "value": 1} for ts in timestamps
        ],
    }

    df = build_live_features(live_data)
    assert len(df) > 0, "No rows survived — need more hours of input data"

    generation = df["generation solar"].reset_index(drop=True)
    last_row = df.iloc[-1]

    # shift(1).rolling(3) at the last row = mean of the 3 hourly values
    # BEFORE the last one (positions -4, -3, -2), never including -1 itself.
    expected = generation.iloc[-4:-1].mean()
    leaked_value = generation.iloc[-3:].mean()  # what it'd be WITH the leak

    assert round(last_row["solar_rolling_3h"], 4) == round(expected, 4)
    assert round(last_row["solar_rolling_3h"], 4) != round(leaked_value, 4)


def test_no_feature_depends_on_future_data(tmp_path):
    """General leak check: mutating present/future rows must never
    change past rows' feature values. Doesn't require knowing which
    column might leak — catches any of them, known or not."""

    def make_csv(path, corrupt_from=None):
        n = 48
        solar = [i * 10 for i in range(n)]
        if corrupt_from is not None:
            solar[corrupt_from:] = [999999] * (n - corrupt_from)
        pd.DataFrame({
            "time": pd.date_range("2018-08-27", periods=n, freq="h", tz="UTC"),
            "generation solar": solar,
            "generation wind onshore": [5] * n,
        }).to_csv(path, index=False)

    clean_path = tmp_path / "clean.csv"
    corrupt_path = tmp_path / "corrupt.csv"
    make_csv(clean_path)
    make_csv(corrupt_path, corrupt_from=30)  # corrupt row 30 onward

    df_clean = load_and_engineer(path=str(clean_path))
    df_corrupt = load_and_engineer(path=str(corrupt_path))

    # Only compare rows strictly before the corruption point (row 30)
    cutoff = pd.Timestamp("2018-08-27", tz="UTC") + pd.Timedelta(hours=30)
    before_clean = df_clean[df_clean["time"] < cutoff].reset_index(drop=True)
    before_corrupt = df_corrupt[df_corrupt["time"] < cutoff].reset_index(drop=True)

    pd.testing.assert_frame_equal(before_clean, before_corrupt)