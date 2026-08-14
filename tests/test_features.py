

import pandas as pd
from features import load_and_engineer

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