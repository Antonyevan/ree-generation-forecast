import pandas as pd


def load_and_engineer(path="data/energy_dataset.csv"):
    """Load raw energy data and build model-ready features."""
    df = pd.read_csv(path)
    df['time'] = pd.to_datetime(df['time'], utc=True)
    df = df.sort_values('time').reset_index(drop=True)

    df['solar_lag_24h'] = df['generation solar'].shift(24)
    df['wind_lag_24h'] = df['generation wind onshore'].shift(24)
    df['solar_rolling_3h'] = df['generation solar'].shift(1).rolling(window=3).mean()
    df['hour'] = df['time'].dt.hour
    df['day_of_week'] = df['time'].dt.dayofweek

    return df.dropna(subset=['solar_lag_24h', 'solar_rolling_3h', 'wind_lag_24h', 'generation solar'])


def time_based_split(df, test_days=180):
    """Split chronologically — train on the past, test on the most recent window."""
    split_date = df['time'].max() - pd.Timedelta(days=test_days)
    train = df[df['time'] < split_date]
    test = df[df['time'] >= split_date]
    return train, test


def build_live_features(live_data):
    """
    Build the same features used in training, but from live ESIOS data.

    live_data is the dict loaded from data/live_solar_cache.json — raw
    5-minute readings for actual solar and wind, plus hourly forecast.
    We resample to hourly first, since that's the resolution the model
    was trained on.
    """
    solar = pd.DataFrame(live_data["actual_solar"])
    wind = pd.DataFrame(live_data["actual_wind"])
    forecast = pd.DataFrame(live_data["forecast_solar"])

    for df in (solar, wind, forecast):
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

    # Resample 5-min readings to hourly (mean), matching training data's resolution
    solar_hourly = solar.set_index("datetime")["value"].resample("h").mean()
    wind_hourly = wind.set_index("datetime")["value"].resample("h").mean()
    forecast_hourly = forecast.set_index("datetime")["value"].resample("h").mean()

    df = pd.DataFrame({
        "generation solar": solar_hourly,
        "generation wind onshore": wind_hourly,
        "forecast solar day ahead": forecast_hourly,
    }).dropna(subset=["generation solar"])  # only keep hours with real actual data

    df = df.reset_index().rename(columns={"datetime": "time"}).sort_values("time")

    df['solar_lag_24h'] = df['generation solar'].shift(24)
    df['wind_lag_24h'] = df['generation wind onshore'].shift(24)
    df['solar_rolling_3h'] = df['generation solar'].shift(1).rolling(window=3).mean()
    df['hour'] = df['time'].dt.hour
    df['day_of_week'] = df['time'].dt.dayofweek

    return df.dropna(subset=['solar_lag_24h', 'solar_rolling_3h', 'wind_lag_24h'])


if __name__ == "__main__":
    df_clean = load_and_engineer()
    train, test = time_based_split(df_clean)
    print(f"Train: {len(train)} rows, Test: {len(test)} rows")
