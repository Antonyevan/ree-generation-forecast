import pandas as pd


def load_and_engineer(path="data/energy_dataset.csv"):
    """Load raw energy data and build model-ready features."""
    df = pd.read_csv(path)
    df['time'] = pd.to_datetime(df['time'], utc=True)
    df = df.sort_values('time').reset_index(drop=True)

    df['solar_lag_24h'] = df['generation solar'].shift(24)
    df['wind_lag_24h'] = df['generation wind onshore'].shift(24)
    df['solar_rolling_3h'] = df['generation solar'].rolling(window=3).mean()
    df['hour'] = df['time'].dt.hour
    df['day_of_week'] = df['time'].dt.dayofweek

    return df.dropna(subset=['solar_lag_24h', 'solar_rolling_3h', 'wind_lag_24h'])


def time_based_split(df, test_days=180):
    """Split chronologically — train on the past, test on the most recent window."""
    split_date = df['time'].max() - pd.Timedelta(days=test_days)
    train = df[df['time'] < split_date]
    test = df[df['time'] >= split_date]
    return train, test


if __name__ == "__main__":
    # Lets you still run this file directly to sanity-check it, like before
    df_clean = load_and_engineer()
    train, test = time_based_split(df_clean)
    print(f"Train: {len(train)} rows, Test: {len(test)} rows")