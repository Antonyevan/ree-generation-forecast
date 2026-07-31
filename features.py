import pandas as pd

df = pd.read_csv("data/energy_dataset.csv")
df['time'] = pd.to_datetime(df['time'], utc=True)
df = df.sort_values('time').reset_index(drop=True)

df['solar_lag_24h'] = df['generation solar'].shift(24)

print(df[['time', 'generation solar', 'solar_lag_24h']].head(30))

df['hour'] = df['time'].dt.hour
df['solar_rolling_3h'] = df['generation solar'].rolling(window=3).mean()
print(df[['time', 'generation solar', 'hour', 'solar_rolling_3h']].head(10))

# Drop rows with NaN from our lag/rolling features (can't train on missing values)
df_clean = df.dropna(subset=['solar_lag_24h', 'solar_rolling_3h'])

# Use roughly the last 6 months as test data — everything before that is training
split_date = df_clean['time'].max() - pd.Timedelta(days=180)

train = df_clean[df_clean['time'] < split_date]
test = df_clean[df_clean['time'] >= split_date]

print(f"Train: {train['time'].min()} to {train['time'].max()} ({len(train)} rows)")
print(f"Test:  {test['time'].min()} to {test['time'].max()} ({len(test)} rows)")

df['wind_lag_24h'] = df['generation wind onshore'].shift(24)
df['day_of_week'] = df['time'].dt.dayofweek
df_clean = df.dropna(subset=['solar_lag_24h', 'solar_rolling_3h', 'wind_lag_24h'])

split_date = df_clean['time'].max() - pd.Timedelta(days=180)
train = df_clean[df_clean['time'] < split_date]
test = df_clean[df_clean['time'] >= split_date]

print(train[['time', 'hour', 'day_of_week', 'solar_lag_24h', 'wind_lag_24h', 'solar_rolling_3h']].head())