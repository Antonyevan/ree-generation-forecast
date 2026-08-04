import json
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from features import build_live_features, time_based_split

FEATURES = ['hour', 'day_of_week', 'solar_lag_24h', 'wind_lag_24h', 'solar_rolling_3h']

with open("data/recent_solar_data.json") as f:
    recent_data = json.load(f)

df_clean = build_live_features(recent_data)
print(f"Total usable rows after feature engineering: {len(df_clean)}")

train, test = time_based_split(df_clean, test_days=60)
print(f"Train: {len(train)} rows, Test: {len(test)} rows")

model = GradientBoostingRegressor(random_state=42)
model.fit(train[FEATURES], train['generation solar'])

predictions = model.predict(test[FEATURES])

baseline_mae = mean_absolute_error(test['generation solar'], test['forecast solar day ahead'])
model_mae = mean_absolute_error(test['generation solar'], predictions)

print(f"\nREE's current forecast MAE: {baseline_mae:.1f} MW")
print(f"Recent-data model MAE:      {model_mae:.1f} MW")

improvement = (baseline_mae - model_mae) / baseline_mae * 100
print(f"Improvement: {improvement:.1f}%")

test = test.copy()
test['model_error'] = abs(test['generation solar'] - predictions)
test['baseline_error'] = abs(test['generation solar'] - test['forecast solar day ahead'])

monthly = test.groupby(test['time'].dt.to_period('M'))[['model_error', 'baseline_error']].mean()
print("\nMonthly comparison:")
print(monthly)
