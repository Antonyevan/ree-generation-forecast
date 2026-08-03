
from features import load_and_engineer, time_based_split
from sklearn.metrics import mean_absolute_error

df_clean = load_and_engineer()
train, test = time_based_split(df_clean)

print(f"Train: {len(train)} rows, Test: {len(test)} rows")


baseline_mae = mean_absolute_error(test['generation solar'], test['forecast solar day ahead'])
print(f"REE's baseline forecast MAE: {baseline_mae:.1f} MW")

from sklearn.ensemble import GradientBoostingRegressor

features = ['hour', 'day_of_week', 'solar_lag_24h', 'wind_lag_24h', 'solar_rolling_3h']

X_train = train[features]
y_train = train['generation solar']

X_test = test[features]
y_test = test['generation solar']

model = GradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

model_mae = mean_absolute_error(y_test, predictions)
print(f"Your model's MAE: {model_mae:.1f} MW")
print(f"REE's baseline MAE: {baseline_mae:.1f} MW")