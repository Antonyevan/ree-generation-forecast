import json
import pandas as pd
import mlflow
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from features import build_live_features, time_based_split

FEATURES = ['hour', 'day_of_week', 'solar_lag_24h', 'wind_lag_24h', 'solar_rolling_3h']

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("solar-forecast-recent")

with mlflow.start_run():

    with open("data/recent_solar_data.json") as f:
        recent_data = json.load(f)

    df_clean = build_live_features(recent_data)
    print(f"Total usable rows after feature engineering: {len(df_clean)}")

    test_days = 60
    train, test = time_based_split(df_clean, test_days=test_days)
    print(f"Train: {len(train)} rows, Test: {len(test)} rows")

    # --- log params: the settings, so you can compare runs later ---
    mlflow.log_param("test_days", test_days)
    mlflow.log_param("features", FEATURES)
    mlflow.log_param("model_type", "GradientBoostingRegressor")
    mlflow.log_param("random_state", 42)
    mlflow.log_param("train_rows", len(train))
    mlflow.log_param("test_rows", len(test))

    model = GradientBoostingRegressor(random_state=42)
    model.fit(train[FEATURES], train['generation solar'])
    predictions = model.predict(test[FEATURES])

    baseline_mae = mean_absolute_error(test['generation solar'], test['forecast solar day ahead'])
    model_mae = mean_absolute_error(test['generation solar'], predictions)
    improvement = (baseline_mae - model_mae) / baseline_mae * 100

    print(f"\nREE's current forecast MAE: {baseline_mae:.1f} MW")
    print(f"Recent-data model MAE:      {model_mae:.1f} MW")
    print(f"Improvement: {improvement:.1f}%")

    # --- log metrics: the results, so you can track them over time ---
    mlflow.log_metric("baseline_mae", baseline_mae)
    mlflow.log_metric("model_mae", model_mae)
    mlflow.log_metric("improvement_pct", improvement)

    test = test.copy()
    test['model_error'] = abs(test['generation solar'] - predictions)
    test['baseline_error'] = abs(test['generation solar'] - test['forecast solar day ahead'])
    monthly = test.groupby(test['time'].dt.to_period('M'))[['model_error', 'baseline_error']].mean()
    print("\nMonthly comparison:")
    print(monthly)

    # --- log the model itself, so you have the actual artifact, not just numbers ---
    mlflow.sklearn.log_model(model, "model")
    
import joblib
joblib.dump(model, "recent_model.pkl")
print("Model saved to recent_model.pkl")