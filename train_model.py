import pandas as pd
import mlflow
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from features import load_and_engineer, time_based_split

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("solar-forecast-historical")

with mlflow.start_run():

    df_clean = load_and_engineer()
    train, test = time_based_split(df_clean)
    print(f"Train: {len(train)} rows, Test: {len(test)} rows")

    features = ['hour', 'day_of_week', 'solar_lag_24h', 'wind_lag_24h', 'solar_rolling_3h']
    X_train = train[features]
    y_train = train['generation solar']
    X_test = test[features]
    y_test = test['generation solar']

    # --- log params ---
    mlflow.log_param("features", features)
    mlflow.log_param("model_type", "GradientBoostingRegressor")
    mlflow.log_param("random_state", 42)
    mlflow.log_param("train_rows", len(train))
    mlflow.log_param("test_rows", len(test))

    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    baseline_mae = mean_absolute_error(y_test, test['forecast solar day ahead'])
    model_mae = mean_absolute_error(y_test, predictions)

    print(f"Model's MAE: {model_mae:.1f} MW")
    print(f"REE's baseline MAE: {baseline_mae:.1f} MW")

    # --- log metrics ---
    mlflow.log_metric("baseline_mae", baseline_mae)
    mlflow.log_metric("model_mae", model_mae)
    mlflow.log_metric("improvement_pct", (baseline_mae - model_mae) / baseline_mae * 100)

    test = test.copy()
    test['model_pred'] = predictions
    test['model_error'] = abs(test['generation solar'] - test['model_pred'])
    test['baseline_error'] = abs(test['generation solar'] - test['forecast solar day ahead'])
    monthly_comparison = test.groupby(test['time'].dt.to_period('M'))[['model_error', 'baseline_error']].mean()
    print(monthly_comparison)

    # --- log the model artifact ---
    mlflow.sklearn.log_model(model, "model")