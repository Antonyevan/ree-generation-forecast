#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 17:37:57 2026

@author: antonyevanalosius
"""

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from features import load_and_engineer, time_based_split

df_clean = load_and_engineer()
train, test = time_based_split(df_clean)

features = ['hour', 'day_of_week', 'solar_lag_24h', 'wind_lag_24h', 'solar_rolling_3h']
model = GradientBoostingRegressor(random_state=42)
model.fit(train[features], train['generation solar'])

test = test.copy()
test['model_pred'] = model.predict(test[features])
test['baseline_error_signed'] = test['generation solar'] - test['forecast solar day ahead']

# Worst days in August
august = test[test['time'].dt.month == 8].copy()
worst_days = august.groupby(august['time'].dt.date)['baseline_error_signed'].mean().sort_values()
print(worst_days.head())

# Weather cross-reference for the worst day
weather = pd.read_csv("data/weather_features.csv")
weather['dt_iso'] = pd.to_datetime(weather['dt_iso'], utc=True)

midday = weather[(weather['dt_iso'].dt.date == pd.to_datetime('2018-08-27').date()) &
                  (weather['dt_iso'].dt.hour.between(10, 15))]
print(midday[['city_name', 'dt_iso', 'clouds_all', 'weather_description']])