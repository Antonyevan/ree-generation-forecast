#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 13:40:46 2026

@author: antonyevanalosius
"""

"""
Pull ~1 year of solar, wind, and forecast data from REE's ESIOS API,
in chunks, with retries — same defensive pattern as pull_ree_generation_data.py.

Saved in the same JSON shape as data/live_solar_cache.json (actual_solar,
forecast_solar, actual_wind), so features.py's build_live_features() can be
reused directly to build training features from this data too.
"""

import os
import json
import time
import requests
from datetime import datetime, timedelta

TOKEN = os.environ["ESIOS_API_TOKEN"]
HEADERS = {"x-api-key": TOKEN, "Accept": "application/json"}

INDICATORS = {
    "actual_solar": 1295,
    "forecast_solar": 542,
    "actual_wind": 551,
}

CHUNK_DAYS = 30
DAYS_BACK = 365


def fetch_chunk(indicator_id, start_date, end_date, max_attempts=3):
    url = f"https://api.esios.ree.es/indicators/{indicator_id}"
    params = {
        "start_date": start_date.strftime("%Y-%m-%dT%H:%M"),
        "end_date": end_date.strftime("%Y-%m-%dT%H:%M"),
    }
    for attempt in range(1, max_attempts + 1):
        resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
        if resp.status_code == 200:
            return resp.json()["indicator"]["values"]
        wait = 5 * attempt
        print(f"    Attempt {attempt} failed ({resp.status_code}), retrying in {wait}s...")
        time.sleep(wait)
    print(f"    Giving up on chunk {start_date.date()} to {end_date.date()}")
    return []


def fetch_indicator_full_year(label, indicator_id):
    end = datetime.now()
    start = end - timedelta(days=DAYS_BACK)

    all_values = []
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(days=CHUNK_DAYS), end)
        print(f"  Fetching {label}: {chunk_start.date()} to {chunk_end.date()}...")
        values = fetch_chunk(indicator_id, chunk_start, chunk_end)
        all_values.extend(values)
        chunk_start = chunk_end
        time.sleep(1)  # be polite to the API between chunks

    return all_values


def main():
    recent_data = {"fetched_at": datetime.now().isoformat()}

    for label, indicator_id in INDICATORS.items():
        print(f"Fetching {label} (indicator {indicator_id}) — last {DAYS_BACK} days...")
        recent_data[label] = fetch_indicator_full_year(label, indicator_id)
        print(f"  Got {len(recent_data[label])} points for {label}.\n")

    os.makedirs("data", exist_ok=True)
    with open("data/recent_solar_data.json", "w") as f:
        json.dump(recent_data, f)

    print("Saved data/recent_solar_data.json")
    print(f"  actual_solar: {len(recent_data['actual_solar'])} points")
    print(f"  forecast_solar: {len(recent_data['forecast_solar'])} points")
    print(f"  actual_wind: {len(recent_data['actual_wind'])} points")


if __name__ == "__main__":
    main()