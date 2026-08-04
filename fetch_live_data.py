#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 12:58:37 2026

@author: antonyevanalosius
"""

import os
import json
import requests
from datetime import datetime

TOKEN = os.environ["ESIOS_API_TOKEN"]
HEADERS = {"x-api-key": TOKEN, "Accept": "application/json"}

INDICATORS = {
    "actual_solar": 1295,   
    "forecast_solar": 542,  
}


def fetch_indicator(indicator_id):
    url = f"https://api.esios.ree.es/indicators/{indicator_id}"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json()["indicator"]["values"]


def main():
    live_data = {"fetched_at": datetime.now().isoformat()}

    for label, indicator_id in INDICATORS.items():
        print(f"Fetching {label} (indicator {indicator_id})...")
        live_data[label] = fetch_indicator(indicator_id)

    with open("data/live_solar_cache.json", "w") as f:
        json.dump(live_data, f)

    print(f"Saved live cache with {len(live_data['actual_solar'])} actual points "
          f"and {len(live_data['forecast_solar'])} forecast points.")


if __name__ == "__main__":
    main()