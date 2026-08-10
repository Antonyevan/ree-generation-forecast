import os
import json
import requests
from datetime import datetime, timedelta

TOKEN = os.environ["ESIOS_API_TOKEN"]
HEADERS = {"x-api-key": TOKEN, "Accept": "application/json"}

INDICATORS = {
    "actual_solar": 1295,
    "forecast_solar": 542,
    "actual_wind": 551,
}


def fetch_indicator(indicator_id, start_date):
    url = f"https://api.esios.ree.es/indicators/{indicator_id}"
    params = {
        "start_date": start_date.strftime("%Y-%m-%dT%H:%M"),
        # Deliberately omitting end_date: testing showed ESIOS's date-ranged
        # endpoint caps results ~1-2h behind real-time when end_date is set
        # to "now", but returns fully current data when only start_date is given.
    }
    resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()["indicator"]["values"]


def main():
    end_date = datetime.now()
    start_date = end_date - timedelta(hours=48)

    live_data = {"fetched_at": datetime.now().isoformat()}

    for label, indicator_id in INDICATORS.items():
        print(f"Fetching {label} (indicator {indicator_id})...")
        live_data[label] = fetch_indicator(indicator_id, start_date)

    os.makedirs("data", exist_ok=True)
    with open("data/live_solar_cache.json", "w") as f:
        json.dump(live_data, f)

    print(f"Saved live cache: {len(live_data['actual_solar'])} actual solar points, "
          f"{len(live_data['forecast_solar'])} forecast points, "
          f"{len(live_data['actual_wind'])} wind points.")


if __name__ == "__main__":
    main()
