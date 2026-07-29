
"""
Pull historical electricity generation-structure data for Spain (peninsula)
from REE's free public REData API — no signup, no API key required.
 
Docs: https://www.ree.es/en/datos/apidata
Endpoint category: generacion / estructura-generacion
 
This script pulls ~6 months of hourly data in monthly chunks (the API can
be picky about very large date ranges at hourly resolution), flattens the
JSON response into a tidy DataFrame, and saves it as a CSV.
"""
 
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
 
BASE_URL = "https://apidatos.ree.es/en/datos/generacion/estructura-generacion"
 
# Peninsula Spain (the standard scope for this kind of analysis)
GEO_TRUNC = "electric_system"
GEO_LIMIT = "peninsular"
GEO_IDS = "8741"
 
 
def fetch_month(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Fetch one chunk of hourly generation data and flatten it into rows."""
    params = {
        "start_date": start_date.strftime("%Y-%m-%dT%H:%M"),
        "end_date": end_date.strftime("%Y-%m-%dT%H:%M"),
        "time_trunc": "hour",
        "geo_trunc": GEO_TRUNC,
        "geo_limit": GEO_LIMIT,
        "geo_ids": GEO_IDS,
    }
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        resp = requests.get(BASE_URL, params=params, timeout=30)
        if resp.status_code == 200:
            break
        try:
            detail = resp.json()
        except ValueError:
            detail = resp.text
        # REE occasionally reports "currently unavailable" under load — worth retrying
        transient = isinstance(detail, dict) and "unavailable" in str(detail).lower()
        if transient and attempt < max_attempts:
            wait = 5 * attempt
            print(f"  Temporary REE error, retrying in {wait}s (attempt {attempt}/{max_attempts})...")
            time.sleep(wait)
            continue
        raise requests.exceptions.RequestException(f"{resp.status_code} — {detail}")
    data = resp.json()
 
    rows = []
    for series in data.get("included", []):
        source_type = series["attributes"]["title"]  # e.g. "Solar photovoltaic", "Wind"
        for point in series["attributes"]["values"]:
            rows.append(
                {
                    "datetime": point["datetime"],
                    "source": source_type,
                    "value_mw": point["value"],
                    "percentage": point.get("percentage"),
                }
            )
    return pd.DataFrame(rows)
 
 
def main():
    end = datetime.today()
    start = end - timedelta(days=180)  # ~6 months of history
 
    all_chunks = []
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(days=7), end)
        print(f"Fetching {chunk_start.date()} to {chunk_end.date()} ...")
        try:
            df_chunk = fetch_month(chunk_start, chunk_end)
            all_chunks.append(df_chunk)
        except requests.exceptions.RequestException as e:
            print(f"  Failed for this chunk: {e}")
        chunk_start = chunk_end
        time.sleep(1.5)  # be polite to REE's API between chunks
 
    if not all_chunks:
        print("No data retrieved — check your internet connection or the API status.")
        return
 
    full_df = pd.concat(all_chunks, ignore_index=True)
    full_df["datetime"] = pd.to_datetime(full_df["datetime"], utc=True)
    full_df = full_df.sort_values(["source", "datetime"]).drop_duplicates()
 
    out_path = "data/ree_generation_structure.csv"
    full_df.to_csv(out_path, index=False)
    print(f"\nSaved {len(full_df)} rows to {out_path}")
    print(f"Sources found: {sorted(full_df['source'].unique())}")
 
 
if __name__ == "__main__":
    main()
 