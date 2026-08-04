# ☀️ Spain Solar Generation Forecast

Forecasting solar power generation in Spain, benchmarked directly against Red Eléctrica de España's (REE) own official day-ahead forecast — not a synthetic baseline.

**Live dashboard:** _[add your Streamlit Community Cloud URL here once deployed]_

---

## Headline result

A Gradient Boosting model trained on the last ~1 year of REE's live grid data **outperforms REE's own current day-ahead forecast by 36.3% on average** (638.5 MW vs 1,001.7 MW MAE, on a 60-day held-out test set).

A second, earlier model — trained on 2015–2018 historical data — outperforms REE's forecast **from that same period** by ~22% (96.6 MW vs 123.6 MW MAE). Both results are consistent across multiple months, not driven by a single lucky day (see *Project Journey* below).

---

## What's in the dashboard

Two tabs, each backed by a separately trained model:

- **🔴 Live (Today):** pulls real, current data from REE's ESIOS API, and shows the recent-data model's live prediction against REE's own live forecast.
- **📅 Historical Replay:** lets you scrub through the 2018 test period and compare the historical model against REE's 2018 forecast, day by day.

---

## Project Journey

This project didn't arrive at its current form in a straight line, and the path is worth documenting honestly rather than hiding.

**1. Started with historical data.** REE's live API was down for the first week of this project (a genuine, multi-day outage — confirmed via isolated `curl` tests against multiple endpoints), so the first model was trained on a well-established Kaggle dataset: 4 years (2015–2018) of Spanish electricity generation, demand, and weather data.

**2. Found and investigated a real forecast weakness.** Exploratory analysis showed REE's own 2018 forecast systematically *overestimated* solar generation during daylight hours — a bias that held up across every season and every year in the dataset, not just a fluke. Digging into the single worst-forecast day (Aug 27, 2018), cross-referencing weather data revealed a "proximity thunderstorm" near Seville — a major solar-generating region — during peak hours. Notably, the *numeric* cloud-cover reading for that hour (20%) completely missed this; only the categorical weather description caught it.

**3. Built and validated a model against that historical benchmark.** A Gradient Boosting model, using lag features, rolling averages, and time-based features (with a strictly time-respecting train/test split — no random shuffling, to avoid leaking future data into training), beat REE's 2018 forecast by 22% on average, consistently across all 6 months of the test period.

**4. Connected to live 2026 data — and discovered distribution shift.** Once REE's live API access was restored (via a separate, token-based ESIOS API), the *historical* model's predictions on live 2026 data came back oddly flat and capped near ~5,000 MW — far below the ~30,000 MW peaks actually happening today. The cause: Spain's installed solar capacity has grown substantially since 2018, and the model had simply never seen values this large during training. This is a well-known, real production ML failure mode called **distribution shift**.

**5. Responded the way production systems actually do: retrained, didn't pretend to self-correct.** Rather than claim the model "adapts" (it doesn't, and claiming that would be dishonest), a second model was trained from scratch on a fresh year of live ESIOS data. This model correctly tracks today's real generation scale, and outperforms REE's *current* forecast by 36.3%. Both models are kept in the repository — the original as validated evidence of the initial finding, the retrained one as the operational, current-facing result.

This progression — build, validate, deploy, discover a real limitation, respond appropriately — is arguably the most representative part of this project, more so than any single metric.

---

## Data sources

- **Historical model training data:** ["Hourly energy demand generation and weather"](https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather) (Kaggle) — 4 years of Spanish generation, demand, and weather data.
- **Recent model training data + live dashboard data:** [REE's ESIOS API](https://www.esios.ree.es/) — a token-based API for real-time and historical grid data. Per ESIOS's terms of use, live data is fetched and cached periodically (`fetch_live_data.py`), not queried live per dashboard visitor.

---

## Repository structure

| File | Purpose |
|---|---|
| `pull_ree_generation_data.py` | Original script to pull data from REE's public REData API (affected by a real, week-long outage during this project — kept as-is, documents the issue and the retry logic built to handle it) |
| `eda.py` | Exploratory analysis on the 2015–2018 historical dataset; the thunderstorm/forecast-bias investigation |
| `features.py` | Shared feature-engineering logic — used by both the historical and recent models, and by the live dashboard |
| `train_model.py` | Trains and evaluates the original model on 2015–2018 historical data |
| `evaluate_errors.py` | Deep-dive error investigation for the historical model (worst-day analysis, weather cross-reference) |
| `pull_recent_data.py` | Pulls ~1 year of recent data from ESIOS for retraining |
| `train_model_recent.py` | Trains and evaluates the recent-data model against REE's current forecast |
| `fetch_live_data.py` | Fetches and caches today's live data from ESIOS, for the dashboard's Live tab |
| `dashboard.py` | The Streamlit dashboard — Live and Historical Replay tabs |

---

## Running it locally

```bash
pip install streamlit scikit-learn pandas requests

# Get the historical dataset from Kaggle and place CSVs in data/
# (see Data sources above)

# Get an ESIOS API token (free, requested by email) and set it:
export ESIOS_API_TOKEN="your-token-here"

# Pull data for the recent model and live dashboard
python3 pull_recent_data.py
python3 fetch_live_data.py

# Run the dashboard
streamlit run dashboard.py
```

---

## Key technical decisions

- **Time-respecting train/test splits throughout** — never randomly shuffled, since that would leak future information into training in a way that wouldn't exist in a real deployment.
- **Benchmarked against REE's real forecast, not a naive baseline** — a genuinely harder, more meaningful bar than "predict yesterday's value."
- **A GenAI explanation layer (LLM-generated plain-language summaries of forecast misses) was considered and deliberately scoped out**, to keep the project focused on the core forecasting problem rather than adding a component for its own sake.
- **Live data is cached, not queried live per visitor**, per ESIOS's API terms of use.

---

## Tech stack

Python, pandas, scikit-learn (Gradient Boosting), Streamlit, REE ESIOS API, Kaggle dataset.