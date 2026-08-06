# ☀️ Spain Solar Generation Forecast

Forecasting solar power generation in Spain, benchmarked directly against Red Eléctrica de España's (REE) own official day-ahead forecast — not a synthetic baseline.

**🔴 Live dashboard:** [ree-solar-forecast.streamlit.app](https://ree-solar-forecast.streamlit.app)

---

## Headline result

A Gradient Boosting model trained on the last ~1 year of REE's live grid data **wins a majority of days when back-tested against REE's own current day-ahead forecast**, with the exact win rate and daily error trend shown live on the dashboard (computed fresh from real data, not a fixed claim — see *Why the numbers aren't hardcoded* below).

A second, earlier model — trained on 2015–2018 historical data — outperforms REE's forecast **from that same period** by ~22% (96.6 MW vs 123.6 MW MAE), consistently across all 6 months of its test period.

---

## What's in the dashboard

Two tabs, each backed by a separately trained model:

- **🔴 Live (Today):** pulls real, current data from REE's ESIOS API (auto-refreshed every 30 minutes — see *Live data auto-refresh* below), shows the recent-data model's live prediction against REE's own live forecast, a full back-tested win-rate over the whole test period, a daily error chart across that period, and a "most recent week" table for transparency.
- **📅 Historical Replay:** lets you scrub through the 2018 test period and compare the historical model against REE's 2018 forecast, day by day.

---

## Live data auto-refresh

Two scheduled GitHub Actions workflows keep the dashboard's data genuinely current, with no manual script runs required:

- **`fetch_live_data.yml`** — runs **every 30 minutes**, fetches the latest actual/forecast data from ESIOS, and commits the updated `data/live_solar_cache.json` back to the repo. Powers the Live tab's "right now" chart and metrics.
- **`refresh_recent_data.yml`** — runs **every Sunday at 03:00 UTC**, re-pulls a full fresh year of ESIOS data and commits the updated `data/recent_solar_data.json`. This keeps the 60-day back-tested window genuinely rolling forward over time, rather than staying frozen at whichever date the data was first pulled.

The deployed dashboard simply reads whatever's currently committed to the repo — it never calls ESIOS directly itself. Its own cache (`@st.cache_data(ttl=3600)`) rebuilds at most once per hour, so these scheduled refreshes actually reach what visitors see, rather than being silently ignored by a cache that never expires.

Per ESIOS's terms of use, live data is fetched and cached periodically this way, rather than queried live by each dashboard visitor.

---

## Why the numbers aren't hardcoded

Early in this project, the dashboard displayed a fixed "36.3% improvement" claim. Digging into a single recent week showed the model actually *losing* most days during that stretch — which, on inspection, turned out to be a real, sustained rough patch (not a bug), while the model was winning strongly in the weeks before it. A single fixed percentage, or a single fixed week, can't represent that honestly.

The dashboard now computes its win-rate and "weakest stretch" callout **live, from whatever data currently exists** — so the numbers shown always reflect the real, current state of the model's performance, not a snapshot from whenever the dashboard was last edited.

---

## Project Journey

This project didn't arrive at its current form in a straight line, and the path is worth documenting honestly rather than hiding.

**1. Started with historical data.** REE's live API was down for the first week of this project (a genuine, multi-day outage — confirmed via isolated `curl` tests against multiple endpoints), so the first model was trained on a well-established Kaggle dataset: 4 years (2015–2018) of Spanish electricity generation, demand, and weather data.

**2. Found and investigated a real forecast weakness.** Exploratory analysis showed REE's own 2018 forecast systematically *overestimated* solar generation during daylight hours — a bias that held up across every season and every year in the dataset, not just a fluke. Digging into the single worst-forecast day (Aug 27, 2018), cross-referencing weather data revealed a "proximity thunderstorm" near Seville — a major solar-generating region — during peak hours. Notably, the *numeric* cloud-cover reading for that hour (20%) completely missed this; only the categorical weather description caught it.

**3. Built and validated a model against that historical benchmark.** A Gradient Boosting model, using lag features, rolling averages, and time-based features (with a strictly time-respecting train/test split — no random shuffling, to avoid leaking future data into training), beat REE's 2018 forecast by 22% on average, consistently across all 6 months of the test period.

**4. Connected to live 2026 data — and discovered distribution shift.** Once REE's live API access was restored (via a separate, token-based ESIOS API), the *historical* model's predictions on live 2026 data came back oddly flat and capped near ~5,000 MW — far below the ~30,000 MW peaks actually happening today. The cause: Spain's installed solar capacity has grown substantially since 2018, and the model had simply never seen values this large during training. This is a well-known, real production ML failure mode called **distribution shift**.

**5. Responded the way production systems actually do: retrained, didn't pretend to self-correct.** Rather than claim the model "adapts" (it doesn't, and claiming that would be dishonest), a second model was trained from scratch on a fresh year of live ESIOS data. This model correctly tracks today's real generation scale. Both models are kept in the repository — the original as validated evidence of the initial finding, the retrained one as the operational, current-facing result.

**6. Investigated an apparent contradiction rather than papering over it.** A 7-day snapshot of the recent model's performance showed it losing most days — seemingly at odds with its strong overall test-period average. Pulling the full month's daily breakdown resolved it: the model won the large majority of days, with a genuine, sustained rough patch only in the final stretch. Rather than hide this, the dashboard now surfaces the full back-tested trend and dynamically calls out the weakest period, so the honest picture — strong overall, with real limits — is always visible, not just a flattering average.

**7. Automated the live data pipeline.** Rather than requiring manual script runs to keep the Live tab current, a GitHub Actions workflow now fetches fresh data every 30 minutes and commits it back to the repo automatically — the same operational pattern a real production system would use.

This progression — build, validate, deploy, discover a real limitation, respond appropriately, automate — is arguably the most representative part of this project, more so than any single metric.

**Update:** REE's original REData API (`pull_ree_generation_data.py`) came back online during this project, after being down for its first ~10 days. This project has since standardized on the ESIOS API for live/recent data, so that script remains as a historical artifact of the initial outage investigation rather than part of the active pipeline.

---

## Data sources

- **Historical model training data:** ["Hourly energy demand generation and weather"](https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather) (Kaggle) — 4 years of Spanish generation, demand, and weather data.
- **Recent model training data + live dashboard data:** [REE's ESIOS API](https://www.esios.ree.es/) — a token-based API for real-time and historical grid data. Per ESIOS's terms of use, live data is fetched and cached periodically, not queried live per dashboard visitor.

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
| `fetch_live_data.py` | Fetches and caches today's live data from ESIOS, for the dashboard's Live tab (run automatically by GitHub Actions every 30 minutes) |
| `dashboard.py` | The Streamlit dashboard — Live and Historical Replay tabs |
| `.github/workflows/fetch_live_data.yml` | GitHub Actions workflow: refreshes live cache every 30 minutes |
| `.github/workflows/refresh_recent_data.yml` | GitHub Actions workflow: refreshes the full year of recent training data every Sunday |
| `requirements.txt` | Python dependencies for deployment |

---

## Running it locally

```bash
pip install -r requirements.txt

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

To set up the automated live-data refresh yourself: add `ESIOS_API_TOKEN` as a repository secret (Settings → Secrets and variables → Actions), and ensure Actions has "Read and write permissions" (Settings → Actions → General → Workflow permissions).

---

## Key technical decisions

- **Time-respecting train/test splits throughout** — never randomly shuffled, since that would leak future information into training in a way that wouldn't exist in a real deployment.
- **Benchmarked against REE's real forecast, not a naive baseline** — a genuinely harder, more meaningful bar than "predict yesterday's value."
- **Dashboard metrics are computed live from current data, not hardcoded** — including the win-rate and the "weakest stretch" callout — so the displayed numbers never go stale or misrepresent the current state of the model.
- **Live data is cached via a scheduled pipeline, not queried live per visitor** — per ESIOS's API terms of use, and matching how real production systems handle third-party API rate limits.
- **A GenAI explanation layer (LLM-generated plain-language summaries of forecast misses) was considered and deliberately scoped out**, to keep the project focused on the core forecasting problem rather than adding a component for its own sake.

---

## Tech stack

Python, pandas, scikit-learn (Gradient Boosting), Streamlit, GitHub Actions, REE ESIOS API, Kaggle dataset.