# ☀️ Spain Solar Generation Forecast

A machine learning system forecasting solar power generation in Spain, benchmarked against Red Eléctrica de España's (REE) official day-ahead forecast, with all evaluation metrics computed live and validated through rigorous review.

**🔴 Live dashboard:** [ree-solar-forecast.streamlit.app](https://ree-solar-forecast.streamlit.app)

---

## What's in the dashboard

Two tabs, each backed by a separately trained model:

- **🔴 Live (Today):** live data from REE's ESIOS API (auto-refreshed every 30 minutes), the recent-data model's live prediction against REE's own live forecast, a back-tested win rate over the full test period, and a daily error trend chart.
- **📅 Historical Replay:** the 2018 test period, with the historical model compared against REE's 2018 forecast on a day-by-day basis.

All comparison statistics are computed live from the underlying data on every page load, not hardcoded.

---

## Live data auto-refresh

Two scheduled GitHub Actions workflows maintain data currency without manual intervention:

- **`fetch_live_data.yml`** — runs every 30 minutes, fetching current actual and forecast data from ESIOS.
- **`refresh_recent_data.yml`** — runs weekly, refreshing the full year of training data so the back-tested window advances over time.

The dashboard's cache (`@st.cache_data(ttl=3600)`) rebuilds hourly, ensuring scheduled data refreshes are reflected in what visitors see.

---

## Engineering Principles Applied

**Metrics are computed dynamically, not hardcoded.** This has been validated twice: first when a fixed improvement percentage failed to hold up against a single noisy week of data, and second when the underlying model result itself required correction following a data leakage investigation (Project Journey, step 5). In both cases, the dashboard was designed to compute figures live from current data specifically so it cannot misrepresent results as they change — including self-correcting the moment the leakage fix was deployed.

**Rolling-window features require explicit shifting.** A generalizable finding from this project: any `.rolling()` calculation intended to reflect only past information must be preceded by `.shift(1)`. Without it, the current row is included by default — a subtle but critical source of data leakage in time-series feature engineering.

---

## Project Journey

1. **Historical baseline.** REE's live API was unavailable during the first development week; a well-established Kaggle dataset (2015–2018) was used to proceed without delay.
2. **Forecast weakness identified.** Root-cause analysis of REE's largest 2018 forecast error traced it to a localized storm system near Seville, undetected by the numeric weather feature but present in the categorical description.
3. **Initial model development.** A Gradient Boosting model was trained using lag, rolling-average, and time-based features, with a strictly time-respecting train/test split to prevent temporal leakage.
4. **Distribution shift identified on live data.** Applying the historical model to 2026 data revealed predictions plateauing near 5,000 MW — Spain's solar capacity has grown substantially since 2018, exceeding the model's training range. A second model was trained on recent live data in response.
5. **Data leakage identified and corrected.** An earlier version of this model reported a significant improvement over REE's forecast. During post-deployment review, a root-cause investigation into that result identified a data leakage defect in one engineered feature: `solar_rolling_3h`, a 3-hour rolling average, was computed without excluding the current observation — meaning the feature partially encoded the value the model was predicting.

   **Root cause:** `df['generation solar'].rolling(window=3).mean()` includes the current row by default.
   **Fix:** `df['generation solar'].shift(1).rolling(window=3).mean()`, ensuring the feature reflects only prior observations.
   **Verification:** confirmed by direct comparison of feature values against the target column across multiple rows, showing exact arithmetic inclusion of the current-hour value prior to the fix.
   **Impact:** with the defect corrected, the model's true performance is meaningfully lower than initially reported. Current, live figures are shown on the dashboard rather than restated here, since the recent model's dataset refreshes weekly.

   This defect was identified independently, post-publication, through structured validation of the result rather than an external report — and corrected transparently across the dashboard, this repository, and all public communications referencing the original figures.
6. **Pipeline automation.** Two scheduled GitHub Actions workflows maintain the live cache and training dataset without manual intervention, ensuring corrected figures remain current going forward.

---

## Data sources

- **Historical model training data:** ["Hourly energy demand generation and weather"](https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather) (Kaggle) — 4 years of Spanish generation, demand, and weather data.
- **Recent model training data + live dashboard data:** [REE's ESIOS API](https://www.esios.ree.es/) — a token-based API for real-time and historical grid data. Per ESIOS's terms of use, live data is fetched and cached periodically, not queried live per visitor.

---

## Repository structure

| File | Purpose |
|---|---|
| `pull_ree_generation_data.py` | Original REE REData API integration; retained as documentation of a multi-day outage encountered early in development |
| `eda.py` | Exploratory analysis of the 2015–2018 dataset; the Seville storm forecast-error investigation |
| `features.py` | Shared feature engineering, used by both models and the live dashboard; contains the data leakage fix documented above |
| `train_model.py` | Trains and evaluates the historical (2015–2018) model |
| `evaluate_errors.py` | Error analysis for the historical model |
| `pull_recent_data.py` | Pulls ~1 year of recent ESIOS data for retraining |
| `train_model_recent.py` | Trains and evaluates the recent-data model |
| `fetch_live_data.py` | Fetches and caches live ESIOS data (automated via GitHub Actions) |
| `dashboard.py` | Streamlit dashboard; all comparison statistics computed live |
| `.github/workflows/fetch_live_data.yml` | Scheduled workflow: live cache refresh, every 30 minutes |
| `.github/workflows/refresh_recent_data.yml` | Scheduled workflow: training data refresh, weekly |
| `requirements.txt` | Python dependencies |

---

## Running it locally

```bash
pip install -r requirements.txt

# Download the historical dataset from Kaggle and place CSVs in data/
# (see Data sources above)

# Request an ESIOS API token and set it:
export ESIOS_API_TOKEN="your-token-here"

python3 pull_recent_data.py
python3 fetch_live_data.py
streamlit run dashboard.py
```

To enable automated refresh: add `ESIOS_API_TOKEN` as a repository secret (Settings → Secrets and variables → Actions), and set Actions permissions to "Read and write" (Settings → Actions → General → Workflow permissions).

---

## Key Technical Decisions

- **Time-respecting train/test splits** throughout, preventing future information from leaking into training.
- **Benchmarked against REE's real, current forecast**, not a synthetic baseline.
- **Rolling-window features are explicitly shifted** prior to computation, following the correction documented above.
- **All dashboard metrics computed live**, ensuring accuracy is maintained automatically as underlying data or models change.
- **Live data cached via scheduled pipeline**, in compliance with ESIOS API terms of use and standard practice for third-party API rate limits.
- **A GenAI explanation layer was evaluated and deliberately excluded** to maintain focus on the core forecasting problem.

---

## Tech Stack

Python, pandas, scikit-learn (Gradient Boosting), Streamlit, GitHub Actions, REE ESIOS API, Kaggle dataset.