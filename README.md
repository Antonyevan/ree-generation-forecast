# ☀️ Spain Solar Generation Forecast

[![Run Tests](https://github.com/Antonyevan/ree-generation-forecast/actions/workflows/run_tests.yml/badge.svg)](https://github.com/Antonyevan/ree-generation-forecast/actions/workflows/run_tests.yml)

A machine learning system forecasting solar power generation in Spain, benchmarked against Red Eléctrica de España's (REE) official day-ahead forecast, with all evaluation metrics computed live and validated through rigorous review.

**🔴 Live dashboard:** [ree-solar-forecast.streamlit.app](https://ree-solar-forecast.streamlit.app)

---

## What's in the dashboard

Two tabs, each backed by a separately trained model:

- **🔴 Live (Today):** live data from REE's ESIOS API (auto-refreshed every 30 minutes), the recent-data model's live prediction against REE's own live forecast, a back-tested win rate over the full test period, statistical anomaly detection on daily error, and a daily error trend chart.
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

## Testing

The `solar_rolling_3h` leakage fix (see Project Journey, step 5) is locked in by automated regression tests, not just a one-time manual fix:

- **Two targeted tests** verify the exact `.shift(1)` fix holds in both the historical (`load_and_engineer`) and live (`build_live_features`) code paths — the live path had the identical, previously undetected leak, found and fixed while adding this test coverage.
- **One general test** verifies that no feature depends on future data at all, by comparing feature output on identical datasets where only future rows differ — designed to catch a similar leak in any feature, not just the one already found.
- **Two tests for anomaly detection** (`tests/test_anomaly_detection.py`) verify the statistical threshold correctly flags true outliers and correctly ignores normal variation — including a fix to the test itself after an initial small-sample test case incorrectly failed due to a genuine, documented limitation of the mean+2σ method on very small datasets (see Anomaly Detection, below).

All tests run automatically via GitHub Actions on every push to `main` (see badge above), so a future change reintroducing this class of bug would fail visibly before being merged, not silently ship.

Run locally:
```bash
pip install pytest
pytest tests/ -v
```

---

## Experiment tracking

Both training scripts (`train_model.py` for the historical model, `train_model_recent.py` for the recent-data model) log every run via **MLflow** — parameters, evaluation metrics, and the trained model artifact — to a local SQLite-backed store, so results are never lost to a scrolled-past terminal window.

Each run records:
- **Parameters:** feature list, model type, random state, train/test row counts, test window size
- **Metrics:** REE baseline MAE, model MAE, and improvement percentage
- **Artifact:** the trained sklearn model itself

This matters specifically because the recent model's win rate against REE varies by test window (documented as a real, honest range in the Project Journey) — MLflow keeps an exact, queryable, timestamped record of every retrain and its result, rather than relying on memory or a single restated number.

MLflow tracking is a local, personal tool for development — not part of the deployed pipeline — and is intentionally excluded from version control (`mlruns/`, `mlflow.db` are gitignored).

To view it locally:
```bash
pip install mlflow
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```
Then open `http://127.0.0.1:5001`.

---

## Containerization

The project ships as two separate Docker containers, reflecting their different roles:

- **`Dockerfile`** — containerizes the live dashboard (`dashboard.py`). A long-running service, exposing port 8501.
- **`Dockerfile.train`** — containerizes the training pipeline (`train_model.py` / `train_model_recent.py`). A one-shot job: runs, logs to MLflow, exits.

This split follows standard practice: a persistent service and a run-once job have different operational needs and don't belong in the same container.

Build and run the dashboard:
```bash
docker build -t solar-dashboard .
docker run -p 8501:8501 solar-dashboard
```

Build and run training (mount `mlflow.db` so results persist outside the container):
```bash
docker build -f Dockerfile.train -t solar-training .
docker run -v $(pwd)/mlflow.db:/app/mlflow.db solar-training
```

---

## Anomaly detection

A standalone module, `anomaly_detection.py`, flags days where model error is statistically unusual (mean + 2 standard deviations over the test period), surfaced directly on the Live tab.

This exists for a specific reason beyond flagging bad days: it turns individual dates into starting points for investigation. A flagged date can be cross-referenced against external context (weather, grid events, demand anomalies) to build an evidence-based case for new features — the same process that originally identified the Seville storm as an explanation for REE's worst 2018 forecast day, now made repeatable rather than a one-off manual investigation.

Kept as a pure function, independent of Streamlit, specifically so it's directly testable — `tests/test_anomaly_detection.py` covers both a true-outlier case and a no-outlier case. Worth noting as a real, documented limitation: the mean+2σ method needs a reasonably sized sample to behave as expected — an initial test using only 5 data points failed because a single large outlier in a tiny sample distorts its own detection threshold. The test was corrected to use a realistic sample size rather than changing the underlying method, since the function's behavior was mathematically correct.

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
7. **Automated regression testing.** Following the leakage investigation, a `pytest` suite was added covering both the historical and live feature-engineering paths, plus a general future-data-independence check. Wired into GitHub Actions to run on every push, so this class of bug cannot silently reappear (see Testing, above).
8. **Experiment tracking added.** Both training scripts now log parameters, metrics, and model artifacts via MLflow on every run, replacing print-statement-only output with a permanent, queryable training history (see Experiment tracking, above).
9. **Containerization.** Both the dashboard and training pipeline were containerized separately via Docker, verified by running each in isolation and confirming identical results — the dashboard rendering correctly at `localhost:8501`, and a training run's MLflow-logged metrics matching a native run exactly, proving results persist correctly outside the container via a mounted volume.
10. **Separated training from serving.** The dashboard was found to be silently retraining a fresh model on every hourly cache refresh, completely disconnected from the MLflow-tracked training scripts — meaning the live-serving model was never actually tracked, and local training runs had no effect on the deployed site. Fixed by having `train_model_recent.py` save the trained model to a file, which `dashboard.py` now loads directly instead of training internally — the standard production pattern of separating training from serving.
11. **Anomaly detection added.** A statistical anomaly detector was added to flag unusually high-error days on the Live tab, built as a standalone, independently tested module rather than inline dashboard logic (see Anomaly Detection, above). Also corrected a labeling inaccuracy discovered in the process: the "Most recent week" section could lag the actual current date by up to a week, due to the weekly training-data refresh cadence — relabeled to accurately describe what it shows.

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
| `anomaly_detection.py` | Standalone, testable statistical anomaly detection on daily model error |
| `train_model.py` | Trains and evaluates the historical (2015–2018) model; logs to MLflow |
| `evaluate_errors.py` | Error analysis for the historical model |
| `pull_recent_data.py` | Pulls ~1 year of recent ESIOS data for retraining |
| `train_model_recent.py` | Trains and evaluates the recent-data model; logs to MLflow; saves the trained model for the dashboard to load |
| `fetch_live_data.py` | Fetches and caches live ESIOS data (automated via GitHub Actions) |
| `dashboard.py` | Streamlit dashboard; loads a pre-trained model rather than training itself; all comparison statistics computed live |
| `tests/test_features.py` | Regression tests for the feature engineering leakage fix |
| `tests/test_anomaly_detection.py` | Tests for the anomaly detection module |
| `Dockerfile` | Containerizes the dashboard as a persistent service |
| `Dockerfile.train` | Containerizes the training pipeline as a one-shot job |
| `.dockerignore` | Excludes local-only files (MLflow tracking data, tests, git metadata) from the build context |
| `.github/workflows/fetch_live_data.yml` | Scheduled workflow: live cache refresh, every 30 minutes |
| `.github/workflows/refresh_recent_data.yml` | Scheduled workflow: training data refresh, weekly |
| `.github/workflows/run_tests.yml` | Scheduled workflow: runs pytest on every push to `main` |
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
python3 train_model_recent.py
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
- **Feature engineering is covered by automated regression tests**, run on every push via GitHub Actions, ensuring the leakage fix (and future feature changes) can't silently break correctness.
- **Every training run is logged via MLflow**, giving an exact, timestamped record of parameters and results across retrains rather than relying on memory or a single restated figure.
- **Dashboard and training are containerized separately via Docker**, reflecting their different operational shapes (persistent service vs. one-shot job) and giving anyone an environment-identical way to reproduce results.
- **Training is separated from serving**: the dashboard loads a deliberately trained model file rather than retraining itself, avoiding an untracked, disconnected model silently running in production.
- **Anomaly detection is a standalone, tested module**, not inline dashboard logic — kept independent of Streamlit specifically so its statistical behavior can be verified directly.

---

## Tech Stack

Python, pandas, scikit-learn (Gradient Boosting), Streamlit, GitHub Actions, MLflow, Docker, REE ESIOS API, Kaggle dataset.