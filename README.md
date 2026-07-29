# REE Generation Forecast

Forecasting renewable energy generation in Spain.

## Data sources
- **Training data:** Kaggle's "Hourly energy demand generation and weather" 
  dataset (4 years of historical Spanish generation, demand, and weather data)
- **Live data (for deployment):** REE's public REData API 
  (`pull_ree_generation_data.py`) — currently unavailable due to an ongoing 
  service issue on REE's end as of July 2026; will be used once the API 
  is back for the live dashboard stage.
