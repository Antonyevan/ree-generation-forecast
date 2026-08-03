## Running the dashboard
```bash
pip install streamlit scikit-learn pandas
streamlit run dashboard.py
```

## Note on scope
A GenAI explanation layer (translating forecast errors into plain-language 
summaries) was considered but deliberately scoped out to keep the project 
focused on the core forecasting problem.

## Note on live data
REE's public REData API has been unavailable since July 24, 2026. The dashboard 
currently replays historical test-period data. REE's ESIOS API (a separate, 
token-based service) is a planned integration path once available.