"""ops_monitor.py — internal telemetry console. Run: streamlit run ops_monitor.py"""

import json
import sqlite3
from contextlib import closing
from pathlib import Path

import streamlit as st

from sync_repo import sync

st.set_page_config(page_title="ops_monitor", layout="centered")

st.markdown("""
<style>
  .stApp { background: #050a08; }
  #MainMenu, footer, header { visibility: hidden; }
  * { font-family: 'JetBrains Mono', 'SF Mono', Consolas, monospace !important; }

  .console-head {
    display:flex; align-items:baseline; justify-content:space-between;
    border-bottom:1px solid #163025; padding-bottom:14px; margin-bottom:22px;
  }
  .console-head .id { color:#39ff88; font-size:13px; letter-spacing:0.04em; }
  .console-head .id b { color:#8effb8; }
  .console-head .clock { color:#3e5a4d; font-size:12px; }

  div[data-testid="stMetric"] {
    background:#0a1410; border:1px solid #163025; border-radius:3px; padding:14px 16px;
  }
  div[data-testid="stMetricLabel"] { color:#5c8a71 !important; font-size:11px !important; }
  div[data-testid="stMetricValue"] { color:#c9f5da !important; font-size:26px !important; }

  .readout {
    border:1px solid #163025; border-radius:3px; padding:20px; margin-bottom:18px;
    background:radial-gradient(circle at 15% 20%, #0d1f16 0%, #05100b 70%);
  }
  .readout .label { color:#5c8a71; font-size:11px; letter-spacing:0.05em; margin-bottom:6px; }
  .readout .big { font-size:44px; line-height:1; }
  .readout .big.ok { color:#39ff88; text-shadow:0 0 14px #39ff8855; }
  .readout .big.warn { color:#ffb02e; text-shadow:0 0 14px #ffb02e55; }
  .readout .sub { color:#3e5a4d; font-size:12px; margin-top:6px; }

  .logline { font-size:12.5px; color:#8fae9c; padding:5px 0; border-bottom:1px solid #101f18; }
  .logline .tag { color:#39ff88; }
  .logline .warn-tag { color:#ffb02e; }
  .logline .num { color:#c9f5da; }

  .section-h { color:#5c8a71; font-size:11px; letter-spacing:0.05em; margin:22px 0 8px; }
  div.stButton > button {
    background:#0a1410; border:1px solid #235a3d; color:#39ff88; border-radius:3px;
  }
  div.stButton > button:hover { border-color:#39ff88; color:#8effb8; }
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="console-head"><span class="id">OPS_MONITOR // <b>ree-generation-forecast</b></span>'
    '<span class="clock">local read-only console</span></div>',
    unsafe_allow_html=True,
)

if st.button("↻ sync"):
    with st.spinner("pulling..."):
        ok = sync()
    st.write("synced." if ok else "sync failed — showing last local snapshot.")

metrics_path = Path("latest_metrics.json")

if not metrics_path.exists():
    st.error("no latest_metrics.json — run compute_latest_metrics.py or wait for the next scheduled refresh.")
else:
    with open(metrics_path) as f:
        m = json.load(f)

    stale = m["hours_since_live_fetch"] > 2
    win_class = "warn" if stale else "ok"
    st.markdown(f"""
    <div class="readout">
      <div class="label">LIVE DATA AGE</div>
      <div class="big {win_class}">{m['hours_since_live_fetch']:.1f}h</div>
      <div class="sub">{'⚠ stale — expected refresh every 30min, check fetch_live_data workflow' if stale else 'nominal'}</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("MODEL MAE", f"{m['model_mae']:.0f} MW")
    c2.metric("REE MAE", f"{m['baseline_mae']:.0f} MW")
    c3.metric("WIN RATE", f"{m['win_rate_pct']:.1f}%")
    st.metric("Δ vs REE", f"{m['improvement_pct']:.1f}%")

    st.markdown('<div class="section-h">ANOMALY LOG</div>', unsafe_allow_html=True)
    if m["anomaly_count"] == 0:
        st.markdown('<div class="logline"><span class="tag">[clear]</span> no flagged days in current window</div>', unsafe_allow_html=True)
    else:
        for a in m["anomalies"]:
            st.markdown(
                f'<div class="logline"><span class="warn-tag">[flag]</span> {a["date"]} '
                f'— error <span class="num">{a["model_error"]:.0f} MW</span></div>',
                unsafe_allow_html=True,
            )

    st.caption(f"computed {m['computed_at']} · live fetch {m['live_data_fetched_at']}")

# --- MLflow run trace -------------------------------------------------------
st.markdown('<div class="section-h">MLFLOW RUN TRACE</div>', unsafe_allow_html=True)

db_path = Path("mlflow.db")
if not db_path.exists():
    st.markdown('<div class="logline">[--] mlflow.db not found locally</div>', unsafe_allow_html=True)
else:
    try:
        with closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)) as conn:
            rows = conn.execute("""
                SELECT r.run_uuid, e.name, r.start_time
                FROM runs r JOIN experiments e ON e.experiment_id = r.experiment_id
                WHERE r.lifecycle_stage = 'active'
                ORDER BY r.start_time DESC LIMIT 5
            """).fetchall()

        if not rows:
            st.markdown('<div class="logline">[--] no runs logged</div>', unsafe_allow_html=True)
        for run_id, exp_name, start_ms in rows:
            with closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)) as conn:
                metrics = dict(conn.execute(
                    "SELECT key, value FROM latest_metrics WHERE run_uuid = ? AND is_nan = 0",
                    (run_id,),
                ))
            mae = metrics.get("model_mae")
            imp = metrics.get("improvement_pct")
            mae_str = f"{mae:.0f} MW" if mae is not None else "—"
            imp_str = f"{imp:+.1f}%" if imp is not None else "—"
            st.markdown(
                f'<div class="logline"><span class="tag">[{exp_name}]</span> '
                f'mae=<span class="num">{mae_str}</span> Δ=<span class="num">{imp_str}</span></div>',
                unsafe_allow_html=True,
            )
    except sqlite3.OperationalError as exc:
        st.markdown(f'<div class="logline">[--] could not read mlflow.db: {exc}</div>', unsafe_allow_html=True)