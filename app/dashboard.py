"""Unified Streamlit dashboard for the stock-etl-pipeline ⇄ mlops-stock-platform system.

One screen that makes the two cooperating platforms read as a single product. Deliberately
lean — four panels, no clutter:

  1. Data freshness / health   — is the upstream ETL data recent and complete?
  2. Live prediction           — ask the serving API for a next-day direction.
  3. Champion vs challenger     — the promotion decision, side by side.
  4. Decay trend               — is the production model's edge holding up?

Per-feature drift tables, the EDA gallery, and raw OHLCV grids are intentionally left to the
native MLflow / charts surfaces. Everything here reads existing interfaces (the FastAPI
service, the MLflow tracking server, and the ETL MinIO store) — the dashboard owns no state.

Run from the host:  make dashboard   →   http://localhost:8501
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from platform_core.config import (
    CONTEXT_TICKERS,
    MLFLOW_TRACKING_URI,
    PRODUCTION_ALIAS,
    CHALLENGER_ALIAS,
    REGISTERED_MODEL_NAME,
    TICKERS,
)

API_URL = os.environ.get("API_URL", "http://localhost:8000")
MONITOR_EXPERIMENT = "production-monitoring"

st.set_page_config(page_title="Stock ML Platform", page_icon="📈", layout="wide")


# ----------------------------------------------------------------------------- data access
def _api_get(path: str, params: dict | None = None) -> tuple[dict | None, int]:
    """GET the serving API; return (json|None, status_code). 0 == unreachable."""
    url = f"{API_URL}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return json.loads(resp.read()), resp.status
    except urllib.error.HTTPError as exc:
        try:
            return json.loads(exc.read()), exc.code
        except Exception:
            return None, exc.code
    except Exception:
        return None, 0


@st.cache_data(ttl=60)
def etl_freshness() -> dict:
    """Latest date + coverage of our universe in the ETL store, with a CSV-cache fallback."""
    from platform_core import etl_source
    from platform_core.data import load_cached

    universe = TICKERS + CONTEXT_TICKERS
    frames = etl_source.load_from_etl(universe)
    source = "minio"
    if frames is None:  # ETL offline / incomplete → read the materialised cache mirror
        source = "local"
        try:
            frames = {t: load_cached(t) for t in universe}
        except FileNotFoundError:
            return {"source": "none", "tickers": 0, "latest": None, "staleness": None}
    latest = max(df.index.max() for df in frames.values()).date()
    return {
        "source": source,
        "tickers": len(frames),
        "expected": len(universe),
        "latest": latest,
        "staleness": (date.today() - latest).days,
    }


@st.cache_resource
def _mlflow_client():
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return MlflowClient()


@st.cache_data(ttl=30)
def model_versions() -> dict:
    """Production / challenger versions and their holdout metrics + shadow comparison."""
    client = _mlflow_client()

    def _slot(alias: str) -> dict | None:
        try:
            mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, alias)
        except Exception:
            return None
        m = client.get_run(mv.run_id).data.metrics
        return {
            "version": mv.version,
            "accuracy": m.get("holdout_accuracy"),
            "edge": m.get("holdout_edge_over_baseline"),
            "auc": m.get("holdout_auc"),
        }

    shadow = None
    try:
        import mlflow

        runs = mlflow.search_runs(
            experiment_names=[MONITOR_EXPERIMENT],
            filter_string="attributes.run_name = 'shadow-comparison'",
            order_by=["attributes.start_time DESC"],
            max_results=1,
        )
        if len(runs):
            r = runs.iloc[0]
            shadow = {
                "prod": r.get("metrics.shadow_prod_accuracy"),
                "chall": r.get("metrics.shadow_chall_accuracy"),
                "n": r.get("metrics.shadow_n_scored"),
            }
    except Exception:
        pass

    return {
        "production": _slot(PRODUCTION_ALIAS),
        "challenger": _slot(CHALLENGER_ALIAS),
        "shadow": shadow,
    }


@st.cache_data(ttl=30)
def decay_trend() -> pd.DataFrame:
    """Time series of recent edge-over-baseline from the production-monitoring experiment."""
    import mlflow

    runs = mlflow.search_runs(
        experiment_names=[MONITOR_EXPERIMENT],
        filter_string="attributes.run_name LIKE 'decay-check%'",
        order_by=["attributes.start_time ASC"],
    )
    if not len(runs):
        return pd.DataFrame()
    out = pd.DataFrame(
        {
            "time": pd.to_datetime(runs["start_time"]),
            "edge": runs.get("metrics.recent_edge_over_baseline"),
            "accuracy": runs.get("metrics.recent_accuracy"),
            "degraded": runs.get("metrics.degraded"),
        }
    ).dropna(subset=["edge"])
    return out


# ----------------------------------------------------------------------------- status bar
health, hcode = _api_get("/health")
info, _ = _api_get("/model-info")
fresh = etl_freshness()

st.title("📈 Stock ML Platform")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("API", "● healthy" if hcode == 200 else "○ down")
prod_v = (info or {}).get("production", {}).get("version", "—") if info else "—"
chall = (info or {}).get("challenger") if info else None
c2.metric("production", f"v{prod_v}")
c3.metric("challenger", f"v{chall['version']}" if chall else "—")
c4.metric("data source", fresh["source"])
if c5.button("↻ refresh"):
    st.cache_data.clear()
    st.rerun()

st.divider()

# ----------------------------------------------------------------------------- panels
left, right = st.columns(2)

# Panel 1 — Data freshness / health (ETL side)
with left:
    st.subheader("Data freshness")
    if fresh["source"] == "none":
        st.error("No data available — start the ETL stack or populate the CSV cache.")
    else:
        st.write(f"**Latest bar:** {fresh['latest']}  ·  **{fresh['staleness']} day(s)** old")
        st.write(f"**Coverage:** {fresh['tickers']} / {fresh['expected']} series")
        if fresh["source"] == "local":
            st.caption("⚠️ ETL store unreachable — showing the local cache mirror.")
        elif fresh["staleness"] is not None and fresh["staleness"] > 5:
            st.warning("Data is stale (>5 days) — check the ETL DAG.")
        else:
            st.caption("✅ Fresh and complete (matches the ETL freshness gate).")

# Panel 2 — Live prediction
with right:
    st.subheader("Live prediction")
    ticker = st.selectbox("Ticker", TICKERS, index=0)
    pred, pcode = _api_get("/predict", {"ticker": ticker})
    if pcode == 200 and pred:
        p_up = pred["prob_up_next_day"]
        st.metric(
            f"{ticker} — next day",
            "▲ UP" if pred["direction"] == "up" else "▼ DOWN",
            f"P(up) = {p_up:.1%}",
        )
        st.progress(float(p_up))
        st.caption(f"served by model v{pred['model_version']}")
    elif pcode == 503:
        st.info("No production model serving yet — run `make promote`.")
    elif pcode == 0:
        st.info(f"Serving API unreachable at {API_URL}.")
    else:
        st.warning(f"/predict returned {pcode}.")

st.divider()
left2, right2 = st.columns(2)

# Panel 3 — Champion vs challenger
with left2:
    st.subheader("Champion vs challenger")
    mv = model_versions()
    prod, ch = mv["production"], mv["challenger"]
    if prod is None:
        st.info("No production model registered yet — run `make promote`.")
    else:
        rows = []
        def _row(label, slot):
            return {
                "model": label,
                "version": f"v{slot['version']}" if slot else "—",
                "holdout_acc": f"{slot['accuracy']:.3f}" if slot and slot["accuracy"] else "—",
                "edge": f"{slot['edge']*100:+.1f}pp" if slot and slot["edge"] else "—",
            }
        rows.append(_row("production", prod))
        rows.append(_row("challenger", ch))
        st.table(pd.DataFrame(rows).set_index("model"))

        sh = mv["shadow"]
        if sh and sh.get("prod") is not None and sh.get("chall") is not None:
            verdict = "challenger leads" if sh["chall"] > sh["prod"] else "production leads"
            st.caption(
                f"Shadow (live traffic, n={int(sh['n'] or 0)}): "
                f"prod {sh['prod']:.3f} vs chall {sh['chall']:.3f} → **{verdict}**"
            )
        elif ch and prod and ch["edge"] and prod["edge"]:
            lead = "challenger" if ch["edge"] > prod["edge"] else "production"
            st.caption(f"On holdout edge, **{lead}** is ahead. Run `make promote` to gate a swap.")

# Panel 4 — Decay trend
with right2:
    st.subheader("Decay trend")
    trend = decay_trend()
    if trend.empty:
        st.info("No decay checks logged yet — run `make monitor`.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=trend["time"], y=trend["edge"] * 100,
                                 mode="lines+markers", name="edge"))
        fig.add_hline(y=0, line_dash="dot", line_color="gray")
        fig.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0),
                          yaxis_title="edge over baseline (pp)")
        st.plotly_chart(fig, width="stretch")
        last = trend.iloc[-1]
        if last.get("degraded"):
            st.warning(f"⚠️ Latest check flagged degraded (edge {last['edge']*100:+.1f}pp).")
        else:
            st.caption(f"Latest edge {last['edge']*100:+.1f}pp as of {last['time'].date()}.")

st.divider()
st.caption(
    f"ETL store: {fresh['source']}  ·  MLflow: {MLFLOW_TRACKING_URI}  ·  API: {API_URL}  ·  "
    f"refreshed {datetime.now():%H:%M:%S}"
)
