"""Serving API: always serves whatever version @production points to.

The model manager re-checks the alias (with a small TTL) and hot-reloads
when promote.py moves the pointer — deploys and promotions are decoupled.

Run locally:  uv run uvicorn app.main:app --port 8000
"""

from __future__ import annotations

import logging
import time

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow import MlflowClient

from platform_core.config import (
    MLFLOW_TRACKING_URI,
    PRODUCTION_ALIAS,
    REGISTERED_MODEL_NAME,
    TICKERS,
)
from platform_core.features import FEATURES, TECH_FEATURES, build_dataset

log = logging.getLogger("uvicorn.error")

ALIAS_CHECK_TTL_SECONDS = 30.0


class ModelManager:
    """Caches the production model; swaps it when the alias moves."""

    def __init__(self) -> None:
        self.model = None
        self.version: str | None = None
        self.with_tech = False
        self._last_check = 0.0

    def refresh_if_stale(self) -> None:
        if time.monotonic() - self._last_check < ALIAS_CHECK_TTL_SECONDS:
            return
        self._last_check = time.monotonic()
        client = MlflowClient()
        mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, PRODUCTION_ALIAS)
        if mv.version == self.version:
            return
        log.info("loading %s v%s (was v%s)", REGISTERED_MODEL_NAME, mv.version, self.version)
        model = mlflow.sklearn.load_model(
            f"models:/{REGISTERED_MODEL_NAME}/{mv.version}"
        )
        run = client.get_run(mv.run_id)
        self.with_tech = run.data.params.get("with_tech", "False") == "True"
        self.model, self.version = model, mv.version

    @property
    def feature_cols(self) -> list[str]:
        return FEATURES + (TECH_FEATURES if self.with_tech else [])


manager = ModelManager()
app = FastAPI(title="stock-predictor serving", version="1.0")


@app.on_event("startup")
def startup() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    try:
        manager.refresh_if_stale()
    except Exception as exc:  # stay up; surface the problem via /health
        log.warning("model not loaded at startup: %s", exc)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": manager.model is not None}


@app.get("/model-info")
def model_info() -> dict:
    manager.refresh_if_stale()
    if manager.model is None:
        raise HTTPException(503, "no production model available")
    return {
        "registered_model": REGISTERED_MODEL_NAME,
        "alias": PRODUCTION_ALIAS,
        "version": manager.version,
        "with_tech": manager.with_tech,
        "n_features": len(manager.feature_cols),
    }


@app.get("/predict")
def predict(ticker: str) -> dict:
    ticker = ticker.upper()
    if ticker not in TICKERS:
        raise HTTPException(404, f"unknown ticker {ticker}; supported: {TICKERS}")
    manager.refresh_if_stale()
    if manager.model is None:
        raise HTTPException(503, "no production model available")

    ds = build_dataset(tickers=[ticker], with_tech=manager.with_tech)
    row = ds.iloc[[-1]]
    X = row[manager.feature_cols + ["ticker"]].copy()
    X["ticker"] = pd.Categorical(X["ticker"], categories=TICKERS)
    proba_up = float(manager.model.predict_proba(X)[0, 1])
    return {
        "ticker": ticker,
        "as_of": str(row.index[0].date()),
        "prob_up_next_day": round(proba_up, 4),
        "direction": "up" if proba_up > 0.5 else "down",
        "model_version": manager.version,
        "disclaimer": "demo platform; directional edge is ~1-3pp over coin flip",
    }
