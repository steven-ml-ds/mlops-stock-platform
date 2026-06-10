"""Serving API: always serves whatever version @production points to.

The model manager re-checks the aliases (with a small TTL) and hot-reloads
when promote.py moves a pointer — deploys and promotions are decoupled.

Shadow deployment: if a @challenger alias exists, its prediction is computed
on every /predict and appended to data/predictions/shadow.jsonl, but only the
production model ever answers. monitor.compare_shadow() turns that log into a
live-traffic model comparison once outcomes realize.

Run locally:  uv run uvicorn app.main:app --port 8000
"""

from __future__ import annotations

import json
import logging
import time

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow import MlflowClient

from platform_core.config import (
    CHALLENGER_ALIAS,
    MLFLOW_TRACKING_URI,
    PRODUCTION_ALIAS,
    REGISTERED_MODEL_NAME,
    SHADOW_LOG,
    TICKERS,
)
from platform_core.features import FEATURES, TECH_FEATURES, build_dataset

log = logging.getLogger("uvicorn.error")

ALIAS_CHECK_TTL_SECONDS = 30.0


class AliasSlot:
    def __init__(self, alias: str):
        self.alias = alias
        self.model = None
        self.version: str | None = None
        self.with_tech = False

    @property
    def feature_cols(self) -> list[str]:
        return FEATURES + (TECH_FEATURES if self.with_tech else [])


class ModelManager:
    """Caches one model per alias; swaps a slot when its alias moves."""

    def __init__(self) -> None:
        self.slots = {
            PRODUCTION_ALIAS: AliasSlot(PRODUCTION_ALIAS),
            CHALLENGER_ALIAS: AliasSlot(CHALLENGER_ALIAS),
        }
        self._last_check = 0.0

    def refresh_if_stale(self) -> None:
        if time.monotonic() - self._last_check < ALIAS_CHECK_TTL_SECONDS:
            return
        self._last_check = time.monotonic()
        client = MlflowClient()
        for slot in self.slots.values():
            try:
                mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, slot.alias)
            except Exception:
                slot.model, slot.version = None, None  # alias not set (yet)
                continue
            if mv.version == slot.version:
                continue
            log.info("loading @%s -> v%s (was v%s)", slot.alias, mv.version, slot.version)
            slot.model = mlflow.sklearn.load_model(
                f"models:/{REGISTERED_MODEL_NAME}/{mv.version}"
            )
            run = client.get_run(mv.run_id)
            slot.with_tech = run.data.params.get("with_tech", "False") == "True"
            slot.version = mv.version

    @property
    def production(self) -> AliasSlot:
        return self.slots[PRODUCTION_ALIAS]

    @property
    def challenger(self) -> AliasSlot:
        return self.slots[CHALLENGER_ALIAS]


def _proba_up(slot: AliasSlot, row: pd.DataFrame) -> float:
    X = row[slot.feature_cols + ["ticker"]].copy()
    X["ticker"] = pd.Categorical(X["ticker"], categories=TICKERS)
    return float(slot.model.predict_proba(X)[0, 1])


def _log_shadow(record: dict) -> None:
    try:
        SHADOW_LOG.parent.mkdir(parents=True, exist_ok=True)
        with SHADOW_LOG.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
    except OSError as exc:  # shadow logging must never break serving
        log.warning("shadow log write failed: %s", exc)


manager = ModelManager()
app = FastAPI(title="stock-predictor serving", version="2.0")


@app.on_event("startup")
def startup() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    try:
        manager.refresh_if_stale()
    except Exception as exc:  # stay up; surface the problem via /health
        log.warning("models not loaded at startup: %s", exc)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": manager.production.model is not None}


@app.get("/model-info")
def model_info() -> dict:
    manager.refresh_if_stale()
    if manager.production.model is None:
        raise HTTPException(503, "no production model available")
    return {
        "registered_model": REGISTERED_MODEL_NAME,
        "production": {
            "version": manager.production.version,
            "with_tech": manager.production.with_tech,
            "n_features": len(manager.production.feature_cols),
        },
        "challenger": (
            {"version": manager.challenger.version, "shadow": True}
            if manager.challenger.model is not None
            else None
        ),
    }


@app.get("/predict")
def predict(ticker: str) -> dict:
    ticker = ticker.upper()
    if ticker not in TICKERS:
        raise HTTPException(404, f"unknown ticker {ticker}; supported: {TICKERS}")
    manager.refresh_if_stale()
    prod = manager.production
    if prod.model is None:
        raise HTTPException(503, "no production model available")

    # superset dataset so production and challenger slice their own columns
    ds = build_dataset(tickers=[ticker], with_tech=True)
    row = ds.iloc[[-1]]
    proba_up = _proba_up(prod, row)

    chall = manager.challenger
    if chall.model is not None:
        _log_shadow({
            "date": str(row.index[0].date()),
            "ticker": ticker,
            "prod_version": prod.version,
            "prod_proba": round(proba_up, 6),
            "chall_version": chall.version,
            "chall_proba": round(_proba_up(chall, row), 6),
        })

    return {
        "ticker": ticker,
        "as_of": str(row.index[0].date()),
        "prob_up_next_day": round(proba_up, 4),
        "direction": "up" if proba_up > 0.5 else "down",
        "model_version": prod.version,
        "disclaimer": "demo platform; directional edge is ~1-3pp over coin flip",
    }
