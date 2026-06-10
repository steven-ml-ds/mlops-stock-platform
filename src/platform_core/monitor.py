"""Production-model decay check: rolling directional accuracy on recent days.

Logged to its own MLflow experiment so the trend is visible over time.
The weekly DAG runs this first; a degraded flag is the platform's signal
that the scheduled retrain matters this week (and, in a richer setup,
would trigger an off-schedule retrain or an alert).
"""

from __future__ import annotations

import argparse
import logging

import mlflow
from mlflow import MlflowClient

import json

import pandas as pd

from platform_core.config import (
    MLFLOW_TRACKING_URI,
    PRODUCTION_ALIAS,
    REGISTERED_MODEL_NAME,
    SHADOW_LOG,
)
from platform_core.evaluate import direction_metrics
from platform_core.features import FEATURES, TARGET_DIR, TECH_FEATURES, build_dataset

log = logging.getLogger(__name__)

MONITOR_EXPERIMENT = "production-monitoring"
RECENT_DAYS = 60
DECAY_THRESHOLD = -0.02  # degraded when edge_over_baseline drops below -2pp


def check_decay(recent_days: int = RECENT_DAYS) -> dict:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, PRODUCTION_ALIAS)
    model = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}/{mv.version}")
    with_tech = (
        client.get_run(mv.run_id).data.params.get("with_tech", "False") == "True"
    )
    cols = FEATURES + (TECH_FEATURES if with_tech else []) + ["ticker"]

    ds = build_dataset(with_tech=with_tech)
    ds["ticker"] = ds["ticker"].astype("category")
    dates = ds.index.unique().sort_values()
    recent = ds[ds.index >= dates[-recent_days]]

    metrics = direction_metrics(recent[TARGET_DIR], model.predict_proba(recent[cols])[:, 1])
    degraded = metrics["edge_over_baseline"] < DECAY_THRESHOLD

    mlflow.set_experiment(MONITOR_EXPERIMENT)
    with mlflow.start_run(run_name=f"decay-check-v{mv.version}"):
        mlflow.log_params({"model_version": mv.version, "recent_days": recent_days})
        mlflow.log_metrics({f"recent_{k}": v for k, v in metrics.items()})
        mlflow.log_metric("degraded", int(degraded))

    result = {"model_version": mv.version, "degraded": degraded, **metrics}
    log.info("decay check: %s", result)
    return result


def compare_shadow(min_rows: int = 30) -> dict:
    """Live-traffic comparison: join shadow log with realized outcomes.

    Each shadow record stores both models' prob_up for (date, ticker);
    once the next day's return realizes, both predictions can be scored
    on identical real traffic — stronger evidence than any holdout.
    """
    if not SHADOW_LOG.exists():
        return {"status": "no shadow log yet"}
    records = [json.loads(line) for line in SHADOW_LOG.read_text().splitlines()]
    shadow = (
        pd.DataFrame(records)
        .drop_duplicates(subset=["date", "ticker"], keep="last")
        .assign(date=lambda d: pd.to_datetime(d["date"]))
    )

    realized = build_dataset()[["ticker", TARGET_DIR]].reset_index()
    joined = shadow.merge(realized, on=["date", "ticker"], how="inner")
    result: dict = {"n_scored": len(joined), "n_pending": len(shadow) - len(joined)}
    if len(joined) < min_rows:
        result["status"] = f"accumulating ({len(joined)}/{min_rows} scored rows)"
        return result

    y = joined[TARGET_DIR]
    prod_acc = float(((joined["prod_proba"] > 0.5).astype(int) == y).mean())
    chall_acc = float(((joined["chall_proba"] > 0.5).astype(int) == y).mean())
    result.update({"prod_accuracy": prod_acc, "chall_accuracy": chall_acc})

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MONITOR_EXPERIMENT)
    with mlflow.start_run(run_name="shadow-comparison"):
        mlflow.log_metrics({
            "shadow_n_scored": len(joined),
            "shadow_prod_accuracy": prod_acc,
            "shadow_chall_accuracy": chall_acc,
        })
    log.info("shadow comparison: %s", result)
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--recent-days", type=int, default=RECENT_DAYS)
    check_decay(recent_days=p.parse_args().recent_days)
