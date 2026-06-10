"""Champion/challenger promotion against the registry's @production alias.

Flow (also the tail of the Airflow retrain DAG):
1. train a challenger and register it as a new stock-predictor version
2. score champion (@production) and challenger on the SAME current holdout
3. move the alias only if the challenger wins by at least --min-edge

The alias is a pointer, so promotion and rollback are O(1) and atomic.

Usage:
    uv run python -m platform_core.promote            # train + compare + maybe promote
    uv run python -m platform_core.promote --min-edge 0.005
"""

from __future__ import annotations

import argparse
import logging

import mlflow
from mlflow import MlflowClient

from platform_core.config import (
    MLFLOW_TRACKING_URI,
    PRODUCTION_ALIAS,
    REGISTERED_MODEL_NAME,
)
from platform_core.evaluate import direction_metrics, holdout_split
from platform_core.features import FEATURES, TARGET_DIR, TECH_FEATURES, build_dataset
from platform_core.train import train_once

log = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "learning_rate": 0.03,
    "num_leaves": 7,
    "n_estimators": 400,
    "min_child_samples": 40,
}


def should_promote(champion: dict | None, challenger: dict, min_edge: float = 0.0) -> bool:
    """Pure decision rule, unit-tested in tests/test_promote.py."""
    if champion is None:  # cold start: any model beats no model
        return True
    return challenger["accuracy"] >= champion["accuracy"] + min_edge


def _score_version(client: MlflowClient, version: str, holdout) -> dict:
    """Evaluate one registered version on the current holdout window."""
    model = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}/{version}")
    run = client.get_run(client.get_model_version(REGISTERED_MODEL_NAME, version).run_id)
    with_tech = run.data.params.get("with_tech", "False") == "True"
    cols = FEATURES + (TECH_FEATURES if with_tech else []) + ["ticker"]
    X = holdout[cols].copy()
    X["ticker"] = X["ticker"].astype("category")
    return direction_metrics(holdout[TARGET_DIR], model.predict_proba(X)[:, 1])


def run_promotion(min_edge: float = 0.0, params: dict | None = None) -> dict:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    # 1. challenger: fresh training run, registered as a new version
    run_id = train_once(params or DEFAULT_PARAMS, run_name="challenger", register=True)
    challenger_version = next(
        v.version
        for v in client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
        if v.run_id == run_id
    )

    # 2. score both on the same current holdout
    ds = build_dataset(with_tech=True)  # superset so either feature set slices out
    ds["ticker"] = ds["ticker"].astype("category")
    _, holdout = holdout_split(ds)

    challenger_metrics = _score_version(client, challenger_version, holdout)
    try:
        champion_v = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, PRODUCTION_ALIAS)
        champion_metrics = _score_version(client, champion_v.version, holdout)
    except Exception:
        champion_v, champion_metrics = None, None

    # 3. decide and (maybe) move the pointer
    promoted = should_promote(champion_metrics, challenger_metrics, min_edge)
    if promoted:
        client.set_registered_model_alias(
            REGISTERED_MODEL_NAME, PRODUCTION_ALIAS, challenger_version
        )

    result = {
        "challenger_version": challenger_version,
        "challenger_accuracy": challenger_metrics["accuracy"],
        "champion_version": champion_v.version if champion_v else None,
        "champion_accuracy": champion_metrics["accuracy"] if champion_metrics else None,
        "promoted": promoted,
    }
    log.info("promotion result: %s", result)
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--min-edge", type=float, default=0.0,
                   help="challenger must beat champion holdout accuracy by this much")
    a = p.parse_args()
    run_promotion(min_edge=a.min_edge)
