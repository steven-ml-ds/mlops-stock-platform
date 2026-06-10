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
import numpy as np
import pandas as pd
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


WIN_PROB_THRESHOLD = 0.60  # challenger must win in >=60% of bootstrap resamples


def bootstrap_win_prob(
    y_true, proba_champ, proba_chall, dates, n_resamples: int = 1000, seed: int = 42
) -> float:
    """P(challenger accuracy > champion accuracy) under a date-cluster bootstrap.

    Resampling whole dates (all tickers of a day move together) respects the
    cross-sectional correlation that row-wise bootstrap would destroy.
    Ties split 50/50 so equal models land at win_prob 0.5, not 0 or 1.
    """
    y = np.asarray(y_true, dtype=int)
    champ_hit = ((np.asarray(proba_champ) > 0.5).astype(int) == y).astype(float)
    chall_hit = ((np.asarray(proba_chall) > 0.5).astype(int) == y).astype(float)
    date_idx = pd.Index(dates)
    unique_dates = date_idx.unique()
    rng = np.random.default_rng(seed)
    groups = {d: np.flatnonzero(date_idx == d) for d in unique_dates}

    wins = 0.0
    for _ in range(n_resamples):
        sampled = rng.choice(unique_dates, size=len(unique_dates), replace=True)
        rows = np.concatenate([groups[d] for d in sampled])
        diff = chall_hit[rows].mean() - champ_hit[rows].mean()
        wins += 1.0 if diff > 0 else (0.5 if diff == 0 else 0.0)
    return wins / n_resamples


def should_promote(
    champion: dict | None, challenger: dict, win_prob: float | None = None,
    threshold: float = WIN_PROB_THRESHOLD,
) -> bool:
    """Pure decision rule, unit-tested in tests/test_promote.py.

    Cold start promotes unconditionally; otherwise the challenger needs a
    bootstrap win probability at or above the threshold — a point-estimate
    win inside the noise band is no longer enough to move production.
    """
    if champion is None:  # cold start: any model beats no model
        return True
    if win_prob is None:
        raise ValueError("win_prob is required when a champion exists")
    return win_prob >= threshold


def _score_version(client: MlflowClient, version: str, holdout):
    """Evaluate one registered version on the current holdout window.

    Returns (metrics, proba_up) so the caller can bootstrap model comparisons.
    """
    model = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}/{version}")
    run = client.get_run(client.get_model_version(REGISTERED_MODEL_NAME, version).run_id)
    with_tech = run.data.params.get("with_tech", "False") == "True"
    cols = FEATURES + (TECH_FEATURES if with_tech else []) + ["ticker"]
    X = holdout[cols].copy()
    X["ticker"] = X["ticker"].astype("category")
    proba = model.predict_proba(X)[:, 1]
    return direction_metrics(holdout[TARGET_DIR], proba), proba


def compare_and_promote(
    challenger_run_id: str, threshold: float = WIN_PROB_THRESHOLD
) -> dict:
    """Steps 2–3 for an already-registered challenger (Airflow passes the
    run_id via XCom from the training task)."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    challenger_version = next(
        v.version
        for v in client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
        if v.run_id == challenger_run_id
    )

    # 2. score both on the same current holdout
    ds = build_dataset(with_tech=True)  # superset so either feature set slices out
    ds["ticker"] = ds["ticker"].astype("category")
    _, holdout = holdout_split(ds)

    challenger_metrics, challenger_proba = _score_version(
        client, challenger_version, holdout
    )
    try:
        champion_v = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, PRODUCTION_ALIAS)
        champion_metrics, champion_proba = _score_version(client, champion_v.version, holdout)
        win_prob = bootstrap_win_prob(
            holdout[TARGET_DIR], champion_proba, challenger_proba, holdout.index
        )
    except Exception:
        champion_v, champion_metrics, win_prob = None, None, None

    # 3. decide and (maybe) move the pointer
    promoted = should_promote(champion_metrics, challenger_metrics, win_prob, threshold)
    if promoted:
        client.set_registered_model_alias(
            REGISTERED_MODEL_NAME, PRODUCTION_ALIAS, challenger_version
        )

    result = {
        "challenger_version": challenger_version,
        "challenger_accuracy": challenger_metrics["accuracy"],
        "champion_version": champion_v.version if champion_v else None,
        "champion_accuracy": champion_metrics["accuracy"] if champion_metrics else None,
        "win_prob": win_prob,
        "promoted": promoted,
    }
    log.info("promotion result: %s", result)
    return result


def run_promotion(
    threshold: float = WIN_PROB_THRESHOLD, params: dict | None = None
) -> dict:
    """Train a fresh challenger then run the comparison gate (manual entry point)."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    run_id = train_once(params or DEFAULT_PARAMS, run_name="challenger", register=True)
    return compare_and_promote(run_id, threshold=threshold)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--win-prob", type=float, default=WIN_PROB_THRESHOLD,
                   help="bootstrap win probability the challenger needs to take production")
    a = p.parse_args()
    run_promotion(threshold=a.win_prob)
