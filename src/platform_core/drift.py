"""Feature drift monitoring via PSI (population stability index).

Early-warning layer: inputs shifting away from what the production model was
trained on. Complements monitor.check_decay (the late, confirmed signal).

PSI rule of thumb: <0.1 stable, 0.1-0.2 watch, >0.2 drifted.
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
from platform_core.features import FEATURES, build_dataset
from platform_core.monitor import MONITOR_EXPERIMENT

log = logging.getLogger(__name__)

REFERENCE_DAYS = 252
CURRENT_DAYS = 60
PSI_DRIFT_THRESHOLD = 0.2
N_BINS = 10


def psi(reference: pd.Series, current: pd.Series, n_bins: int = N_BINS) -> float:
    """PSI with decile bins fixed on the reference distribution."""
    ref = reference.dropna()
    cur = current.dropna()
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:  # near-constant feature (e.g. dow has few values)
        edges = np.unique(np.concatenate([edges, edges[-1:] + 1e-9]))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_frac = np.histogram(ref, bins=edges)[0] / len(ref)
    cur_frac = np.histogram(cur, bins=edges)[0] / len(cur)
    eps = 1e-4  # keep log finite on empty bins
    ref_frac = np.clip(ref_frac, eps, None)
    cur_frac = np.clip(cur_frac, eps, None)
    return float(np.sum((cur_frac - ref_frac) * np.log(cur_frac / ref_frac)))


def check_drift(
    reference_days: int = REFERENCE_DAYS, current_days: int = CURRENT_DAYS
) -> dict:
    """PSI per feature: production training era vs the most recent window."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, PRODUCTION_ALIAS)
    dev_end = pd.Timestamp(
        client.get_run(mv.run_id).data.params.get("dev_end")
    )

    ds = build_dataset()
    dates = ds.index.unique().sort_values()
    ref_dates = dates[(dates <= dev_end)][-reference_days:]
    cur_dates = dates[-current_days:]
    reference, current = ds.loc[ref_dates], ds.loc[cur_dates]

    scores = {f: psi(reference[f], current[f]) for f in FEATURES}
    drifted = sorted(
        ((f, s) for f, s in scores.items() if s > PSI_DRIFT_THRESHOLD),
        key=lambda kv: -kv[1],
    )

    mlflow.set_experiment(MONITOR_EXPERIMENT)
    with mlflow.start_run(run_name=f"drift-check-v{mv.version}"):
        mlflow.log_params({
            "model_version": mv.version,
            "reference_end": str(dev_end.date()),
            "reference_days": reference_days,
            "current_days": current_days,
        })
        mlflow.log_metrics({f"psi_{f}": s for f, s in scores.items()})
        mlflow.log_metrics({"n_drifted": len(drifted), "drift_flag": int(bool(drifted))})

    result = {
        "model_version": mv.version,
        "n_drifted": len(drifted),
        "drifted": dict(drifted),
        "max_psi": max(scores.values()),
    }
    log.info("drift check: %s", result)
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--current-days", type=int, default=CURRENT_DAYS)
    r = check_drift(current_days=p.parse_args().current_days)
    print(f"\nmax PSI = {r['max_psi']:.3f}, drifted features: {r['drifted'] or 'none'}")
