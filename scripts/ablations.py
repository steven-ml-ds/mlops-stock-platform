"""M10 model-improvement ablations, each logged as a comparable MLflow run.

1. target engineering: vol-normalized regression / three-class with flat band
2. seed ensemble: averaged probabilities across 5 seeds
3. data scale: 10 tickers x 10 years
4. feature importance: LightGBM gain + permutation importance (figure artifact)

Every variant is scored with the same protocol (chronological 120-day holdout,
directional accuracy vs always-up baseline) so the comparison is honest.

Run: uv run python scripts/ablations.py
"""

from __future__ import annotations

import logging

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from sklearn.inspection import permutation_importance

from platform_core.config import EXPERIMENT_NAME, MLFLOW_TRACKING_URI
from platform_core.data import fetch_ticker
from platform_core.evaluate import direction_metrics, holdout_split
from platform_core.features import FEATURES, TARGET_DIR, TARGET_RET, build_dataset
from platform_core.promote import DEFAULT_PARAMS
from platform_core.train import make_model

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

X_COLS = FEATURES + ["ticker"]
SCALE_TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA",
                 "JPM", "XOM", "JNJ"]


def _log_run(run_name: str, params: dict, holdout, proba_up) -> dict:
    metrics = direction_metrics(holdout[TARGET_DIR], proba_up)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics({f"holdout_{k}": v for k, v in metrics.items()})
    log.info("%-22s acc=%.4f baseline=%.4f edge=%+.4f",
             run_name, metrics["accuracy"], metrics["baseline_accuracy"],
             metrics["edge_over_baseline"])
    return metrics


def _prepared(tickers=None, years=None):
    if years:  # extend the cache window before building
        from platform_core.config import CONTEXT_TICKERS
        for t in (tickers or []) + CONTEXT_TICKERS:
            fetch_ticker(t, years=years)
    ds = build_dataset(tickers=tickers)
    ds["ticker"] = ds["ticker"].astype("category")
    return holdout_split(ds)


def ablation_vol_normalized(dev, holdout):
    """Regression on ret/vol_21 (more stationary target), direction by sign."""
    target = dev[TARGET_RET] / dev["vol_5"].clip(lower=1e-6)
    model = lgb.LGBMRegressor(random_state=42, verbose=-1, **DEFAULT_PARAMS)
    model.fit(dev[X_COLS], target, categorical_feature=["ticker"])
    pred = model.predict(holdout[X_COLS])
    proba_like = 1 / (1 + np.exp(-pred))  # monotone squash: ranking-valid for AUC
    _log_run("ablation-volnorm-reg", {**DEFAULT_PARAMS, "target": "vol_norm_reg"},
             holdout, proba_like)


def ablation_three_class(dev, holdout):
    """up/flat/down with a 0.2-sigma flat band: don't learn noise near zero."""
    sigma = dev[TARGET_RET].std()
    band = 0.2 * sigma
    y3 = np.where(dev[TARGET_RET] > band, 2, np.where(dev[TARGET_RET] < -band, 0, 1))
    model = lgb.LGBMClassifier(objective="multiclass", num_class=3,
                               random_state=42, verbose=-1, **DEFAULT_PARAMS)
    model.fit(dev[X_COLS], y3, categorical_feature=["ticker"])
    proba = model.predict_proba(holdout[X_COLS])
    # direction = renormalized P(up) vs P(down), ignoring the flat mass
    proba_up = proba[:, 2] / (proba[:, 0] + proba[:, 2] + 1e-12)
    _log_run("ablation-three-class",
             {**DEFAULT_PARAMS, "target": "three_class", "flat_band_sigma": 0.2},
             holdout, proba_up)


def ablation_seed_ensemble(dev, holdout, n_seeds: int = 5):
    """Average probabilities over seeds: variance reduction for a weak signal."""
    probas = []
    for seed in range(n_seeds):
        model = make_model(DEFAULT_PARAMS)
        model.set_params(random_state=seed)
        model.fit(dev[X_COLS], dev[TARGET_DIR], categorical_feature=["ticker"])
        probas.append(model.predict_proba(holdout[X_COLS])[:, 1])
    _log_run("ablation-seed-ensemble",
             {**DEFAULT_PARAMS, "target": "binary", "n_seeds": n_seeds},
             holdout, np.mean(probas, axis=0))


def ablation_data_scale():
    """10 tickers x 10 years: does more cross-section + history help?"""
    dev, holdout = _prepared(tickers=SCALE_TICKERS, years=10)
    model = make_model(DEFAULT_PARAMS)
    model.fit(dev[X_COLS], dev[TARGET_DIR], categorical_feature=["ticker"])
    _log_run("ablation-data-10x10",
             {**DEFAULT_PARAMS, "target": "binary", "n_tickers": len(SCALE_TICKERS),
              "history_years": 10, "n_rows_dev": len(dev)},
             holdout, model.predict_proba(holdout[X_COLS])[:, 1])


def feature_importance_report(dev, holdout):
    """Gain + permutation importance on the holdout, logged as an artifact."""
    model = make_model(DEFAULT_PARAMS)
    model.fit(dev[X_COLS], dev[TARGET_DIR], categorical_feature=["ticker"])
    gain = model.booster_.feature_importance(importance_type="gain")
    perm = permutation_importance(
        model, holdout[X_COLS], holdout[TARGET_DIR],
        n_repeats=10, random_state=42, scoring="accuracy",
    )
    order = np.argsort(gain)
    fig, axes = plt.subplots(1, 2, figsize=(13, 8))
    axes[0].barh(np.array(X_COLS)[order], gain[order])
    axes[0].set_title("LightGBM gain importance")
    order_p = np.argsort(perm.importances_mean)
    axes[1].barh(np.array(X_COLS)[order_p], perm.importances_mean[order_p],
                 xerr=perm.importances_std[order_p], color="darkorange")
    axes[1].set_title("Permutation importance (holdout accuracy)")
    fig.tight_layout()
    with mlflow.start_run(run_name="feature-importance"):
        mlflow.log_params(DEFAULT_PARAMS)
        mlflow.log_figure(fig, "feature_importance.png")
    top = np.array(X_COLS)[np.argsort(-gain)][:8]
    log.info("top gain features: %s", ", ".join(top))


if __name__ == "__main__":
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    dev, holdout = _prepared()
    # reference point with identical protocol (no calibration, default params)
    model = make_model(DEFAULT_PARAMS)
    model.fit(dev[X_COLS], dev[TARGET_DIR], categorical_feature=["ticker"])
    _log_run("ablation-reference", {**DEFAULT_PARAMS, "target": "binary"},
             holdout, model.predict_proba(holdout[X_COLS])[:, 1])

    ablation_vol_normalized(dev, holdout)
    ablation_three_class(dev, holdout)
    ablation_seed_ensemble(dev, holdout)
    feature_importance_report(dev, holdout)
    ablation_data_scale()
