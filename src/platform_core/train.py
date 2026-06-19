"""Train a direction classifier with walk-forward CV, fully logged to MLflow.

Usage:
    uv run python -m platform_core.train --run-name lr05 --learning-rate 0.05
    uv run python -m platform_core.train --with-tech   # tech-indicator ablation
"""

from __future__ import annotations

import argparse
import logging

import lightgbm as lgb
import matplotlib
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from platform_core.calibration import (
    CalibratedDirectionModel,
    fit_calibrator,
    reliability_curve,
)
from platform_core.config import EXPERIMENT_NAME, MLFLOW_TRACKING_URI
from platform_core.evaluate import direction_metrics, holdout_split
from platform_core.features import FEATURES, TARGET_DIR, TECH_FEATURES, build_dataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _reliability_figure(y_true, raw_proba, cal_proba):
    fig, ax = plt.subplots(figsize=(6, 6))
    for label, proba in (("raw", raw_proba), ("calibrated", cal_proba)):
        mp, obs, _ = reliability_curve(y_true, proba)
        ax.plot(mp, obs, "o-", label=label)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect")
    ax.set_xlabel("mean predicted prob_up")
    ax.set_ylabel("observed up-rate")
    ax.set_title("Reliability diagram (holdout)")
    ax.legend()
    return fig

log = logging.getLogger(__name__)

CV_FOLDS = 5
PURGE_GAP_DAYS = 5  # dead zone between train end and validation start
CALIBRATION_DAYS = 120  # tail of the dev window reserved for isotonic fit


def walk_forward_splits(dates: pd.Index, n_folds: int = CV_FOLDS, gap: int = PURGE_GAP_DAYS):
    """Expanding-window splits over unique dates with a purge gap.

    Pooling 3 tickers means rows share dates, so we split on dates, never on
    row position — otherwise one ticker's day t could sit in train while
    another's day t sits in validation.
    """
    unique = dates.unique().sort_values()
    fold_size = len(unique) // (n_folds + 1)
    for k in range(1, n_folds + 1):
        train_end = unique[k * fold_size - 1]
        val_start = unique[k * fold_size - 1 + gap]
        val_end = unique[min((k + 1) * fold_size - 1 + gap, len(unique) - 1)]
        yield train_end, val_start, val_end


def make_model(params: dict) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        num_leaves=params["num_leaves"],
        min_child_samples=params["min_child_samples"],
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        random_state=42,
        verbose=-1,
    )


def train_once(params: dict, with_tech: bool = False, run_name: str | None = None,
               register: bool = False) -> str:
    """One experiment run: CV + holdout, everything logged. Returns run_id."""
    feature_cols = FEATURES + (TECH_FEATURES if with_tech else [])
    ds = build_dataset(with_tech=with_tech)
    ds["ticker"] = ds["ticker"].astype("category")
    dev, holdout = holdout_split(ds)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_params({
            "with_tech": with_tech,
            "calibration": "isotonic",
            "calibration_days": CALIBRATION_DAYS,
            "n_features": len(feature_cols),
            "n_rows_dev": len(dev),
            "n_rows_holdout": len(holdout),
            "dev_end": str(dev.index.max().date()),
        })

        # ---- walk-forward CV on the dev window
        cv_acc, cv_auc = [], []
        X_cols = feature_cols + ["ticker"]
        for train_end, val_start, val_end in walk_forward_splits(dev.index):
            tr = dev[dev.index <= train_end]
            va = dev[(dev.index >= val_start) & (dev.index <= val_end)]
            model = make_model(params)
            model.fit(tr[X_cols], tr[TARGET_DIR], categorical_feature=["ticker"])
            m = direction_metrics(va[TARGET_DIR], model.predict_proba(va[X_cols])[:, 1])
            cv_acc.append(m["accuracy"])
            cv_auc.append(m["auc"])
        mlflow.log_metrics({
            "cv_accuracy_mean": float(np.mean(cv_acc)),
            "cv_accuracy_std": float(np.std(cv_acc)),
            "cv_auc_mean": float(np.mean(cv_auc)),
        })

        # ---- final fit + time-ordered isotonic calibration, scored on holdout
        # base learns on dev minus the calibration tail; the calibrator maps the
        # base's raw scores to honest probabilities on data it never trained on
        dev_dates = dev.index.unique().sort_values()
        cal_start = dev_dates[-CALIBRATION_DAYS]
        fit_part, cal_part = dev[dev.index < cal_start], dev[dev.index >= cal_start]
        base = make_model(params)
        base.fit(fit_part[X_cols], fit_part[TARGET_DIR], categorical_feature=["ticker"])
        raw_cal = base.predict_proba(cal_part[X_cols])[:, 1]
        model = CalibratedDirectionModel(
            base,
            fit_calibrator(raw_cal, cal_part[TARGET_DIR]),
            categoricals={"ticker": list(ds["ticker"].cat.categories)},
        )

        raw_hold = base.predict_proba(holdout[X_cols])[:, 1]
        cal_hold = model.predict_proba(holdout[X_cols])[:, 1]
        hold = direction_metrics(holdout[TARGET_DIR], cal_hold)
        mlflow.log_metrics({f"holdout_{k}": v for k, v in hold.items()})
        mlflow.log_metric(
            "holdout_brier_uncalibrated",
            float(brier_score_loss(holdout[TARGET_DIR], raw_hold)),
        )
        mlflow.log_figure(
            _reliability_figure(holdout[TARGET_DIR], raw_hold, cal_hold),
            "reliability_diagram.png",
        )

        # MLflow can't serialize a pandas category column; hand it a plain-string ticker so
        # signature inference and serving-input validation round-trip cleanly (the wrapper
        # re-imposes the categories at predict time).
        example = dev[X_cols].tail(3).copy()
        example["ticker"] = example["ticker"].astype(str)
        mlflow.sklearn.log_model(
            model,
            name="model",
            input_example=example,
            registered_model_name="stock-predictor" if register else None,
        )
        log.info(
            "run %s  cv_acc=%.4f  holdout_acc=%.4f (baseline %.4f, edge %+.4f)",
            run.info.run_id, np.mean(cv_acc), hold["accuracy"],
            hold["baseline_accuracy"], hold["edge_over_baseline"],
        )
        return run.info.run_id


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", default=None)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--num-leaves", type=int, default=15)
    p.add_argument("--n-estimators", type=int, default=300)
    p.add_argument("--min-child-samples", type=int, default=40)
    p.add_argument("--with-tech", action="store_true")
    p.add_argument("--register", action="store_true",
                   help="register the model as a new stock-predictor version")
    return p.parse_args(argv)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    a = parse_args()
    params = {
        "learning_rate": a.learning_rate,
        "num_leaves": a.num_leaves,
        "n_estimators": a.n_estimators,
        "min_child_samples": a.min_child_samples,
    }
    train_once(params, with_tech=a.with_tech, run_name=a.run_name, register=a.register)
