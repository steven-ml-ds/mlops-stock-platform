"""Optuna hyperparameter search over the walk-forward CV objective.

Trials are cheap (no MLflow logging, no holdout touch); only the best
configuration is retrained through the full train_once() pipeline so it
lands in MLflow with calibration, holdout metrics, and artifacts.

Run: uv run python scripts/tune.py --trials 40
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import optuna

from platform_core.evaluate import direction_metrics, holdout_split
from platform_core.features import FEATURES, TARGET_DIR, build_dataset
from platform_core.train import make_model, train_once, walk_forward_splits

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
optuna.logging.set_verbosity(optuna.logging.WARNING)

X_COLS = FEATURES + ["ticker"]


def cv_accuracy(dev, params: dict) -> float:
    accs = []
    for train_end, val_start, val_end in walk_forward_splits(dev.index):
        tr = dev[dev.index <= train_end]
        va = dev[(dev.index >= val_start) & (dev.index <= val_end)]
        model = make_model(params)
        model.fit(tr[X_COLS], tr[TARGET_DIR], categorical_feature=["ticker"])
        m = direction_metrics(va[TARGET_DIR], model.predict_proba(va[X_COLS])[:, 1])
        accs.append(m["accuracy"])
    return float(np.mean(accs))


def main(n_trials: int) -> None:
    ds = build_dataset()
    ds["ticker"] = ds["ticker"].astype("category")
    dev, _ = holdout_split(ds)  # the holdout stays untouched by the search

    def objective(trial: optuna.Trial) -> float:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 3, 63),
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 120),
        }
        return cv_accuracy(dev, params)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f"best cv_accuracy={study.best_value:.4f}  params={study.best_params}")
    run_id = train_once(study.best_params, run_name="optuna-best", register=False)
    print(f"final run logged: {run_id}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=40)
    main(p.parse_args().trials)
