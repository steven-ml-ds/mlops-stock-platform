"""Shared evaluation: every model must beat the always-up baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score


def direction_metrics(y_true: pd.Series, proba_up: np.ndarray) -> dict[str, float]:
    """Directional accuracy + AUC + calibration + edge over always-up baseline."""
    pred = (proba_up > 0.5).astype(int)
    acc = float((pred == y_true).mean())
    baseline = float(max(y_true.mean(), 1 - y_true.mean()))  # majority class
    return {
        "accuracy": acc,
        "auc": float(roc_auc_score(y_true, proba_up)),
        "brier": float(brier_score_loss(y_true, proba_up)),
        "baseline_accuracy": baseline,
        "edge_over_baseline": acc - baseline,
    }


def holdout_split(ds: pd.DataFrame, holdout_days: int = 120):
    """Chronological split on unique dates; the holdout is the live exam that
    promote.py later uses to compare champion vs challenger."""
    dates = ds.index.unique().sort_values()
    cutoff = dates[-holdout_days]
    return ds[ds.index < cutoff], ds[ds.index >= cutoff]
