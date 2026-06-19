"""Probability calibration: isotonic regression fit on a held-out time window.

LightGBM's raw scores rank well but are not probabilities. Serving returns
prob_up to users, so we calibrate on the most recent slice of the dev window
(time-ordered — never a random split, which would leak).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


class CalibratedDirectionModel:
    """sklearn-compatible wrapper: base classifier + isotonic calibrator.

    Picklable via cloudpickle, so it rides through MLflow's sklearn flavor
    and loads identically in the API and Airflow containers.

    ``categoricals`` maps each categorical feature to the full category list seen at fit time.
    ``predict_proba`` re-imposes it, so the model predicts correctly even when a consumer
    passes the column as plain strings — e.g. MLflow's serving-input validation, which
    round-trips the input example through JSON and drops the pandas category dtype. Without
    this, LightGBM raises "train and valid dataset categorical_feature do not match".
    """

    # Class-level default so models pickled before this attribute existed (e.g. an already
    # registered production version) unpickle and serve without AttributeError — they simply
    # skip coercion, matching their original behaviour.
    categoricals: dict[str, list] = {}

    def __init__(self, base, calibrator: IsotonicRegression,
                 categoricals: dict[str, list] | None = None):
        self.base = base
        self.calibrator = calibrator
        self.categoricals = categoricals or {}

    def _coerce(self, X):
        if not self.categoricals or not isinstance(X, pd.DataFrame):
            return X
        X = X.copy()
        for col, cats in self.categoricals.items():
            if col in X.columns:
                X[col] = pd.Categorical(X[col], categories=cats)
        return X

    def predict_proba(self, X) -> np.ndarray:
        raw = self.base.predict_proba(self._coerce(X))[:, 1]
        cal = self.calibrator.predict(raw)
        return np.column_stack([1.0 - cal, cal])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


def fit_calibrator(raw_proba: np.ndarray, y_true) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(raw_proba, y_true)
    return iso


def reliability_curve(y_true, proba, n_bins: int = 10):
    """(bin_mean_predicted, bin_observed_rate, bin_count) for a reliability plot."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(proba, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    mean_pred, obs_rate, counts = [], [], []
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        mean_pred.append(p[mask].mean())
        obs_rate.append(y[mask].mean())
        counts.append(int(mask.sum()))
    return np.array(mean_pred), np.array(obs_rate), np.array(counts)
