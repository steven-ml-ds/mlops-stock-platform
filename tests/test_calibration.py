"""The calibrated model must predict the same whether the categorical `ticker` column is
passed as a pandas category or as plain strings.

MLflow's serving-input validation round-trips the input example through JSON, which turns the
category column into plain strings. Without category coercion in the wrapper, LightGBM raises
"train and valid dataset categorical_feature do not match" — the regression this guards.
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from platform_core.calibration import CalibratedDirectionModel, fit_calibrator

TICKERS = ["AAA", "BBB", "CCC"]


def _fit_model():
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame(
        {
            "f1": rng.normal(size=n),
            "f2": rng.normal(size=n),
            "ticker": pd.Categorical(rng.choice(TICKERS, n), categories=TICKERS),
        }
    )
    y = (df["f1"] + rng.normal(scale=0.1, size=n) > 0).astype(int)
    base = LGBMClassifier(n_estimators=20, num_leaves=7, verbose=-1)
    base.fit(df[["f1", "f2", "ticker"]], y, categorical_feature=["ticker"])
    raw = base.predict_proba(df[["f1", "f2", "ticker"]])[:, 1]
    model = CalibratedDirectionModel(
        base, fit_calibrator(raw, y), categoricals={"ticker": TICKERS}
    )
    return model, df[["f1", "f2", "ticker"]]


def test_predict_proba_accepts_string_ticker():
    model, X = _fit_model()
    cat = model.predict_proba(X)[:, 1]

    X_str = X.copy()
    X_str["ticker"] = X_str["ticker"].astype(str)  # what MLflow's JSON round-trip yields
    string = model.predict_proba(X_str)[:, 1]

    np.testing.assert_allclose(cat, string)


def test_old_pickle_without_categoricals_still_serves():
    """A model pickled before `categoricals` existed must serve without AttributeError —
    unpickling restores __dict__ and skips __init__, so the class-level default must apply."""
    model, X = _fit_model()
    legacy = CalibratedDirectionModel.__new__(CalibratedDirectionModel)
    legacy.base = model.base
    legacy.calibrator = model.calibrator  # note: no `categoricals` in __dict__
    proba = legacy.predict_proba(X)  # must not raise
    assert proba.shape == (len(X), 2)
