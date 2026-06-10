"""Leakage guards: the tests that keep the model honest."""

import numpy as np
import pandas as pd
import pytest

from platform_core.features import FEATURES, TARGET_RET, _ticker_features


def synthetic_ohlcv(n=400, seed=7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    open_ = close * np.exp(rng.normal(0, 0.005, n))
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 0.005, n)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.005, n)))
    volume = rng.integers(1e6, 5e6, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


def test_target_is_next_day_return():
    df = synthetic_ohlcv()
    f = _ticker_features(df, with_tech=False)
    logret = np.log(df["close"]).diff()
    # fwd_ret at t must equal the realized return of t+1
    expected = logret.shift(-1)
    pd.testing.assert_series_equal(
        f[TARGET_RET].dropna(), expected.dropna(), check_names=False
    )


@pytest.mark.parametrize("col", [c for c in FEATURES if c not in
                                 ("spy_ret_lag1", "vix_level", "vix_chg", "curve_slope")])
def test_no_lookahead_per_feature(col):
    """Mutating the future must not change any feature value at time t."""
    df = synthetic_ohlcv()
    t = 300  # checkpoint well past all warm-up windows (max 252)
    base = _ticker_features(df, with_tech=False)[col].iloc[: t + 1]

    corrupted = df.copy()
    corrupted.iloc[t + 1 :] *= 7.77  # absurd future
    after = _ticker_features(corrupted, with_tech=False)[col].iloc[: t + 1]

    pd.testing.assert_series_equal(base, after, check_names=False)


def test_target_does_change_with_future():
    """Sanity check that the leakage test has teeth: the target SHOULD move."""
    df = synthetic_ohlcv()
    t = 300
    base = _ticker_features(df, with_tech=False)[TARGET_RET].iloc[t]
    corrupted = df.copy()
    corrupted.iloc[t + 1 :, corrupted.columns.get_loc("close")] *= 7.77
    after = _ticker_features(corrupted, with_tech=False)[TARGET_RET].iloc[t]
    assert base != after


def test_no_raw_price_levels():
    """All features must be scale-free: multiplying prices by 1000 changes nothing
    except volume-based and calendar features (which are price-free anyway)."""
    df = synthetic_ohlcv()
    scaled = df.copy()
    scaled[["open", "high", "low", "close"]] *= 1000.0
    a = _ticker_features(df, with_tech=False)
    b = _ticker_features(scaled, with_tech=False)
    price_free = {"volu_z20", "dow"}
    for col in FEATURES:
        if col in ("spy_ret_lag1", "vix_level", "vix_chg", "curve_slope"):
            continue
        if col in price_free:
            continue
        pd.testing.assert_series_equal(
            a[col].dropna(), b[col].dropna(), check_names=False,
            rtol=1e-9, obj=col,
        )
