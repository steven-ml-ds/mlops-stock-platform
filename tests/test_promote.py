import numpy as np
import pandas as pd
import pytest

from platform_core.promote import bootstrap_win_prob, should_promote

CHAMP = {"accuracy": 0.52}
CHALL = {"accuracy": 0.53}


def test_cold_start_promotes():
    assert should_promote(None, {"accuracy": 0.40})


def test_decisive_win_promotes():
    assert should_promote(CHAMP, CHALL, win_prob=0.75, threshold=0.60)


def test_noisy_point_win_is_blocked():
    # challenger's point estimate is higher, but the bootstrap says coin flip
    assert not should_promote(CHAMP, CHALL, win_prob=0.52, threshold=0.60)


def test_missing_win_prob_raises():
    with pytest.raises(ValueError):
        should_promote(CHAMP, CHALL)


def _setup(n_days=120, tickers=3, seed=0):
    rng = np.random.default_rng(seed)
    dates = np.repeat(pd.bdate_range("2025-01-01", periods=n_days), tickers)
    y = rng.integers(0, 2, len(dates))
    return dates, y, rng


def test_bootstrap_equal_models_near_half():
    dates, y, rng = _setup()
    proba = rng.uniform(0, 1, len(y))
    wp = bootstrap_win_prob(y, proba, proba, dates, n_resamples=200)
    assert wp == 0.5  # identical predictions -> all ties


def test_bootstrap_dominant_challenger_near_one():
    dates, y, _ = _setup()
    champ = np.where(y == 1, 0.4, 0.6)   # always wrong
    chall = np.where(y == 1, 0.6, 0.4)   # always right
    wp = bootstrap_win_prob(y, champ, chall, dates, n_resamples=200)
    assert wp == 1.0


def test_bootstrap_marginal_edge_is_uncertain():
    # challenger fixes 12 calls but breaks 8 others: net better, still noisy
    dates, y, rng = _setup(seed=3)
    champ = rng.uniform(0, 1, len(y))
    chall = champ.copy()
    idx = rng.choice(len(y), size=20, replace=False)
    fix, brk = idx[:12], idx[12:]
    chall[fix] = np.where(y[fix] == 1, 0.9, 0.1)
    chall[brk] = np.where(y[brk] == 1, 0.1, 0.9)
    wp = bootstrap_win_prob(y, champ, chall, dates, n_resamples=300)
    assert 0.5 < wp < 0.99  # better, but not certain -> the gate can block it
