import numpy as np
import pandas as pd

from platform_core.drift import psi


def test_psi_same_distribution_is_small():
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(0, 1, 2000))
    b = pd.Series(rng.normal(0, 1, 2000))
    assert psi(a, b) < 0.05


def test_psi_shifted_distribution_is_large():
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(0, 1, 2000))
    b = pd.Series(rng.normal(1.5, 1, 2000))  # big mean shift
    assert psi(a, b) > 0.2


def test_psi_scale_change_detected():
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(0, 1, 2000))
    b = pd.Series(rng.normal(0, 3, 2000))  # vol regime change
    assert psi(a, b) > 0.2


def test_psi_handles_low_cardinality():
    a = pd.Series([0, 1, 2, 3, 4] * 100)  # like the dow feature
    b = pd.Series([0, 1, 2, 3, 4] * 100)
    assert psi(a, b) < 0.05
