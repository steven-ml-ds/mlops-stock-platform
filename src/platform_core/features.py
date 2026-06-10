"""Feature engineering — the 23-feature list finalized by the EDA gate.

See docs/notes/02 for the pruning evidence. Invariants enforced here and
tested in tests/test_features.py:
- every feature at date t uses information from dates <= t only
- target[t] is the NEXT day's log return (the one thing allowed to look ahead)
- all features are returns/ratios, never raw price levels (non-stationary)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from platform_core.data import load_cached

FEATURES = [
    "ret_lag1", "ret_lag2", "ret_lag3", "ret_lag5", "ret_lag10", "ret_lag21",
    "mom_21",
    "gap", "intraday",
    "px_ma20", "dist_hi252",
    "vol_5", "vol_63", "parkinson_21", "vol_ratio",
    "volu_z20", "range_pct", "close_pos",
    "dow",
    "spy_ret_lag1", "vix_level", "vix_chg", "curve_slope",
]

# Optional add-on set for "with vs without technical indicators" experiments.
TECH_FEATURES = ["rsi_14", "macd_hist", "bb_pctb"]

TARGET_RET = "fwd_ret"  # next-day log return
TARGET_DIR = "fwd_up"   # 1 if next-day return > 0


def _ticker_features(df: pd.DataFrame, with_tech: bool) -> pd.DataFrame:
    c, o, h, l, v = df["close"], df["open"], df["high"], df["low"], df["volume"]
    r = np.log(c).diff()
    f = pd.DataFrame(index=df.index)

    for lag in (1, 2, 3, 5, 10, 21):
        f[f"ret_lag{lag}"] = r.shift(lag - 1)
    f["mom_21"] = np.log(c / c.shift(21))
    f["gap"] = np.log(o / c.shift(1))
    f["intraday"] = np.log(c / o)
    f["px_ma20"] = c / c.rolling(20).mean() - 1
    f["dist_hi252"] = c / c.rolling(252).max() - 1
    f["vol_5"] = r.rolling(5).std()
    f["vol_63"] = r.rolling(63).std()
    f["parkinson_21"] = np.sqrt(
        (np.log(h / l) ** 2).rolling(21).mean() / (4 * np.log(2))
    )
    f["vol_ratio"] = f["vol_5"] / f["vol_63"]
    f["volu_z20"] = (v - v.rolling(20).mean()) / v.rolling(20).std()
    f["range_pct"] = (h - l) / c
    f["close_pos"] = ((c - l) / (h - l)).where(h > l)
    f["dow"] = df.index.dayofweek

    if with_tech:
        delta = c.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        f["rsi_14"] = 100 - 100 / (1 + gain / loss)
        ema12 = c.ewm(span=12, adjust=False).mean()
        ema26 = c.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        f["macd_hist"] = macd - macd.ewm(span=9, adjust=False).mean()
        ma20, sd20 = c.rolling(20).mean(), c.rolling(20).std()
        f["bb_pctb"] = (c - (ma20 - 2 * sd20)) / (4 * sd20)

    f[TARGET_RET] = r.shift(-1)
    f[TARGET_DIR] = (f[TARGET_RET] > 0).astype(int)
    return f


def build_dataset(
    frames: dict[str, pd.DataFrame] | None = None,
    tickers: list[str] | None = None,
    with_tech: bool = False,
) -> pd.DataFrame:
    """Pooled (date-sorted) dataset across tickers with context features.

    Context series are reindexed to each equity's calendar so that bond/VIX
    holiday mismatches (see EDA §1) cannot shift rows.
    """
    from platform_core.config import TICKERS

    tickers = tickers or TICKERS
    if frames is None:
        frames = {t: load_cached(t) for t in tickers}
    spy = load_cached("SPY")
    vix = load_cached("^VIX")
    tnx = load_cached("^TNX")
    irx = load_cached("^IRX")
    spy_ret = np.log(spy["close"]).diff()
    curve = (tnx["close"] - irx["close"]).reindex(spy.index).ffill(limit=2)

    parts = []
    for ticker in tickers:
        df = frames[ticker].dropna()  # drops the unsettled trailing row
        f = _ticker_features(df, with_tech)
        f["spy_ret_lag1"] = spy_ret.reindex(f.index)
        f["vix_level"] = vix["close"].reindex(f.index).ffill(limit=2)
        f["vix_chg"] = vix["close"].diff().reindex(f.index).ffill(limit=2)
        f["curve_slope"] = curve.reindex(f.index)
        f["ticker"] = ticker
        parts.append(f)

    cols = FEATURES + (TECH_FEATURES if with_tech else [])
    out = pd.concat(parts)
    # rows lacking target (last day) or with incomplete warm-up windows are unusable
    out = out.dropna(subset=cols + [TARGET_RET])
    return out.sort_index(kind="stable")[["ticker"] + cols + [TARGET_RET, TARGET_DIR]]


if __name__ == "__main__":
    ds = build_dataset()
    print(ds.tail())
    print(f"\n{len(ds)} rows, {ds['ticker'].nunique()} tickers, "
          f"{ds.index.min():%Y-%m-%d} → {ds.index.max():%Y-%m-%d}")
    print(f"up-day share: {ds[TARGET_DIR].mean():.3f}")
