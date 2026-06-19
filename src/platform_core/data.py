"""Fetch daily OHLCV from yfinance with a committed CSV cache as offline fallback.

The cache serves two purposes:
1. Reproducibility — the repo trains end-to-end without network access.
2. Resilience — if Yahoo Finance is down or rate-limits, we fall back to the
   last good snapshot instead of failing the pipeline.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd
import yfinance as yf

from platform_core.config import CONTEXT_TICKERS, DATA_DIR, HISTORY_YEARS, TICKERS

log = logging.getLogger(__name__)

COLUMNS = ["open", "high", "low", "close", "volume"]


def _cache_path(ticker: str):
    return DATA_DIR / f"{ticker.replace('^', '_IDX_')}.csv"


def fetch_ticker(ticker: str, years: int = HISTORY_YEARS) -> pd.DataFrame:
    """Download adjusted daily OHLCV; on failure fall back to cached CSV.

    auto_adjust=True folds splits/dividends into OHLC so downstream features
    never see artificial price jumps.
    """
    start = date.today() - timedelta(days=int(years * 365.25))
    try:
        raw = yf.download(
            ticker, start=start.isoformat(), auto_adjust=True, progress=False
        )
        if raw is None or raw.empty:
            raise ValueError(f"empty frame for {ticker}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        df = raw.rename(columns=str.lower)[COLUMNS].copy()
        df.index.name = "date"
        _cache_path(ticker).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(_cache_path(ticker))
        log.info("fetched %s: %d rows -> cached", ticker, len(df))
        return df
    except Exception as exc:  # network, rate-limit, schema drift
        log.warning("download failed for %s (%s); trying cache", ticker, exc)
        return load_cached(ticker)


def load_cached(ticker: str) -> pd.DataFrame:
    path = _cache_path(ticker)
    if not path.exists():
        raise FileNotFoundError(f"no cache for {ticker} at {path}")
    return pd.read_csv(path, index_col="date", parse_dates=True)


def fetch_all(refresh: bool = True) -> dict[str, pd.DataFrame]:
    """Return {ticker: ohlcv} for all model + context tickers.

    Source priority when ``refresh``:
      1. the upstream stock-etl-pipeline MinIO store (clean, DQ-gated, adjusted) —
      2. else per-ticker yfinance with the committed CSV cache as fallback.

    Whatever the source, every frame is written back to the CSV cache so ``load_cached``
    (used for context series in features.build_dataset) and offline reproducibility keep
    working without an ETL stack.
    """
    all_tickers = TICKERS + CONTEXT_TICKERS
    if not refresh:
        return {t: load_cached(t) for t in all_tickers}

    from platform_core import etl_source

    frames = etl_source.load_from_etl(all_tickers)
    if frames is not None:
        for ticker, df in frames.items():
            df.to_csv(_cache_path(ticker))  # materialise the cache mirror
        log.info("fetched %d tickers from ETL MinIO -> cached", len(frames))
        return frames

    log.info("ETL source unavailable; falling back to yfinance/CSV per ticker")
    return {t: fetch_ticker(t) for t in all_tickers}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    frames = fetch_all(refresh=True)
    for name, frame in frames.items():
        print(f"{name:8s} {len(frame):5d} rows  {frame.index.min():%Y-%m-%d} → {frame.index.max():%Y-%m-%d}")
