"""Consume the upstream stock-etl-pipeline's clean OHLCV from its MinIO store.

The sibling stock-etl-pipeline project is the data platform: it fetches the S&P 500 via
Airflow, runs quality gates (OHLC sanity, freshness), and publishes split/dividend-adjusted
OHLCV to a MinIO bucket as date-partitioned Parquet
(``ohlcv/market={market}/dt={date}/part.parquet``). This module reads that **published
contract** — no cross-repo import — and reshapes it into the per-ticker wide frames the rest
of this project already speaks (``{ticker: DataFrame(index=date, cols=OHLCV)}``).

Price alignment: the ETL stores raw ``close`` plus ``adj_close``. This project historically
fetched yfinance with ``auto_adjust=True`` (all of OHLC adjusted), so we fold the adjustment
factor ``adj_close / close`` back into open/high/low and serve ``adj_close`` as ``close`` —
preserving the "no split/dividend jumps" invariant the leakage tests assume.

Returns ``None`` (rather than raising) whenever MinIO is unreachable or a requested ticker is
absent, so ``data.fetch_all`` can fall back to the committed CSV cache / yfinance.
"""

from __future__ import annotations

import io
import logging

import pandas as pd

from platform_core.config import (
    MINIO_ACCESS_KEY,
    MINIO_BUCKET,
    MINIO_ENDPOINT,
    MINIO_SECRET_KEY,
    MINIO_SECURE,
)

log = logging.getLogger(__name__)

COLUMNS = ["open", "high", "low", "close", "volume"]


def _read_all_partitions() -> pd.DataFrame | None:
    """Concatenate every ``ohlcv/`` Parquet partition from MinIO; None if unreachable/empty."""
    try:
        from minio import Minio
    except ImportError:
        log.warning("minio SDK not installed; skipping ETL source")
        return None
    try:
        s3 = Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=MINIO_SECURE,
        )
        objects = list(s3.list_objects(MINIO_BUCKET, prefix="ohlcv/", recursive=True))
        if not objects:
            return None
        frames = []
        for obj in objects:
            resp = s3.get_object(MINIO_BUCKET, obj.object_name)
            frames.append(pd.read_parquet(io.BytesIO(resp.read())))
            resp.close()
            resp.release_conn()
        return pd.concat(frames, ignore_index=True)
    except Exception as exc:  # noqa: BLE001 — any connection/SDK error → caller falls back
        log.warning("ETL MinIO read failed (%s); will fall back", exc)
        return None


def _to_wide(group: pd.DataFrame) -> pd.DataFrame:
    """One ticker's long rows → date-indexed OHLCV, with adjustment folded into OHL."""
    g = group.dropna(subset=["close", "adj_close"]).copy()
    g["date"] = pd.to_datetime(g["date"])
    g = g.drop_duplicates("date").sort_values("date").set_index("date")
    g.index.name = "date"
    # Fold the split/dividend adjustment into open/high/low; serve adj_close as close.
    factor = g["adj_close"] / g["close"]
    out = pd.DataFrame(index=g.index)
    out["open"] = g["open"] * factor
    out["high"] = g["high"] * factor
    out["low"] = g["low"] * factor
    out["close"] = g["adj_close"]
    out["volume"] = g["volume"]
    return out[COLUMNS]


def load_from_etl(tickers: list[str]) -> dict[str, pd.DataFrame] | None:
    """Return ``{ticker: wide adjusted OHLCV}`` for all ``tickers``, or None on any miss.

    "All or nothing": if MinIO is unreachable or any requested ticker is missing, returns
    None so the caller falls back to a complete alternative source rather than training on a
    partial universe.
    """
    raw = _read_all_partitions()
    if raw is None or raw.empty:
        return None

    raw = raw.drop_duplicates(["ticker", "date"], keep="last")
    available = set(raw["ticker"].unique())
    missing = [t for t in tickers if t not in available]
    if missing:
        log.warning("ETL source missing %d/%d tickers (%s…); falling back",
                    len(missing), len(tickers), ", ".join(missing[:5]))
        return None

    out: dict[str, pd.DataFrame] = {}
    for ticker, group in raw[raw["ticker"].isin(tickers)].groupby("ticker"):
        out[ticker] = _to_wide(group)
    log.info("Loaded %d tickers from ETL MinIO (%s)", len(out), MINIO_ENDPOINT)
    return out
