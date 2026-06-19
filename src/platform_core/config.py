"""Central configuration for the platform."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "cache"

# Equities we model: a sector-diverse, high-liquidity slice of the S&P 500. The upstream
# stock-etl-pipeline publishes the whole index, so widening this list is config-only — the
# clean OHLCV is already in its MinIO store (see etl_source). Context series carry
# market-wide signal (broad-market return, volatility, the yield curve).
TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA",  # tech / consumer internet
    "JPM", "BAC",                                              # financials
    "JNJ", "UNH",                                              # health care
    "XOM", "CVX",                                              # energy
    "PG", "KO",                                                # staples
    "HD",                                                      # consumer discretionary
    "CAT",                                                     # industrials
    "VZ",                                                      # communications
    "NEE",                                                     # utilities
    "LIN",                                                     # materials
]
CONTEXT_TICKERS = ["SPY", "^VIX", "^TNX", "^IRX"]

HISTORY_YEARS = 5

# Upstream data source: the stock-etl-pipeline MinIO store (S3-compatible). When reachable,
# fetch_all reads clean, DQ-gated OHLCV from here instead of hitting yfinance directly. The
# defaults match the ETL stack; override via env (and point ENDPOINT at the shared compose
# network's "minio:9000" when running containerised, or "localhost:9000" from the host).
MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ROOT_USER", "minioadmin")
MINIO_SECRET_KEY = os.environ.get("MINIO_ROOT_PASSWORD", "minioadmin123")
MINIO_BUCKET = os.environ.get("MINIO_BUCKET", "stock-data")
MINIO_SECURE = os.environ.get("MINIO_SECURE", "false").lower() == "true"

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
EXPERIMENT_NAME = "stock-direction"
REGISTERED_MODEL_NAME = "stock-predictor"
PRODUCTION_ALIAS = "production"
CHALLENGER_ALIAS = "challenger"  # shadow model: scored on live traffic, never answers
SHADOW_LOG = PROJECT_ROOT / "data" / "predictions" / "shadow.jsonl"
