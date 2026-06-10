"""Central configuration for the platform."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "cache"

# Equities we model; context series carry market-wide signal.
TICKERS = ["AAPL", "MSFT", "NVDA"]
CONTEXT_TICKERS = ["SPY", "^VIX", "^TNX", "^IRX"]

HISTORY_YEARS = 5

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
EXPERIMENT_NAME = "stock-direction"
REGISTERED_MODEL_NAME = "stock-predictor"
PRODUCTION_ALIAS = "production"
