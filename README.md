# mlops-stock-platform

An end-to-end **ML platform** demo built on real MLflow: experiment tracking,
model registry with alias-based promotion, automated weekly retraining via
Airflow, and zero-downtime model serving with FastAPI — all running locally
with `docker compose up`.

The ML task (next-day stock direction) is deliberately simple and its edge is
deliberately reported honestly (~1–3pp over a coin flip). The deliverable is
the **platform engineering**, not alpha.

```
yfinance ──► data/cache ──► features ──► train ──► MLflow Tracking (experiments)
                                          │
                                          ▼
                                  MLflow Model Registry
                                  stock-predictor @production
                                          │
        Airflow DAG (weekly):             ▼
        fetch → decay check →      FastAPI  /predict /model-info /health
        train challenger →         (TTL poll on the alias → hot-swaps the
        promote if better          model when promotion moves the pointer)
```

## What it demonstrates

| Capability | How |
|---|---|
| Experiment tracking | every training run logs params/metrics/model to an MLflow server; comparison grid via `make experiments` |
| Model versioning | registry versions are immutable; `@production` alias is the single source of truth for serving |
| Promotion gate | challenger must beat the champion on the *same* current holdout (`should_promote`, unit-tested); ties go to the fresher model |
| Automated retraining | Airflow 3 TaskFlow DAG: `fetch → decay check → train → promote`, weekly, XCom carries only run IDs |
| Zero-downtime serving | FastAPI polls the alias on a 30s TTL and hot-reloads; promotion and rollback never touch the containers |
| Decay monitoring | rolling 60-day directional accuracy of the production model, trended in its own MLflow experiment |
| Leakage discipline | target/feature alignment, no-lookahead, and scale-freeness are unit tests, not hopes |

## Quickstart

```bash
# prerequisites: docker, uv
uv sync                       # local env (Python 3.12)
docker compose up -d          # mlflow (5001), airflow (8081), api (8000), postgres

make experiments              # 5 comparable runs in MLflow UI
uv run python -m platform_core.promote     # train challenger, maybe promote

curl "localhost:8000/predict?ticker=AAPL"
curl  localhost:8000/model-info             # which version is serving right now

# trigger the weekly pipeline by hand
docker compose exec airflow airflow dags trigger weekly_retrain
```

- MLflow UI: http://127.0.0.1:5001
- Airflow UI: http://127.0.0.1:8081 (auth disabled — local demo only)
- API docs: http://127.0.0.1:8000/docs

Committed OHLCV snapshots under `data/cache/` make everything above work
offline; the fetch step refreshes them when the network allows.

## Design choices (and trade-offs)

- **Aliases, not stages** — model stages are deprecated in MLflow ≥2.9;
  an alias is an O(1) atomic pointer, so promotion *and rollback* are one call.
- **Register every challenger, promote selectively** — losing versions stay
  in the registry as an audit trail of why production didn't change.
- **Champion and challenger are rescored on the same holdout** — historical
  scores from different data windows are not comparable.
- **Walk-forward CV split by date with a 5-day purge gap** — pooling tickers
  means rows share dates; row-wise splits would leak same-day information.
- **Honest baseline gating** — ~53% of days are up, so "always predict up"
  is the accuracy floor every model must clear; metrics log the edge, not
  just the score.
- **No macro release data** — monthly frequency + publication lags/revisions
  make point-in-time alignment a leakage trap; daily market-traded proxies
  (yield-curve slope, VIX) carry the macro signal instead.
- **Thin data layer by design** — heavy ETL/data-quality engineering lives in
  a separate project; this repo spends its complexity budget on the model
  lifecycle.

## Layout

```
src/platform_core/  config · data · features · train · evaluate · promote · monitor
dags/               weekly_retrain (Airflow 3 TaskFlow)
app/                FastAPI serving
scripts/eda.py      the EDA gate that finalized the feature set
tests/              leakage guards · promotion rule · API smoke
```

## Tests

```bash
uv run pytest        # 30 tests: leakage invariants, promotion logic, API
```
