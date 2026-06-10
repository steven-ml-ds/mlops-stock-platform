.PHONY: setup data eda train experiments test mlflow-up mlflow-down

setup:            ## install deps into .venv
	uv sync

data:             ## refresh OHLCV cache from yfinance
	uv run python -m platform_core.data

eda:              ## regenerate EDA charts + summary
	uv run python scripts/eda.py

mlflow-up:        ## start MLflow tracking server (http://127.0.0.1:5001)
	docker compose up -d mlflow

mlflow-down:
	docker compose down

train:            ## one default training run logged to MLflow
	uv run python -m platform_core.train --run-name manual

experiments:      ## the comparison grid from docs/notes/03
	uv run python -m platform_core.train --run-name base
	uv run python -m platform_core.train --run-name lr02-deep --learning-rate 0.02 --num-leaves 31 --n-estimators 600
	uv run python -m platform_core.train --run-name lr10-shallow --learning-rate 0.10 --num-leaves 7 --n-estimators 150
	uv run python -m platform_core.train --run-name shallow-slow --learning-rate 0.03 --num-leaves 7 --n-estimators 400
	uv run python -m platform_core.train --run-name base-tech --with-tech

test:
	uv run pytest -q
