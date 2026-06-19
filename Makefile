.PHONY: setup data eda train experiments test mlflow-up mlflow-down up down promote monitor serve dashboard

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

up:               ## full stack: mlflow + airflow + api
	docker compose up -d

down:
	docker compose down

promote:          ## train a challenger and run the promotion gate
	uv run python -m platform_core.promote

monitor:          ## decay check on the production model
	uv run python -m platform_core.monitor

serve:            ## run the API locally (outside docker)
	uv run uvicorn app.main:app --port 8000

dashboard:        ## unified Streamlit dashboard (http://localhost:8501)
	uv run streamlit run app/dashboard.py --server.port 8501

test:
	uv run pytest -q
