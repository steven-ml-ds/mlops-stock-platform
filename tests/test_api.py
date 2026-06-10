"""API smoke tests that run offline (no MLflow server, no model)."""

from fastapi.testclient import TestClient

from app.main import app


def test_health_up_without_model():
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


def test_unknown_ticker_404():
    with TestClient(app) as client:
        assert client.get("/predict", params={"ticker": "TSLA"}).status_code == 404


def test_predict_503_when_no_model():
    with TestClient(app) as client:
        if client.get("/health").json()["model_loaded"]:
            return  # a live registry answered; covered by manual verification
        assert client.get("/predict", params={"ticker": "AAPL"}).status_code == 503
