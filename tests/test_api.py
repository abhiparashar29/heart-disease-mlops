"""Tests for the FastAPI serving layer."""
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client(trained_model):
    with TestClient(app) as c:  # 'with' triggers model loading at startup
        yield c


def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "/predict" in r.json()["endpoints"]


def test_health_reports_model_loaded(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_predict_returns_prediction_and_confidence(client, sample_patient):
    r = client.post("/predict", json=sample_patient)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0.5 <= body["confidence"] <= 1.0
    assert 0.0 <= body["probability_disease"] <= 1.0
    assert body["risk_level"] in ("low", "moderate", "high")


def test_predict_accepts_missing_optional_fields(client, sample_patient):
    patient = {k: v for k, v in sample_patient.items() if k not in ("ca", "thal")}
    assert client.post("/predict", json=patient).status_code == 200


def test_predict_rejects_out_of_range_value(client, sample_patient):
    assert client.post("/predict", json={**sample_patient, "age": -5}).status_code == 422


def test_predict_rejects_missing_required_field(client, sample_patient):
    patient = {k: v for k, v in sample_patient.items() if k != "chol"}
    assert client.post("/predict", json=patient).status_code == 422


def test_predict_rejects_invalid_thal(client, sample_patient):
    assert client.post("/predict", json={**sample_patient, "thal": 5}).status_code == 422


def test_metrics_endpoint_exposes_custom_metrics(client, sample_patient):
    client.post("/predict", json=sample_patient)
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "heart_predictions_total" in r.text


def test_responses_include_request_id(client):
    assert "x-request-id" in client.get("/").headers
