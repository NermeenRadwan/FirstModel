"""Automated tests for FastAPI endpoints."""

from unittest.mock import patch

import pytest

# pyrefly: ignore[missing-import]
from fastapi.testclient import TestClient

from production_app.main import app
from production_app.schemas import EXAMPLE_FEATURES_DICT


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_health_endpoint(client: TestClient):
    """Requirement 02: GET /health returns healthy when model is available."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict_valid_input(client: TestClient):
    """Requirement 03: POST /predict returns HTTP 200 and structured label."""
    response = client.post("/predict", json={"features": EXAMPLE_FEATURES_DICT})
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in (0, 1)
    assert data["label"] in ("benign", "malignant")


def test_predict_invalid_input(client: TestClient):
    """Requirement 04: Missing or invalid features are rejected with HTTP 422."""
    # Missing features
    res_missing = client.post("/predict", json={"features": {"mean texture": 10.0}})
    assert res_missing.status_code == 422

    # Malformed features (string instead of dict/list)
    res_malformed = client.post("/predict", json={"features": "invalid"})
    assert res_malformed.status_code == 422


def test_model_missing_reports_503(client: TestClient):
    """Requirement 05: Reports HTTP 503 cleanly when model is unavailable."""
    with (
        patch("production_app.main.model", None),
        patch("production_app.main.get_loaded_model", return_value=None),
    ):
        response = client.post("/predict", json={"features": EXAMPLE_FEATURES_DICT})
        assert response.status_code == 503
