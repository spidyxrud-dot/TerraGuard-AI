"""API-level tests for Phase 5 FastAPI endpoints.

Tests cover:
- GET /api/health — liveness, version, model availability flags
- POST /api/analyze — valid request schema, invalid inputs, graceful errors
- GET /api/artifacts — safe serving, path traversal rejection
- Response schema correctness (no fabricated fields)
"""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

def test_health_ok() -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "models_loaded" in data
    assert "timestamp" in data
    assert isinstance(data["models_loaded"], dict)


def test_health_models_loaded_is_boolean_map() -> None:
    response = client.get("/api/health")
    data = response.json()
    for key, val in data["models_loaded"].items():
        assert isinstance(val, bool), f"models_loaded['{key}'] should be bool"


# ---------------------------------------------------------------------------
# POST /api/analyze — schema validation
# ---------------------------------------------------------------------------

VALID_PUNE_REQUEST: dict[str, Any] = {
    "aoi": {
        "min_lon": 73.75,
        "min_lat": 18.45,
        "max_lon": 73.95,
        "max_lat": 18.60,
    },
    "before_date": "2023-01-01",
    "after_date": "2023-06-01",
    "parameters": {
        "change_threshold": 0.5,
        "scl_policy": "strict",
        "pixel_resolution_m": 10.0,
    },
}


def test_analyze_invalid_aoi_lon_reversed() -> None:
    payload = {**VALID_PUNE_REQUEST, "aoi": {
        "min_lon": 74.0, "min_lat": 18.45, "max_lon": 73.75, "max_lat": 18.60,
    }}
    response = client.post("/api/analyze", json=payload)
    assert response.status_code == 422
    errors = response.json()["detail"]
    messages = [e["msg"] for e in errors]
    assert any("min_lon" in m for m in messages)


def test_analyze_invalid_aoi_lat_reversed() -> None:
    payload = {**VALID_PUNE_REQUEST, "aoi": {
        "min_lon": 73.75, "min_lat": 18.60, "max_lon": 73.95, "max_lat": 18.45,
    }}
    response = client.post("/api/analyze", json=payload)
    assert response.status_code == 422


def test_analyze_dates_reversed() -> None:
    payload = {**VALID_PUNE_REQUEST, "before_date": "2023-06-01", "after_date": "2023-01-01"}
    response = client.post("/api/analyze", json=payload)
    assert response.status_code == 422


def test_analyze_same_dates() -> None:
    payload = {**VALID_PUNE_REQUEST, "before_date": "2023-01-01", "after_date": "2023-01-01"}
    response = client.post("/api/analyze", json=payload)
    assert response.status_code == 422


def test_analyze_invalid_threshold_out_of_range() -> None:
    payload = {**VALID_PUNE_REQUEST,
               "parameters": {"change_threshold": 1.5, "scl_policy": "strict", "pixel_resolution_m": 10.0}}
    response = client.post("/api/analyze", json=payload)
    assert response.status_code == 422


def test_analyze_invalid_scl_policy() -> None:
    payload = {**VALID_PUNE_REQUEST,
               "parameters": {"change_threshold": 0.5, "scl_policy": "unknown", "pixel_resolution_m": 10.0}}
    response = client.post("/api/analyze", json=payload)
    assert response.status_code == 422


def test_analyze_missing_required_fields() -> None:
    response = client.post("/api/analyze", json={})
    assert response.status_code == 422


def test_analyze_aoi_out_of_bounds_lat() -> None:
    payload = {**VALID_PUNE_REQUEST, "aoi": {
        "min_lon": -10.0, "min_lat": -95.0, "max_lon": 10.0, "max_lat": 10.0,
    }}
    response = client.post("/api/analyze", json=payload)
    assert response.status_code == 422


def test_analyze_returns_analysis_id() -> None:
    """When data is unavailable, we should still get a JSON response with analysis_id, not a 500."""
    with patch("app.services.analysis.load_sentinel_pair", side_effect=FileNotFoundError("no data")):
        response = client.post("/api/analyze", json=VALID_PUNE_REQUEST)
    assert response.status_code == 200
    data = response.json()
    assert "analysis_id" in data
    assert re.match(r"[0-9a-f\-]{36}", data["analysis_id"])
    assert data["status"] == "no_data"


def test_analyze_no_data_returns_error_field() -> None:
    """Pipeline with missing data returns status=no_data and non-empty error."""
    with patch("app.services.analysis.load_sentinel_pair", side_effect=FileNotFoundError("demo pair not found")):
        response = client.post("/api/analyze", json=VALID_PUNE_REQUEST)
    data = response.json()
    assert data["status"] == "no_data"
    assert data["error"] is not None
    assert len(data["error"]) > 0


def test_analyze_pipeline_error_returns_failed_status() -> None:
    """An unexpected inference error returns status=failed gracefully."""
    from app.services.preprocessing import Observation

    # Fake a minimal observation with required attributes
    fake_band = MagicMock()
    fake_band.crs = "EPSG:32643"
    fake_band.path = Path("/fake/B02.tif")
    fake_obs = MagicMock(spec=Observation)
    fake_obs.date = "2023-01-01"
    fake_obs.cloud_cover = 0.02
    fake_obs.bands = {"B02": fake_band, "B03": fake_band, "B04": fake_band, "B08": fake_band}

    with patch("app.services.analysis.load_sentinel_pair", return_value=(fake_obs, fake_obs)):
        with patch("app.services.analysis.prepare_pair", return_value={}):
            with patch("app.services.analysis.run_inference",
                       side_effect=RuntimeError("GPU OOM")):
                response = client.post("/api/analyze", json=VALID_PUNE_REQUEST)

    data = response.json()
    assert data["status"] == "failed"
    assert "error" in data
    assert data["error"] is not None


def test_analyze_no_fabricated_results_without_data() -> None:
    """When pipeline errors, change_detection and environmental_features must be absent."""
    with patch("app.services.analysis.load_sentinel_pair", side_effect=FileNotFoundError("no demo pair")):
        response = client.post("/api/analyze", json=VALID_PUNE_REQUEST)
    data = response.json()
    assert data["change_detection"] is None
    assert data["environmental_features"] is None
    assert data["priority_assessment"] is None


def test_analyze_aoi_echoed_in_response() -> None:
    with patch("app.services.analysis.load_sentinel_pair", side_effect=FileNotFoundError("no data")):
        response = client.post("/api/analyze", json=VALID_PUNE_REQUEST)
    data = response.json()
    aoi = data["aoi"]
    assert aoi["min_lon"] == pytest.approx(73.75)
    assert aoi["min_lat"] == pytest.approx(18.45)
    assert aoi["max_lon"] == pytest.approx(73.95)
    assert aoi["max_lat"] == pytest.approx(18.60)


# ---------------------------------------------------------------------------
# GET /api/artifacts — safe file serving
# ---------------------------------------------------------------------------

def test_artifact_invalid_analysis_id_rejected() -> None:
    response = client.get("/api/artifacts/../../../etc/passwd/change_mask.tif")
    assert response.status_code in {400, 404, 422}


def test_artifact_disallowed_extension_rejected() -> None:
    response = client.get("/api/artifacts/some-valid-id/../../secret.sh")
    assert response.status_code in {400, 403, 404, 422}


def test_artifact_not_found_returns_404(tmp_path: Path) -> None:
    # analysis_id that exists as a directory but not the requested file
    valid_id = "abcd1234-abcd-abcd-abcd-123456789012"
    response = client.get(f"/api/artifacts/{valid_id}/change_mask.tif")
    assert response.status_code == 404


def test_artifact_valid_json_served(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Serve a real JSON artifact through the endpoint."""
    from app.services.analysis import DEFAULT_PREDICTIONS_DIR

    analysis_id = "aaaabbbb-cccc-dddd-eeee-000011112222"
    art_dir = tmp_path / analysis_id
    art_dir.mkdir(parents=True)
    (art_dir / "inference_summary.json").write_text(
        json.dumps({"test": True}), encoding="utf-8"
    )

    # Patch DEFAULT_PREDICTIONS_DIR in both the router and service modules
    import app.api.analysis as analysis_router_module
    import app.services.analysis as analysis_service_module
    monkeypatch.setattr(analysis_router_module, "DEFAULT_PREDICTIONS_DIR", tmp_path)
    monkeypatch.setattr(analysis_service_module, "DEFAULT_PREDICTIONS_DIR", tmp_path)

    response = client.get(f"/api/artifacts/{analysis_id}/inference_summary.json")
    assert response.status_code == 200
    assert response.json() == {"test": True}
