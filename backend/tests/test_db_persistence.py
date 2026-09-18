"""Phase 6 tests: PostgreSQL/PostGIS persistence.

Two groups:

1. Degradation tests that run everywhere (no database required): the API must
   never fabricate persisted results when PostgreSQL is unreachable.
2. Full persistence tests that run only against a real PostGIS database. They
   auto-skip when ``TERRAGUARD_TEST_DATABASE_URL`` (or the default dev URL) is
   unreachable â€” e.g. ``docker compose up -d postgis`` has not been run. They
   use an isolated ``terraguard_test`` database and never touch dev data.
"""

from __future__ import annotations

import os
import uuid
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

# ---------------------------------------------------------------------------
# Group 1 â€” no-database degradation behaviour (always run)
# ---------------------------------------------------------------------------


def test_health_reports_database_status() -> None:
    data = client.get("/api/health").json()
    assert data["database"] in {"ok", "unreachable", "connected_postgis_missing"}


def test_get_analysis_returns_503_without_database() -> None:
    from app.db import database_available

    if database_available():
        pytest.skip("database reachable; degradation path not exercised here")
    response = client.get(f"/api/analysis/{uuid.uuid4()}")
    assert response.status_code == 503
    assert "unavailable" in response.json()["detail"].lower()


def test_list_analyses_returns_503_without_database() -> None:
    from app.db import database_available

    if database_available():
        pytest.skip("database reachable; degradation path not exercised here")
    response = client.get("/api/analyses")
    assert response.status_code == 503


def test_analyze_without_database_sets_persistence_note() -> None:
    """The pipeline must still run when PostgreSQL is down â€” and say so honestly."""
    from unittest.mock import patch

    from app.db import database_available

    payload = {
        "aoi": {"min_lon": 73.75, "min_lat": 18.45, "max_lon": 73.95, "max_lat": 18.60},
        "before_date": "2023-01-01",
        "after_date": "2023-06-01",
    }
    with patch(
        "app.services.analysis.load_sentinel_pair", side_effect=FileNotFoundError("no data")
    ):
        response = client.post("/api/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "no_data"
    if not database_available():
        assert data["persistence_note"] is not None
        assert "unavailable" in data["persistence_note"]


# ---------------------------------------------------------------------------
# Group 2 â€” real PostGIS persistence (auto-skipped without a database)
# ---------------------------------------------------------------------------

TEST_DB_NAME = "terraguard_test"


def _test_database_url() -> str:
    url = os.environ.get("TERRAGUARD_TEST_DATABASE_URL")
    if url:
        return url
    from app.config import settings

    # Derive an isolated database name on the same server as the dev URL.
    return settings.database_url.rsplit("/", 1)[0] + f"/{TEST_DB_NAME}"


def _admin_url(url: str, connect_timeout: int = 3) -> str:
    """Admin-database URL on the same server, always bounded by connect_timeout."""
    from sqlalchemy import make_url

    u = make_url(url).set(database="postgres")
    if "connect_timeout" not in (u.query or {}):
        u = u.update_query_dict({"connect_timeout": str(connect_timeout)})
    return u.render_as_string(hide_password=False)


def _server_available(url: str) -> bool:
    from sqlalchemy import create_engine

    try:
        engine = create_engine(_admin_url(url), pool_pre_ping=True)
        with engine.connect():
            engine.dispose()
        return True
    except Exception:
        return False


def _paths():
    from pathlib import Path

    backend = Path(__file__).resolve().parents[1]
    return backend / "alembic.ini", backend / "alembic"


_DB_URL = _test_database_url()
requires_postgis = pytest.mark.skipif(
    not _server_available(_DB_URL),
    reason="PostgreSQL not reachable; run 'docker compose up -d postgis' to enable",
)


def _make_analysis_response(analysis_id: str) -> Any:
    """Build a complete fixture response. Values are test fixtures, not measured data."""
    from app.schemas.analysis import (
        AOISchema,
        AnalysisResponse,
        AnalysisStatus,
        ArtifactSchema,
        ChangeDetectionSummary,
        EnvironmentalFeaturesSummary,
        LandscapeMetrics,
        ObservationSummary,
        PriorityLabel,
        PrioritySummary,
        SHAPDriverSchema,
        VegetationMetrics,
    )

    return AnalysisResponse(
        analysis_id=analysis_id,
        status=AnalysisStatus.COMPLETE,
        created_at="2026-09-18T00:00:00+00:00",
        processing_time_s=1.234,
        aoi=AOISchema(min_lon=73.75, min_lat=18.45, max_lon=73.95, max_lat=18.60),
        before_observation=ObservationSummary(
            date="2023-12-15", source="Sentinel-2 L2A", bands=["B02", "B03", "B04", "B08"],
            crs="EPSG:32643", pixel_resolution_m=10.0, cloud_fraction=0.01,
        ),
        after_observation=ObservationSummary(
            date="2024-05-03", source="Sentinel-2 L2A", bands=["B02", "B03", "B04", "B08"],
            crs="EPSG:32643", pixel_resolution_m=10.0, cloud_fraction=0.02,
        ),
        change_detection=ChangeDetectionSummary(
            total_pixels=1_050_625, valid_pixels=1_049_099, valid_fraction=0.998548,
            changed_pixels=1_000, changed_area_m2=100_000.0, changed_area_ha=10.0,
            changed_area_km2=0.1, change_fraction_of_valid=0.000953, threshold_used=0.5,
            model_checkpoint="siamese_unet.pth", crs="EPSG:32643",
        ),
        environmental_features=EnvironmentalFeaturesSummary(
            total_area_ha=10_506.25, valid_area_ha=10_490.99,
            total_area_km2=105.0625, valid_area_km2=104.9099,
            vegetation=VegetationMetrics(
                ndvi_before_mean=0.61, ndvi_after_mean=0.43, ndvi_diff_mean=-0.18,
                change_ndvi_diff_mean=-0.2, veg_loss_ha_01=5.0, veg_loss_ha_02=4.0,
                veg_loss_ha_03=2.0, veg_gain_ha_02=0.5, veg_loss_fraction=0.0038,
            ),
            landscape=LandscapeMetrics(
                patch_count=12, mean_patch_area_ha=0.83, max_patch_area_ha=2.5,
                patch_density_per_km2=0.114, large_patch_count=3,
            ),
            scl_veg_loss_ha=3.2,
            feature_vector_length=25,
        ),
        priority_assessment=PrioritySummary(
            priority=PriorityLabel.HIGH, confidence=0.87,
            probabilities={"LOW": 0.05, "MEDIUM": 0.08, "HIGH": 0.87},
            top_drivers=[
                SHAPDriverSchema(feature="changed_area_ha", shap_value=0.21, feature_value=10.0),
                SHAPDriverSchema(feature="ndvi_diff_mean", shap_value=-0.14, feature_value=-0.18),
            ],
            label_provenance_notice="Prototype rule-derived labels, not expert ground truth.",
        ),
        artifacts=ArtifactSchema(
            analysis_id=analysis_id,
            change_probability_tif=f"/api/artifacts/{analysis_id}/change_probability.tif",
            change_mask_tif=f"/api/artifacts/{analysis_id}/change_mask.tif",
            inference_summary_json=f"/api/artifacts/{analysis_id}/inference_summary.json",
        ),
    )


@pytest.fixture()
def api_on_test_db(monkeypatch: pytest.MonkeyPatch):
    """Point the API router's persistence layer at the isolated test database."""
    import app.api.analysis as api_module
    from app.db import get_session_factory

    monkeypatch.setattr(api_module, "get_session_factory", lambda: get_session_factory(_DB_URL))
    monkeypatch.setattr(api_module, "database_available", lambda: True)


@pytest.fixture()
def db_session():
    """Isolated test database: create, migrate, yield a session, drop."""
    from alembic import command
    from alembic.config import Config
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import sessionmaker

    url = _DB_URL
    ini_path, script_dir = _paths()
    admin_url = _admin_url(url)
    admin = create_engine(admin_url, isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        exists = conn.execute(
            text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": TEST_DB_NAME}
        ).scalar()
        if exists:
            conn.execute(
                text(
                    "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                    "WHERE datname = :n AND pid <> pg_backend_pid()"
                ),
                {"n": TEST_DB_NAME},
            )
            conn.execute(text(f'DROP DATABASE "{TEST_DB_NAME}"'))
        conn.execute(text(f'CREATE DATABASE "{TEST_DB_NAME}"'))
    admin.dispose()

    cfg = Config(str(ini_path))
    cfg.set_main_option("script_location", str(script_dir))
    os.environ["ALEMBIC_DATABASE_URL"] = url
    try:
        command.upgrade(cfg, "head")
        factory = sessionmaker(bind=create_engine(url), expire_on_commit=False)
        session = factory()
        try:
            yield session
        finally:
            session.close()
    finally:
        os.environ.pop("ALEMBIC_DATABASE_URL", None)
        admin = create_engine(admin_url, isolation_level="AUTOCOMMIT")
        with admin.connect() as conn:
            conn.execute(
                text(
                    "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                    "WHERE datname = :n AND pid <> pg_backend_pid()"
                ),
                {"n": TEST_DB_NAME},
            )
            conn.execute(text(f'DROP DATABASE "{TEST_DB_NAME}"'))
        admin.dispose()


# ---------------------------------------------------------------------------
# Persistence tests (run only when PostGIS is reachable)
# ---------------------------------------------------------------------------


@requires_postgis
def test_postgis_extension_available(db_session) -> None:
    from sqlalchemy import text

    row = db_session.execute(
        text("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'postgis')")
    ).scalar()
    assert row is True


@requires_postgis
def test_migration_created_all_tables(db_session) -> None:
    from sqlalchemy import inspect, text

    inspector = inspect(db_session.connection())
    tables = set(inspector.get_table_names())
    expected = {
        "analysis_runs", "analysis_aois", "analysis_observations",
        "change_detection_results", "environmental_features",
        "priority_assessments", "shap_drivers", "artifacts",
    }
    assert expected <= tables
    geom_type = db_session.execute(
        text("SELECT type FROM geometry_columns "
             "WHERE f_table_name = 'analysis_aois' AND f_geometry_column = 'geom'")
    ).scalar()
    assert geom_type == "POLYGON"


@requires_postgis
def test_full_analysis_lifecycle_persists_everything(db_session) -> None:
    from sqlalchemy import text

    analysis_id = str(uuid.uuid4())
    create_run(db_session, analysis_id, {"before_date": "2023-12-15"})
    persist_result(db_session, _make_analysis_response(analysis_id))

    run = get_analysis(db_session, analysis_id)
    assert run is not None
    assert run.status == "complete"
    assert run.completed_at is not None
    assert len(run.observations) == 2
    assert {obs.role for obs in run.observations} == {"before", "after"}
    assert run.change_detection.changed_area_ha == 10.0
    assert run.change_detection.change_polygons_available is False
    assert run.change_detection.change_polygon is None
    feats = sorted(run.environmental_features, key=lambda f: f.position)
    assert [f.position for f in feats] == list(range(len(feats)))
    assert len(feats) == 20
    assert run.priority_assessment.priority == "HIGH"
    assert run.priority_assessment.label_provenance_notice is not None
    drivers = sorted(run.shap_drivers, key=lambda d: d.rank)
    assert [d.rank for d in drivers] == [1, 2]
    assert drivers[0].direction == "increases"
    assert drivers[1].direction == "decreases"
    assert len(run.artifacts) == 3

    wkt = db_session.execute(
        text("SELECT ST_AsText(geom) FROM analysis_aois WHERE analysis_id = :i"),
        {"i": analysis_id},
    ).scalar()
    assert wkt.startswith("POLYGON((73.75 18.45")
    srid = db_session.execute(
        text("SELECT ST_SRID(geom) FROM analysis_aois WHERE analysis_id = :i"),
        {"i": analysis_id},
    ).scalar()
    assert srid == 4326


@requires_postgis
def test_retrieval_endpoint_returns_persisted_analysis(db_session, api_on_test_db) -> None:
    analysis_id = str(uuid.uuid4())
    create_run(db_session, analysis_id, {})
    persist_result(db_session, _make_analysis_response(analysis_id))

    response = client.get(f"/api/analysis/{analysis_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["analysis_id"] == analysis_id
    assert data["status"] == "complete"
    assert data["change_detection"]["changed_area_ha"] == 10.0
    assert data["change_detection"]["change_polygons_available"] is False
    assert len(data["environmental_features"]) == 20
    assert data["environmental_features"][0]["name"] == "total_area_ha"


@requires_postgis
def test_failed_run_persists_safe_error(db_session, api_on_test_db) -> None:
    from app.schemas.analysis import AOISchema, AnalysisResponse, AnalysisStatus

    analysis_id = str(uuid.uuid4())
    create_run(db_session, analysis_id, {})
    failed = AnalysisResponse(
        analysis_id=analysis_id,
        status=AnalysisStatus.FAILED,
        created_at="2026-09-18T00:00:00+00:00",
        aoi=AOISchema(min_lon=73.75, min_lat=18.45, max_lon=73.95, max_lat=18.60),
        error="Model checkpoint not found at siamese_unet.pth. Train the Siamese U-Net first.",
    )
    persist_result(db_session, failed)

    run = get_analysis(db_session, analysis_id)
    assert run is not None
    assert run.status == "failed"
    assert run.error_message is not None and "checkpoint" in run.error_message
    assert run.aoi is not None  # AOI recorded even for failed runs
    assert run.change_detection is None
    assert run.priority_assessment is None

    data = client.get(f"/api/analysis/{analysis_id}").json()
    assert data["status"] == "failed"


@requires_postgis
def test_double_finalize_is_rejected(db_session) -> None:
    analysis_id = str(uuid.uuid4())
    create_run(db_session, analysis_id, {})
    persist_result(db_session, _make_analysis_response(analysis_id))
    with pytest.raises(ValueError, match="already finalized"):
        persist_result(db_session, _make_analysis_response(analysis_id))


@requires_postgis
def test_unknown_analysis_returns_404(db_session) -> None:
    response = client.get(f"/api/analysis/{uuid.uuid4()}")
    assert response.status_code == 404


@requires_postgis
def test_history_endpoint_lists_runs_with_pagination(db_session, api_on_test_db) -> None:
    ids = []
    for _ in range(3):
        analysis_id = str(uuid.uuid4())
        ids.append(analysis_id)
        create_run(db_session, analysis_id, {})
        persist_result(db_session, _make_analysis_response(analysis_id))

    response = client.get("/api/analyses?limit=2&offset=0")
    assert response.status_code == 200
    data = response.json()
    assert data["limit"] == 2 and data["offset"] == 0
    assert len(data["analyses"]) == 2
    assert data["total"] >= 3
    listed = {item["analysis_id"] for item in data["analyses"]}
    assert listed <= set(ids)


@requires_postgis
def test_transaction_rollback_leaves_no_rows(db_session) -> None:
    """Uncommitted ORM writes must vanish on rollback â€” no partial analysis rows."""
    from sqlalchemy import text

    from app.models import AnalysisRun

    analysis_id = str(uuid.uuid4())
    db_session.add(AnalysisRun(id=analysis_id, status="running", request_params={}))
    db_session.flush()
    count_running = db_session.execute(
        text("SELECT COUNT(*) FROM analysis_runs WHERE id = :i"), {"i": analysis_id}
    ).scalar()
    assert count_running == 1

    db_session.rollback()

    count = db_session.execute(
        text("SELECT COUNT(*) FROM analysis_runs WHERE id = :i"), {"i": analysis_id}
    ).scalar()
    assert count == 0