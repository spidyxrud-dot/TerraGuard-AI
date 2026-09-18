"""Persistence repository for analysis runs (Phase 6).

Maps the pipeline's :class:`AnalysisResponse` onto the normalized schema without
duplicating any pipeline logic. A missing or unreachable database must never corrupt
an analysis result — persistence failures are reported, not swallowed silently.
"""

from __future__ import annotations

import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_DIR = REPO_ROOT / "backend"
for _entry in (BACKEND_DIR, REPO_ROOT):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import (
    AnalysisAoi,
    AnalysisObservation,
    AnalysisRun,
    Artifact,
    ChangeDetectionResult,
    EnvironmentalFeature,
    PriorityAssessment,
    ShapDriver,
)
from app.schemas.analysis import AnalysisResponse

try:  # optional at import time; keeps the repository usable without the ML stack
    from ml.priority.explain import FEATURE_DESCRIPTIONS as _FD

    FEATURE_DESCRIPTIONS: dict[str, str] = dict(_FD)
except Exception:  # pragma: no cover - explain module import depends on shap
    FEATURE_DESCRIPTIONS = {}


def _wkt_polygon(min_lon: float, min_lat: float, max_lon: float, max_lat: float) -> str:
    """Axis-aligned WGS-84 bounding-box polygon in WKT (lon lat ordering)."""
    return (
        f"POLYGON(({min_lon} {min_lat}, {max_lon} {min_lat}, "
        f"{max_lon} {max_lat}, {min_lon} {max_lat}, {min_lon} {min_lat}))"
    )


def bbox_area_km2(min_lon: float, min_lat: float, max_lon: float, max_lat: float) -> float:
    """Rough equirectangular bbox area in km2 (provenance only, not survey grade)."""
    mid_lat = math.radians((min_lat + max_lat) / 2.0)
    width_km = (max_lon - min_lon) * 111.32 * math.cos(mid_lat)
    height_km = (max_lat - min_lat) * 110.574
    return round(abs(width_km * height_km), 4)


def create_run(session: Session, analysis_id: str, request_payload: dict[str, Any]) -> None:
    """Insert the initial ``running`` row for an analysis."""
    session.add(AnalysisRun(id=analysis_id, status="running", request_params=request_payload))
    session.commit()


def persist_result(session: Session, response: AnalysisResponse) -> None:
    """Persist a completed (or failed/no_data) pipeline result atomically.

    The run row must already exist (``create_run``); everything else is written in
    one transaction so a crash mid-persist cannot leave a half-recorded analysis.
    """
    run = session.get(AnalysisRun, response.analysis_id)
    if run is None:
        raise ValueError(f"analysis_run {response.analysis_id} does not exist")
    if run.completed_at is not None:  # a run is finalized exactly once
        raise ValueError(f"analysis_run {response.analysis_id} is already finalized")

    run.status = response.status.value
    run.completed_at = datetime.now(tz=timezone.utc)
    run.processing_time_s = response.processing_time_s
    run.error_message = response.error

    session.add(
        AnalysisAoi(
            analysis_id=run.id,
            srid=4326,
            min_lon=response.aoi.min_lon,
            min_lat=response.aoi.min_lat,
            max_lon=response.aoi.max_lon,
            max_lat=response.aoi.max_lat,
            area_km2=bbox_area_km2(
                response.aoi.min_lon, response.aoi.min_lat,
                response.aoi.max_lon, response.aoi.max_lat,
            ),
            geom=_wkt_polygon(
                response.aoi.min_lon, response.aoi.min_lat,
                response.aoi.max_lon, response.aoi.max_lat,
            ),
        )
    )

    for role, obs in (
        ("before", response.before_observation),
        ("after", response.after_observation),
    ):
        if obs is None:
            continue
        session.add(
            AnalysisObservation(
                analysis_id=run.id,
                role=role,
                acquisition_date=obs.date,
                source=obs.source,
                crs=obs.crs,
                pixel_resolution_m=obs.pixel_resolution_m,
                cloud_fraction=obs.cloud_fraction,
                bands=obs.bands,
            )
        )

    cd = response.change_detection
    if cd is not None:
        session.add(
            ChangeDetectionResult(
                analysis_id=run.id,
                total_pixels=cd.total_pixels,
                valid_pixels=cd.valid_pixels,
                valid_fraction=cd.valid_fraction,
                changed_pixels=cd.changed_pixels,
                changed_area_m2=cd.changed_area_m2,
                changed_area_ha=cd.changed_area_ha,
                changed_area_km2=cd.changed_area_km2,
                change_fraction_of_valid=cd.change_fraction_of_valid,
                threshold_used=cd.threshold_used,
                model_checkpoint=cd.model_checkpoint,
                crs=cd.crs,
                # Honest polygon accounting: polygons are stored only when actually
                # generated from a real change mask; otherwise explicitly unavailable.
                change_polygons_available=False,
                change_polygon=None,
            )
        )

    env = response.environmental_features
    if env is not None:
        for position, (name, value) in enumerate(_flatten_environmental_features(env)):
            session.add(
                EnvironmentalFeature(
                    analysis_id=run.id, position=position, name=name, value=value
                )
            )

    pr = response.priority_assessment
    if pr is not None:
        session.add(
            PriorityAssessment(
                analysis_id=run.id,
                priority=pr.priority.value,
                confidence=pr.confidence,
                probabilities=pr.probabilities,
                model_name="priority_xgboost",
                label_provenance_notice=pr.label_provenance_notice,
            )
        )
        for rank, driver in enumerate(pr.top_drivers, start=1):
            session.add(
                ShapDriver(
                    analysis_id=run.id,
                    rank=rank,
                    feature=driver.feature,
                    shap_value=driver.shap_value,
                    feature_value=driver.feature_value,
                    direction="increases" if driver.shap_value >= 0 else "decreases",
                    description=FEATURE_DESCRIPTIONS.get(driver.feature),
                )
            )

    art = response.artifacts
    if art is not None:
        for artifact_type, url in (
            ("change_probability_tif", art.change_probability_tif),
            ("change_mask_tif", art.change_mask_tif),
            ("inference_summary_json", art.inference_summary_json),
            ("change_geojson", art.change_geojson),
        ):
            if url is None:
                continue
            filename = url.rsplit("/", 1)[-1]
            session.add(
                Artifact(
                    analysis_id=run.id,
                    artifact_type=artifact_type,
                    filename=filename,
                    relative_path=f"{run.id}/{filename}",
                    url=url,
                )
            )

    session.commit()


def _flatten_environmental_features(env: Any) -> list[tuple[str, float | None]]:
    """Flatten the response's environmental summary in a stable, named order."""
    rows: list[tuple[str, float | None]] = [
        ("total_area_ha", env.total_area_ha),
        ("valid_area_ha", env.valid_area_ha),
        ("total_area_km2", env.total_area_km2),
        ("valid_area_km2", env.valid_area_km2),
        ("scl_veg_loss_ha", env.scl_veg_loss_ha),
        ("feature_vector_length", float(env.feature_vector_length)),
    ]
    veg = env.vegetation
    rows += [
        ("ndvi_before_mean", veg.ndvi_before_mean),
        ("ndvi_after_mean", veg.ndvi_after_mean),
        ("ndvi_diff_mean", veg.ndvi_diff_mean),
        ("change_ndvi_diff_mean", veg.change_ndvi_diff_mean),
        ("veg_loss_ha_01", veg.veg_loss_ha_01),
        ("veg_loss_ha_02", veg.veg_loss_ha_02),
        ("veg_loss_ha_03", veg.veg_loss_ha_03),
        ("veg_gain_ha_02", veg.veg_gain_ha_02),
        ("veg_loss_fraction", veg.veg_loss_fraction),
    ]
    land = env.landscape
    rows += [
        ("patch_count", float(land.patch_count)),
        ("mean_patch_area_ha", land.mean_patch_area_ha),
        ("max_patch_area_ha", land.max_patch_area_ha),
        ("patch_density_per_km2", land.patch_density_per_km2),
        ("large_patch_count", float(land.large_patch_count)),
    ]
    return rows


def get_analysis(session: Session, analysis_id: str) -> AnalysisRun | None:
    return session.get(AnalysisRun, analysis_id)


def list_analyses(
    session: Session, limit: int = 20, offset: int = 0
) -> tuple[list[AnalysisRun], int]:
    """Recent runs, newest first, with the total count for pagination."""
    total = session.scalar(select(func.count()).select_from(AnalysisRun)) or 0
    runs = (
        session.query(AnalysisRun)
        .order_by(AnalysisRun.created_at.desc(), AnalysisRun.id.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return runs, total

def run_to_dict(run: AnalysisRun) -> dict[str, Any]:
    """Frontend-friendly serialization of a persisted analysis run."""
    aoi = run.aoi
    return {
        "analysis_id": run.id,
        "status": run.status,
        "created_at": run.created_at.isoformat() if run.created_at else None,
        "completed_at": run.completed_at.isoformat() if run.completed_at else None,
        "processing_time_s": run.processing_time_s,
        "error": run.error_message,
        "request_params": run.request_params,
        "aoi": {
            "min_lon": aoi.min_lon,
            "min_lat": aoi.min_lat,
            "max_lon": aoi.max_lon,
            "max_lat": aoi.max_lat,
            "crs": f"EPSG:{aoi.srid}",
            "area_km2": aoi.area_km2,
        }
        if aoi
        else None,
        "observations": [
            {
                "role": obs.role,
                "date": obs.acquisition_date,
                "source": obs.source,
                "crs": obs.crs,
                "pixel_resolution_m": obs.pixel_resolution_m,
                "cloud_fraction": obs.cloud_fraction,
                "bands": obs.bands,
            }
            for obs in run.observations
        ],
        "change_detection": {
            "total_pixels": run.change_detection.total_pixels,
            "valid_pixels": run.change_detection.valid_pixels,
            "valid_fraction": run.change_detection.valid_fraction,
            "changed_pixels": run.change_detection.changed_pixels,
            "changed_area_m2": run.change_detection.changed_area_m2,
            "changed_area_ha": run.change_detection.changed_area_ha,
            "changed_area_km2": run.change_detection.changed_area_km2,
            "change_fraction_of_valid": run.change_detection.change_fraction_of_valid,
            "threshold_used": run.change_detection.threshold_used,
            "model_checkpoint": run.change_detection.model_checkpoint,
            "crs": run.change_detection.crs,
            "change_polygons_available": run.change_detection.change_polygons_available,
        }
        if run.change_detection
        else None,
        "environmental_features": [
            {"position": f.position, "name": f.name, "value": f.value}
            for f in sorted(run.environmental_features, key=lambda f: f.position)
        ],
        "priority_assessment": {
            "priority": run.priority_assessment.priority,
            "confidence": run.priority_assessment.confidence,
            "probabilities": run.priority_assessment.probabilities,
            "model_name": run.priority_assessment.model_name,
            "label_provenance_notice": run.priority_assessment.label_provenance_notice,
        }
        if run.priority_assessment
        else None,
        "shap_drivers": [
            {
                "rank": d.rank,
                "feature": d.feature,
                "shap_value": d.shap_value,
                "feature_value": d.feature_value,
                "direction": d.direction,
                "description": d.description,
            }
            for d in sorted(run.shap_drivers, key=lambda d: d.rank)
        ],
        "artifacts": [
            {
                "artifact_type": a.artifact_type,
                "filename": a.filename,
                "relative_path": a.relative_path,
                "url": a.url,
                "created_at": a.created_at.isoformat() if a.created_at else None,
            }
            for a in run.artifacts
        ],
    }
