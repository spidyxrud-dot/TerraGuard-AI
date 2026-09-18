"""Analysis API endpoints (Phases 5 + 6).

POST /api/analyze           — full TerraGuard pipeline with persistence lifecycle
GET  /api/analysis/{id}     — retrieve a persisted analysis
GET  /api/analyses          — recent analysis history (paginated)
GET  /api/artifacts/{id}/{filename} — safe artifact file serving
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse

from app.db import database_available, get_session_factory
from app.repositories import (
    create_run,
    get_analysis,
    list_analyses,
    persist_result,
    run_to_dict,
)
from app.schemas.analysis import AnalysisRequest, AnalysisResponse
from app.services.analysis import (
    DEFAULT_CHECKPOINT,
    DEFAULT_PREDICTIONS_DIR,
    DEFAULT_PRIORITY_MODEL,
    DEFAULT_PROCESSED_DIR,
    DEFAULT_RAW_DIR,
    run_analysis,
)

router = APIRouter(tags=["analysis"])

# Safe filename allow-list for artifact serving
_SAFE_ARTIFACT_PATTERN = re.compile(
    r"^[a-zA-Z0-9_\-]+\.(tif|json|geojson|png)$"
)


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze(request_body: AnalysisRequest) -> AnalysisResponse:
    """Run the TerraGuard AI pipeline on a Sentinel-2 pair.

    Lifecycle: create analysis_run → run the existing pipeline → persist results →
    mark completed (or failed). If PostgreSQL is unreachable the analysis still
    runs and the response carries an explicit ``persistence_note`` instead of the
    pipeline pretending the database was written.
    """
    analysis_id = str(uuid.uuid4())
    artifact_dir = DEFAULT_PREDICTIONS_DIR / analysis_id

    session_factory = get_session_factory()
    persistence_ok = database_available()
    persistence_note: str | None = None
    if not persistence_ok:
        persistence_note = (
            "persistence unavailable: PostgreSQL not reachable; analysis result is "
            "not recorded in the database"
        )
    session = session_factory() if persistence_ok else None

    try:
        if session is not None:
            try:
                create_run(
                    session,
                    analysis_id,
                    {
                        "aoi": request_body.aoi.model_dump(),
                        "before_date": str(request_body.before_date),
                        "after_date": str(request_body.after_date),
                        "parameters": request_body.parameters.model_dump(),
                    },
                )
            except Exception as exc:  # noqa: BLE001 - degrade, don't fail the analysis
                persistence_ok = False
                persistence_note = f"persistence unavailable: {type(exc).__name__}: {exc}"

        response = run_analysis(
            analysis_id=analysis_id,
            aoi=request_body.aoi,
            before_date=str(request_body.before_date),
            after_date=str(request_body.after_date),
            params=request_body.parameters,
            raw_dir=DEFAULT_RAW_DIR,
            processed_dir=DEFAULT_PROCESSED_DIR,
            output_dir=artifact_dir,
            checkpoint_path=DEFAULT_CHECKPOINT,
            priority_model_path=DEFAULT_PRIORITY_MODEL,
        )

        if persistence_ok and session is not None:
            try:
                persist_result(session, response)
            except Exception as exc:  # noqa: BLE001 - reported, never fabricated
                persistence_note = f"persistence incomplete: {type(exc).__name__}: {exc}"

        response.persistence_note = persistence_note
        return response
    finally:
        if session is not None:
            session.close()


@router.get("/analysis/{analysis_id}")
async def get_persisted_analysis(analysis_id: str) -> dict:
    """Return a persisted analysis in a frontend-friendly structure."""
    if not database_available():
        raise HTTPException(status_code=503, detail="Database unavailable")

    session = get_session_factory()()
    try:
        run = get_analysis(session, analysis_id)
        if run is None:
            raise HTTPException(
                status_code=404, detail=f"Analysis '{analysis_id}' not found"
            )
        return run_to_dict(run)
    finally:
        session.close()


@router.get("/analyses")
async def list_recent_analyses(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> dict:
    """Recent analysis history, newest first."""
    if not database_available():
        raise HTTPException(status_code=503, detail="Database unavailable")

    session = get_session_factory()()
    try:
        runs, total = list_analyses(session, limit=limit, offset=offset)
        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "analyses": [
                {
                    "analysis_id": run.id,
                    "status": run.status,
                    "created_at": run.created_at.isoformat() if run.created_at else None,
                    "completed_at": run.completed_at.isoformat() if run.completed_at else None,
                    "priority": (
                        run.priority_assessment.priority
                        if run.priority_assessment
                        else None
                    ),
                    "changed_area_ha": (
                        run.change_detection.changed_area_ha
                        if run.change_detection
                        else None
                    ),
                }
                for run in runs
            ],
        }
    finally:
        session.close()


@router.get("/artifacts/{analysis_id}/{filename}")
async def get_artifact(analysis_id: str, filename: str) -> FileResponse:
    """Serve a generated analysis artifact (GeoTIFF, JSON, GeoJSON).

    Security controls:
    - analysis_id and filename are validated against safe patterns.
    - Traversal via '..' is rejected.
    - Only files inside the predictions directory are served.
    """
    # Validate IDs to prevent path traversal
    if not re.match(r"^[a-zA-Z0-9\-]{8,64}$", analysis_id):
        raise HTTPException(status_code=400, detail="Invalid analysis_id format")
    if not _SAFE_ARTIFACT_PATTERN.match(filename):
        raise HTTPException(status_code=400, detail="Invalid or disallowed filename")

    artifact_path = (DEFAULT_PREDICTIONS_DIR / analysis_id / filename).resolve()

    # Ensure the resolved path is still within the predictions root
    predictions_root = DEFAULT_PREDICTIONS_DIR.resolve()
    try:
        artifact_path.relative_to(predictions_root)
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")

    if not artifact_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Artifact '{filename}' not found for analysis '{analysis_id}'",
        )

    media_type = "application/octet-stream"
    if filename.endswith(".json") or filename.endswith(".geojson"):
        media_type = "application/json"

    return FileResponse(
        path=str(artifact_path),
        media_type=media_type,
        filename=filename,
    )
