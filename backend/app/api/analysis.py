"""Analysis API endpoints (Phase 5).

POST /api/analyze    — full TerraGuard pipeline
GET  /api/artifacts/{analysis_id}/{filename} — safe artifact file serving
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

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

    The pipeline uses the locally available demo Sentinel-2 pair as the data
    source.  Dates and AOI from the request are recorded in the response for
    provenance; they do not yet drive online satellite acquisition.
    """
    analysis_id = str(uuid.uuid4())
    artifact_dir = DEFAULT_PREDICTIONS_DIR / analysis_id

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
    return response


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
