"""Health check endpoint (Phases 5 + 6)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter

from app.db import database_available, postgis_available
from app.schemas.analysis import HealthResponse
from app.services.analysis import DEFAULT_CHECKPOINT, DEFAULT_PRIORITY_MODEL

router = APIRouter(tags=["health"])

REPO_ROOT = Path(__file__).resolve().parents[4]
VERSION = "0.6.0"


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return API health status, model availability and database status."""
    models_loaded = {
        "siamese_unet": DEFAULT_CHECKPOINT.is_file(),
        "priority_xgboost": DEFAULT_PRIORITY_MODEL.is_file(),
    }

    if not database_available():
        database_status = "unreachable"
    elif postgis_available() is False:
        database_status = "connected_postgis_missing"
    else:
        database_status = "ok"

    return HealthResponse(
        status="ok",
        version=VERSION,
        models_loaded=models_loaded,
        timestamp=datetime.now(tz=timezone.utc).isoformat(),
        database=database_status,
    )
