"""Health check endpoint (Phase 5)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter

from app.schemas.analysis import HealthResponse
from app.services.analysis import DEFAULT_CHECKPOINT, DEFAULT_PRIORITY_MODEL

router = APIRouter(tags=["health"])

REPO_ROOT = Path(__file__).resolve().parents[4]
VERSION = "0.5.0"


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return API health status and model availability."""
    models_loaded = {
        "siamese_unet": DEFAULT_CHECKPOINT.is_file(),
        "priority_xgboost": DEFAULT_PRIORITY_MODEL.is_file(),
    }
    return HealthResponse(
        status="ok",
        version=VERSION,
        models_loaded=models_loaded,
        timestamp=datetime.now(tz=timezone.utc).isoformat(),
    )
