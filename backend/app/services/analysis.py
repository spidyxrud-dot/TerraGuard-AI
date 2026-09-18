"""Analysis orchestration service (Phase 5).

Orchestrates the full TerraGuard AI pipeline:
  request → validate → load pair → preprocess → SCL mask →
  Siamese U-Net → environmental features → XGBoost priority →
  TreeSHAP → structured response

The service layer owns NO business logic of its own — it wires together the
existing ml/ and services/ modules without duplicating them.
"""

from __future__ import annotations

import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
BACKEND_DIR = REPO_ROOT / "backend"
for _entry in (BACKEND_DIR, REPO_ROOT):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

import numpy as np

# Import pipeline modules at module level so tests can patch them reliably.
from app.services.features import extract_features_from_processed
from app.services.preprocessing import load_sentinel_pair, prepare_pair
from ml.change_detection.inference import run_inference
from ml.priority.inference import assess_priority

from app.schemas.analysis import (
    AOISchema,
    AnalysisParameters,
    AnalysisResponse,
    AnalysisStatus,
    ArtifactSchema,
    BoundingBox,
    ChangeDetectionSummary,
    EnvironmentalFeaturesSummary,
    LandscapeMetrics,
    ObservationSummary,
    PriorityLabel,
    PrioritySummary,
    SHAPDriverSchema,
    VegetationMetrics,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MODELS_DIR = REPO_ROOT / "models"
DEFAULT_CHECKPOINT = DEFAULT_MODELS_DIR / "siamese_unet.pth"
DEFAULT_PRIORITY_MODEL = DEFAULT_MODELS_DIR / "priority_xgboost.json"
DEFAULT_RAW_DIR = REPO_ROOT / "data" / "raw" / "demo_area"
DEFAULT_PROCESSED_DIR = REPO_ROOT / "data" / "processed" / "demo_area"
DEFAULT_PREDICTIONS_DIR = REPO_ROOT / "data" / "predictions" / "demo_area"


# ---------------------------------------------------------------------------
# Artifact URL helpers
# ---------------------------------------------------------------------------

def _artifact_url(analysis_id: str, filename: str) -> str:
    """Build a safe artifact URL path."""
    return f"/api/artifacts/{analysis_id}/{filename}"


# ---------------------------------------------------------------------------
# Schema builders
# ---------------------------------------------------------------------------

def _aoi_schema(aoi: BoundingBox) -> AOISchema:
    return AOISchema(
        min_lon=aoi.min_lon,
        min_lat=aoi.min_lat,
        max_lon=aoi.max_lon,
        max_lat=aoi.max_lat,
    )


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

def run_analysis(
    analysis_id: str,
    aoi: BoundingBox,
    before_date: str,
    after_date: str,
    params: AnalysisParameters,
    raw_dir: Path = DEFAULT_RAW_DIR,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    output_dir: Path | None = None,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    priority_model_path: Path = DEFAULT_PRIORITY_MODEL,
) -> AnalysisResponse:
    """Execute the full TerraGuard analysis pipeline.

    Raises structured AnalysisResponse with status=FAILED on expected errors
    rather than raising raw exceptions (so the API endpoint can return 200 JSON
    with error details instead of an unhandled 500).
    """
    started_at = time.monotonic()
    created_at = datetime.now(tz=timezone.utc).isoformat()
    aoi_schema = _aoi_schema(aoi)
    artifact_dir = output_dir or (DEFAULT_PREDICTIONS_DIR / analysis_id)

    def _fail(msg: str) -> AnalysisResponse:
        return AnalysisResponse(
            analysis_id=analysis_id,
            status=AnalysisStatus.FAILED,
            created_at=created_at,
            processing_time_s=round(time.monotonic() - started_at, 3),
            aoi=aoi_schema,
            error=msg,
        )

    # ------------------------------------------------------------------ #
    # 1. Load Sentinel-2 pair from disk                                   #
    # ------------------------------------------------------------------ #
    try:
        before_obs, after_obs = load_sentinel_pair(raw_dir)
    except FileNotFoundError as exc:
        return AnalysisResponse(
            analysis_id=analysis_id,
            status=AnalysisStatus.NO_DATA,
            created_at=created_at,
            processing_time_s=round(time.monotonic() - started_at, 3),
            aoi=aoi_schema,
            error=str(exc),
        )

    before_summary = ObservationSummary(
        date=str(before_obs.date or before_date),
        source="Sentinel-2 L2A",
        bands=list(before_obs.bands.keys()),
        crs=str(before_obs.bands[next(iter(before_obs.bands))].crs or ""),
        pixel_resolution_m=params.pixel_resolution_m,
        cloud_fraction=before_obs.cloud_cover,
    )
    after_summary = ObservationSummary(
        date=str(after_obs.date or after_date),
        source="Sentinel-2 L2A",
        bands=list(after_obs.bands.keys()),
        crs=str(after_obs.bands[next(iter(after_obs.bands))].crs or ""),
        pixel_resolution_m=params.pixel_resolution_m,
        cloud_fraction=after_obs.cloud_cover,
    )

    # ------------------------------------------------------------------ #
    # 2. Preprocessing + cloud masking                                    #
    # ------------------------------------------------------------------ #
    try:
        prepare_pair(raw_dir=raw_dir, processed_dir=processed_dir)
    except (FileNotFoundError, RuntimeError) as exc:
        return _fail(f"Preprocessing failed: {exc}")

    # ------------------------------------------------------------------ #
    # 3. Siamese U-Net inference                                          #
    # ------------------------------------------------------------------ #
    if not checkpoint_path.is_file():
        return _fail(
            f"Model checkpoint not found at {checkpoint_path}. "
            "Train the Siamese U-Net first."
        )

    try:
        inference_result = run_inference(
            checkpoint_path=checkpoint_path,
            processed_dir=processed_dir,
            output_dir=artifact_dir,
            threshold=params.change_threshold,
        )
    except Exception as exc:
        return _fail(f"Inference failed: {type(exc).__name__}: {exc}")

    stats = inference_result.stats
    change_det = ChangeDetectionSummary(
        total_pixels=stats["total_pixels"],
        valid_pixels=stats["valid_pixels"],
        valid_fraction=stats["valid_fraction"],
        changed_pixels=stats["change_pixels"],
        changed_area_m2=float(stats["change_pixels"] * stats["pixel_area_m2"]),
        changed_area_ha=stats["change_area_hectares"],
        changed_area_km2=stats["change_area_km2"],
        change_fraction_of_valid=stats["change_fraction_of_valid"],
        threshold_used=inference_result.threshold,
        model_checkpoint=str(checkpoint_path.name),
        crs=inference_result.crs,
    )

    # ------------------------------------------------------------------ #
    # 4. Environmental feature extraction                                 #
    # ------------------------------------------------------------------ #
    try:
        env_features = extract_features_from_processed(
            processed_dir=processed_dir,
            predictions_dir=artifact_dir,
        )
    except Exception as exc:
        return _fail(f"Feature extraction failed: {type(exc).__name__}: {exc}")

    env_summary = EnvironmentalFeaturesSummary(
        total_area_ha=env_features.total_area_ha,
        valid_area_ha=env_features.valid_area_ha,
        total_area_km2=env_features.total_area_km2,
        valid_area_km2=env_features.valid_area_km2,
        vegetation=VegetationMetrics(
            ndvi_before_mean=env_features.ndvi_before_mean,
            ndvi_after_mean=env_features.ndvi_after_mean,
            ndvi_diff_mean=env_features.ndvi_diff_mean,
            change_ndvi_diff_mean=env_features.change_ndvi_diff_mean,
            veg_loss_ha_01=env_features.veg_loss_ha_01,
            veg_loss_ha_02=env_features.veg_loss_ha_02,
            veg_loss_ha_03=env_features.veg_loss_ha_03,
            veg_gain_ha_02=env_features.veg_gain_ha_02,
            veg_loss_fraction=env_features.veg_loss_fraction,
        ),
        landscape=LandscapeMetrics(
            patch_count=env_features.patch_count,
            mean_patch_area_ha=env_features.mean_patch_area_ha,
            max_patch_area_ha=env_features.max_patch_area_ha,
            patch_density_per_km2=env_features.patch_density_per_km2,
            large_patch_count=env_features.large_patch_count,
        ),
        scl_veg_loss_ha=env_features.scl_veg_loss_ha,
        feature_vector_length=25,
    )

    # ------------------------------------------------------------------ #
    # 5. XGBoost priority classification                                  #
    # ------------------------------------------------------------------ #
    priority_summary: PrioritySummary | None = None
    action_insight: str | None = None

    if priority_model_path.is_file():
        try:
            assessment = assess_priority(
                features=env_features,
                model_path=priority_model_path,
            )
            top_drivers = [
                SHAPDriverSchema(
                    feature=d["feature"],
                    shap_value=float(d["shap_value"]),
                    feature_value=float(d.get("feature_value", 0.0)),
                )
                for d in assessment.top_drivers[:5]
            ]
            priority_summary = PrioritySummary(
                priority=PriorityLabel(assessment.priority),
                confidence=assessment.confidence,
                probabilities=assessment.probabilities,
                top_drivers=top_drivers,
                label_provenance_notice=assessment.label_provenance_notice,
            )
            action_insight = assessment.action_insight
        except Exception as exc:
            # Non-fatal: return change detection results without priority
            action_insight = (
                f"Priority assessment unavailable: {type(exc).__name__}: {exc}"
            )

    # ------------------------------------------------------------------ #
    # 6. Artifacts                                                        #
    # ------------------------------------------------------------------ #
    artifact_schema = ArtifactSchema(
        analysis_id=analysis_id,
        change_probability_tif=_artifact_url(analysis_id, "change_probability.tif"),
        change_mask_tif=_artifact_url(analysis_id, "change_mask.tif"),
        inference_summary_json=_artifact_url(analysis_id, "inference_summary.json"),
    )

    return AnalysisResponse(
        analysis_id=analysis_id,
        status=AnalysisStatus.COMPLETE,
        created_at=created_at,
        processing_time_s=round(time.monotonic() - started_at, 3),
        aoi=aoi_schema,
        before_observation=before_summary,
        after_observation=after_summary,
        change_detection=change_det,
        environmental_features=env_summary,
        priority_assessment=priority_summary,
        action_insight=action_insight,
        artifacts=artifact_schema,
    )
