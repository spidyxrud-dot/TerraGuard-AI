"""Pydantic schemas for TerraGuard analysis API (Phase 5).

Request schema: AnalysisRequest
Response schema: AnalysisResponse (and nested sub-schemas)
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class AnalysisStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    NO_DATA = "no_data"


class PriorityLabel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

class BoundingBox(BaseModel):
    """WGS-84 bounding box (EPSG:4326)."""

    min_lon: float = Field(..., ge=-180.0, le=180.0, description="West boundary (degrees)")
    min_lat: float = Field(..., ge=-90.0, le=90.0, description="South boundary (degrees)")
    max_lon: float = Field(..., ge=-180.0, le=180.0, description="East boundary (degrees)")
    max_lat: float = Field(..., ge=-90.0, le=90.0, description="North boundary (degrees)")

    @model_validator(mode="after")
    def _check_bounds(self) -> "BoundingBox":
        if self.min_lon >= self.max_lon:
            raise ValueError("min_lon must be strictly less than max_lon")
        if self.min_lat >= self.max_lat:
            raise ValueError("min_lat must be strictly less than max_lat")
        return self


class AnalysisParameters(BaseModel):
    """Optional tuning parameters for the analysis pipeline."""

    change_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Probability threshold for binary change mask (0–1)",
    )
    scl_policy: str = Field(
        default="strict",
        description="SCL cloud masking policy: 'strict' or 'lenient'",
    )
    pixel_resolution_m: float = Field(
        default=10.0,
        gt=0.0,
        description="Pixel resolution in metres (default 10 for Sentinel-2 10m bands)",
    )

    @field_validator("scl_policy")
    @classmethod
    def _validate_scl_policy(cls, v: str) -> str:
        if v not in {"strict", "lenient"}:
            raise ValueError("scl_policy must be 'strict' or 'lenient'")
        return v


class AnalysisRequest(BaseModel):
    """Request payload for POST /api/analyze."""

    aoi: BoundingBox = Field(..., description="Area of interest (WGS-84 bounding box)")
    before_date: date = Field(..., description="Acquisition date of the before observation (YYYY-MM-DD)")
    after_date: date = Field(..., description="Acquisition date of the after observation (YYYY-MM-DD)")
    parameters: AnalysisParameters = Field(
        default_factory=AnalysisParameters,
        description="Optional pipeline parameters",
    )

    @model_validator(mode="after")
    def _check_dates(self) -> "AnalysisRequest":
        if self.before_date >= self.after_date:
            raise ValueError("before_date must be strictly before after_date")
        return self


# ---------------------------------------------------------------------------
# Response sub-schemas
# ---------------------------------------------------------------------------

class AOISchema(BaseModel):
    """Echoed AOI in the response."""

    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float
    crs: str = "EPSG:4326"


class ObservationSummary(BaseModel):
    """Summary of a single satellite observation."""

    date: str
    source: str
    bands: list[str]
    crs: str | None = None
    pixel_resolution_m: float | None = None
    cloud_fraction: float | None = None


class ChangeDetectionSummary(BaseModel):
    """Change-detection results from the Siamese U-Net."""

    total_pixels: int
    valid_pixels: int
    valid_fraction: float
    changed_pixels: int
    changed_area_m2: float
    changed_area_ha: float
    changed_area_km2: float
    change_fraction_of_valid: float
    threshold_used: float
    model_checkpoint: str | None = None
    crs: str | None = None


class VegetationMetrics(BaseModel):
    ndvi_before_mean: float | None = None
    ndvi_after_mean: float | None = None
    ndvi_diff_mean: float | None = None
    change_ndvi_diff_mean: float | None = None
    veg_loss_ha_01: float = 0.0
    veg_loss_ha_02: float = 0.0
    veg_loss_ha_03: float = 0.0
    veg_gain_ha_02: float = 0.0
    veg_loss_fraction: float = 0.0


class LandscapeMetrics(BaseModel):
    patch_count: int = 0
    mean_patch_area_ha: float = 0.0
    max_patch_area_ha: float = 0.0
    patch_density_per_km2: float = 0.0
    large_patch_count: int = 0


class EnvironmentalFeaturesSummary(BaseModel):
    """Aggregated environmental change intelligence."""

    total_area_ha: float
    valid_area_ha: float
    total_area_km2: float
    valid_area_km2: float
    vegetation: VegetationMetrics
    landscape: LandscapeMetrics
    scl_veg_loss_ha: float = 0.0
    feature_vector_length: int = 25


class SHAPDriverSchema(BaseModel):
    feature: str
    shap_value: float
    feature_value: float


class PrioritySummary(BaseModel):
    """XGBoost priority assessment with SHAP evidence."""

    priority: PriorityLabel
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: dict[str, float]
    top_drivers: list[SHAPDriverSchema]
    label_provenance_notice: str


class ArtifactSchema(BaseModel):
    """References to generated raster and JSON outputs."""

    analysis_id: str
    change_probability_tif: str | None = None
    change_mask_tif: str | None = None
    inference_summary_json: str | None = None
    change_geojson: str | None = None


class AnalysisResponse(BaseModel):
    """Full structured response from POST /api/analyze."""

    analysis_id: str
    status: AnalysisStatus
    created_at: str
    processing_time_s: float | None = None

    aoi: AOISchema
    before_observation: ObservationSummary | None = None
    after_observation: ObservationSummary | None = None

    change_detection: ChangeDetectionSummary | None = None
    environmental_features: EnvironmentalFeaturesSummary | None = None
    priority_assessment: PrioritySummary | None = None
    action_insight: str | None = None

    artifacts: ArtifactSchema | None = None

    error: str | None = None


class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: dict[str, bool]
    timestamp: str
