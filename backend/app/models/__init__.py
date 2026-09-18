"""SQLAlchemy models for TerraGuard analysis persistence (Phase 6).

Normalized relational schema; generated GeoTIFF/JSON artifacts are stored on the
filesystem and referenced by relative path — never stored in PostgreSQL.
PostGIS geometry columns carry the AOI polygon and (when actually generated from a
real change mask) change polygons; polygon availability is explicit, never invented.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from geoalchemy2 import Geometry
from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class AnalysisRun(Base):
    """One execution of the analysis pipeline."""

    __tablename__ = "analysis_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    processing_time_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    error_message: Mapped[str | None] = mapped_column(String, nullable=True)
    request_params: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    aoi: Mapped["AnalysisAoi | None"] = relationship(back_populates="run", uselist=False)
    observations: Mapped[list["AnalysisObservation"]] = relationship(back_populates="run")
    change_detection: Mapped["ChangeDetectionResult | None"] = relationship(
        back_populates="run", uselist=False
    )
    environmental_features: Mapped[list["EnvironmentalFeature"]] = relationship(
        back_populates="run"
    )
    priority_assessment: Mapped["PriorityAssessment | None"] = relationship(
        back_populates="run", uselist=False
    )
    shap_drivers: Mapped[list["ShapDriver"]] = relationship(back_populates="run")
    artifacts: Mapped[list["Artifact"]] = relationship(back_populates="run")

    __table_args__ = (
        CheckConstraint(
            "status IN ('pending','running','complete','failed','no_data')",
            name="ck_analysis_runs_status",
        ),
        Index("ix_analysis_runs_status_created", "status", "created_at"),
    )


class AnalysisAoi(Base):
    """Area of interest as PostGIS geometry plus its bounding coordinates."""

    __tablename__ = "analysis_aois"

    analysis_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("analysis_runs.id", ondelete="CASCADE"), primary_key=True
    )
    srid: Mapped[int] = mapped_column(Integer, nullable=False, default=4326)
    min_lon: Mapped[float] = mapped_column(Float, nullable=False)
    min_lat: Mapped[float] = mapped_column(Float, nullable=False)
    max_lon: Mapped[float] = mapped_column(Float, nullable=False)
    max_lat: Mapped[float] = mapped_column(Float, nullable=False)
    area_km2: Mapped[float | None] = mapped_column(Float, nullable=True)
    geom: Mapped[Any] = mapped_column(
        Geometry(geometry_type="POLYGON", srid=4326, spatial_index=False), nullable=True
    )

    run: Mapped[AnalysisRun] = relationship(back_populates="aoi")


class AnalysisObservation(Base):
    """Before/after Sentinel-2 observation metadata for one analysis."""

    __tablename__ = "analysis_observations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    analysis_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("analysis_runs.id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[str] = mapped_column(String(8), nullable=False)  # 'before' | 'after'
    acquisition_date: Mapped[str | None] = mapped_column(String(10), nullable=True)
    source: Mapped[str | None] = mapped_column(String(64), nullable=True)
    item_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    mgrs_tile: Mapped[str | None] = mapped_column(String(16), nullable=True)
    platform: Mapped[str | None] = mapped_column(String(32), nullable=True)
    crs: Mapped[str | None] = mapped_column(String(64), nullable=True)
    pixel_resolution_m: Mapped[float | None] = mapped_column(Float, nullable=True)
    cloud_fraction: Mapped[float | None] = mapped_column(Float, nullable=True)
    bands: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)

    run: Mapped[AnalysisRun] = relationship(back_populates="observations")

    __table_args__ = (
        CheckConstraint("role IN ('before','after')", name="ck_observation_role"),
        Index("ix_observations_analysis", "analysis_id"),
    )


class ChangeDetectionResult(Base):
    """Siamese U-Net change-detection statistics for one analysis."""

    __tablename__ = "change_detection_results"

    analysis_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("analysis_runs.id", ondelete="CASCADE"), primary_key=True
    )
    total_pixels: Mapped[int] = mapped_column(Integer, nullable=False)
    valid_pixels: Mapped[int] = mapped_column(Integer, nullable=False)
    valid_fraction: Mapped[float] = mapped_column(Float, nullable=False)
    changed_pixels: Mapped[int] = mapped_column(Integer, nullable=False)
    changed_area_m2: Mapped[float] = mapped_column(Float, nullable=False)
    changed_area_ha: Mapped[float] = mapped_column(Float, nullable=False)
    changed_area_km2: Mapped[float] = mapped_column(Float, nullable=False)
    change_fraction_of_valid: Mapped[float] = mapped_column(Float, nullable=False)
    threshold_used: Mapped[float] = mapped_column(Float, nullable=False)
    model_checkpoint: Mapped[str | None] = mapped_column(String(256), nullable=True)
    crs: Mapped[str | None] = mapped_column(String(64), nullable=True)
    change_polygons_available: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    change_polygon: Mapped[Any] = mapped_column(
        Geometry(geometry_type="MULTIPOLYGON", srid=4326, spatial_index=False), nullable=True
    )

    run: Mapped[AnalysisRun] = relationship(back_populates="change_detection")


class EnvironmentalFeature(Base):
    """One named feature value; preserves FEATURE_NAMES order via ``position``."""

    __tablename__ = "environmental_features"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    analysis_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("analysis_runs.id", ondelete="CASCADE"), nullable=False
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    name: Mapped[str] = mapped_column(String(64), nullable=False)
    value: Mapped[float | None] = mapped_column(Float, nullable=True)

    run: Mapped[AnalysisRun] = relationship(back_populates="environmental_features")

    __table_args__ = (
        Index("ix_env_features_analysis_position", "analysis_id", "position"),
    )


class PriorityAssessment(Base):
    """XGBoost priority output with explicit label provenance."""

    __tablename__ = "priority_assessments"

    analysis_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("analysis_runs.id", ondelete="CASCADE"), primary_key=True
    )
    priority: Mapped[str] = mapped_column(String(8), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    probabilities: Mapped[dict[str, float]] = mapped_column(JSON, nullable=False)
    model_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    model_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    label_provenance_notice: Mapped[str | None] = mapped_column(String, nullable=True)

    run: Mapped[AnalysisRun] = relationship(back_populates="priority_assessment")


class ShapDriver(Base):
    """TreeSHAP contribution of one feature, ranked by absolute contribution."""

    __tablename__ = "shap_drivers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    analysis_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("analysis_runs.id", ondelete="CASCADE"), nullable=False
    )
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    feature: Mapped[str] = mapped_column(String(64), nullable=False)
    shap_value: Mapped[float] = mapped_column(Float, nullable=False)
    feature_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    direction: Mapped[str | None] = mapped_column(String(16), nullable=True)
    description: Mapped[str | None] = mapped_column(String, nullable=True)

    run: Mapped[AnalysisRun] = relationship(back_populates="shap_drivers")

    __table_args__ = (
        CheckConstraint("direction IN ('increases','decreases')", name="ck_shap_direction"),
        Index("ix_shap_drivers_analysis_rank", "analysis_id", "rank"),
    )


class Artifact(Base):
    """Filesystem artifact reference — raster payloads never live in PostgreSQL."""

    __tablename__ = "artifacts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    analysis_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("analysis_runs.id", ondelete="CASCADE"), nullable=False
    )
    artifact_type: Mapped[str] = mapped_column(String(32), nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    relative_path: Mapped[str] = mapped_column(String(512), nullable=False)
    url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    run: Mapped[AnalysisRun] = relationship(back_populates="artifacts")

    __table_args__ = (
        Index("ix_artifacts_analysis", "analysis_id"),
    )
