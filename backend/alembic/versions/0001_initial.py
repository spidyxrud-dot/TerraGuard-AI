"""initial TerraGuard persistence schema

Revision ID: 0001_initial
Revises:
Create Date: 2026-09-18

Creates the PostGIS extension and all analysis-persistence tables:
analysis_runs, analysis_aois (geometry), analysis_observations,
change_detection_results (change polygons nullable + availability flag),
environmental_features, priority_assessments, shap_drivers, artifacts.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from geoalchemy2 import Geometry

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # PostGIS must exist before any geometry column is created. The compose
    # image postgis/postgis:16-3.4 ships the extension; POSTGRES_USER is a
    # superuser there, so CREATE EXTENSION succeeds in dev and CI.
    op.execute("CREATE EXTENSION IF NOT EXISTS postgis")

    op.create_table(
        "analysis_runs",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("status", sa.String(length=16), nullable=False, server_default="pending"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("processing_time_s", sa.Float(), nullable=True),
        sa.Column("error_message", sa.String(), nullable=True),
        sa.Column("request_params", sa.JSON(), nullable=True),
        sa.CheckConstraint(
            "status IN ('pending','running','complete','failed','no_data')",
            name="ck_analysis_runs_status",
        ),
    )
    op.create_index(
        "ix_analysis_runs_status_created", "analysis_runs", ["status", "created_at"]
    )

    op.create_table(
        "analysis_aois",
        sa.Column(
            "analysis_id",
            sa.String(length=36),
            sa.ForeignKey("analysis_runs.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("srid", sa.Integer(), nullable=False, server_default="4326"),
        sa.Column("min_lon", sa.Float(), nullable=False),
        sa.Column("min_lat", sa.Float(), nullable=False),
        sa.Column("max_lon", sa.Float(), nullable=False),
        sa.Column("max_lat", sa.Float(), nullable=False),
        sa.Column("area_km2", sa.Float(), nullable=True),
        sa.Column("geom", Geometry(geometry_type="POLYGON", srid=4326), nullable=True),
    )
    op.create_index(
        "ix_analysis_aois_geom", "analysis_aois", ["geom"], postgresql_using="gist"
    )

    op.create_table(
        "analysis_observations",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "analysis_id",
            sa.String(length=36),
            sa.ForeignKey("analysis_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("role", sa.String(length=8), nullable=False),
        sa.Column("acquisition_date", sa.String(length=10), nullable=True),
        sa.Column("source", sa.String(length=64), nullable=True),
        sa.Column("item_id", sa.String(length=128), nullable=True),
        sa.Column("mgrs_tile", sa.String(length=16), nullable=True),
        sa.Column("platform", sa.String(length=32), nullable=True),
        sa.Column("crs", sa.String(length=64), nullable=True),
        sa.Column("pixel_resolution_m", sa.Float(), nullable=True),
        sa.Column("cloud_fraction", sa.Float(), nullable=True),
        sa.Column("bands", sa.JSON(), nullable=True),
        sa.CheckConstraint("role IN ('before','after')", name="ck_observation_role"),
    )
    op.create_index("ix_observations_analysis", "analysis_observations", ["analysis_id"])

    op.create_table(
        "change_detection_results",
        sa.Column(
            "analysis_id",
            sa.String(length=36),
            sa.ForeignKey("analysis_runs.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("total_pixels", sa.Integer(), nullable=False),
        sa.Column("valid_pixels", sa.Integer(), nullable=False),
        sa.Column("valid_fraction", sa.Float(), nullable=False),
        sa.Column("changed_pixels", sa.Integer(), nullable=False),
        sa.Column("changed_area_m2", sa.Float(), nullable=False),
        sa.Column("changed_area_ha", sa.Float(), nullable=False),
        sa.Column("changed_area_km2", sa.Float(), nullable=False),
        sa.Column("change_fraction_of_valid", sa.Float(), nullable=False),
        sa.Column("threshold_used", sa.Float(), nullable=False),
        sa.Column("model_checkpoint", sa.String(length=256), nullable=True),
        sa.Column("crs", sa.String(length=64), nullable=True),
        sa.Column("change_polygons_available", sa.Boolean(), nullable=False,
                  server_default=sa.text("false")),
        sa.Column(
            "change_polygon",
            Geometry(geometry_type="MULTIPOLYGON", srid=4326),
            nullable=True,
        ),
    )

    op.create_table(
        "environmental_features",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "analysis_id",
            sa.String(length=36),
            sa.ForeignKey("analysis_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=64), nullable=False),
        sa.Column("value", sa.Float(), nullable=True),
    )
    op.create_index(
        "ix_env_features_analysis_position",
        "environmental_features",
        ["analysis_id", "position"],
    )

    op.create_table(
        "priority_assessments",
        sa.Column(
            "analysis_id",
            sa.String(length=36),
            sa.ForeignKey("analysis_runs.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("priority", sa.String(length=8), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("probabilities", sa.JSON(), nullable=False),
        sa.Column("model_name", sa.String(length=128), nullable=True),
        sa.Column("model_version", sa.String(length=64), nullable=True),
        sa.Column("label_provenance_notice", sa.String(), nullable=True),
    )

    op.create_table(
        "shap_drivers",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "analysis_id",
            sa.String(length=36),
            sa.ForeignKey("analysis_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("rank", sa.Integer(), nullable=False),
        sa.Column("feature", sa.String(length=64), nullable=False),
        sa.Column("shap_value", sa.Float(), nullable=False),
        sa.Column("feature_value", sa.Float(), nullable=True),
        sa.Column("direction", sa.String(length=16), nullable=True),
        sa.Column("description", sa.String(), nullable=True),
        sa.CheckConstraint(
            "direction IN ('increases','decreases')", name="ck_shap_direction"
        ),
    )
    op.create_index("ix_shap_drivers_analysis_rank", "shap_drivers", ["analysis_id", "rank"])

    op.create_table(
        "artifacts",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "analysis_id",
            sa.String(length=36),
            sa.ForeignKey("analysis_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("artifact_type", sa.String(length=32), nullable=False),
        sa.Column("filename", sa.String(length=255), nullable=False),
        sa.Column("relative_path", sa.String(length=512), nullable=False),
        sa.Column("url", sa.String(length=512), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("ix_artifacts_analysis", "artifacts", ["analysis_id"])


def downgrade() -> None:
    op.drop_table("artifacts")
    op.drop_table("shap_drivers")
    op.drop_table("priority_assessments")
    op.drop_table("environmental_features")
    op.drop_table("change_detection_results")
    op.drop_table("analysis_observations")
    op.drop_table("analysis_aois")
    op.drop_table("analysis_runs")