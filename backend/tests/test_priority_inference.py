"""Tests for priority inference pipeline (Phase 4 / Priority Model)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.services.features import EnvironmentalFeatures
from ml.priority.dataset import generate_prototype_dataset
from ml.priority.inference import PriorityAssessment, assess_priority, main
from ml.priority.model import PriorityClassifier, PriorityModelConfig


@pytest.fixture()
def sample_features() -> EnvironmentalFeatures:
    return EnvironmentalFeatures(
        total_pixels=10000,
        valid_pixels=9950,
        valid_fraction=0.995,
        changed_pixels=1200,
        changed_area_m2=120000.0,
        changed_area_ha=12.0,
        changed_area_km2=0.12,
        change_fraction_of_valid=0.1206,
        ndvi_before_mean=0.68,
        ndvi_after_mean=0.25,
        ndvi_diff_mean=-0.43,
        change_ndvi_before_mean=0.72,
        change_ndvi_after_mean=0.18,
        change_ndvi_diff_mean=-0.54,
        change_ndvi_diff_std=0.12,
        veg_loss_pixels_01=1100,
        veg_loss_pixels_02=950,
        veg_loss_pixels_03=400,
        veg_gain_pixels_02=0,
        veg_loss_ha_01=11.0,
        veg_loss_ha_02=9.5,
        veg_loss_ha_03=4.0,
        veg_gain_ha_02=0.0,
        veg_loss_fraction=0.0955,
        patch_count=4,
        mean_patch_area_ha=3.0,
        max_patch_area_ha=8.2,
        patch_density_per_km2=4.0,
        large_patch_count=3,
        delta_b04_mean=0.18,
        delta_b08_mean=-0.32,
        scl_veg_loss_ha=8.0,
    )


@pytest.fixture()
def trained_model_file(tmp_path: Path) -> Path:
    dataset = generate_prototype_dataset(n_samples=150, seed=42)
    config = PriorityModelConfig(n_estimators=15, max_depth=3, random_state=42)
    classifier = PriorityClassifier(config=config)
    classifier.fit(dataset.X, dataset.y)

    model_file = tmp_path / "priority_xgboost.json"
    classifier.save(model_file)
    return model_file


def test_assess_priority_from_features(sample_features: EnvironmentalFeatures, trained_model_file: Path) -> None:
    assessment = assess_priority(sample_features, model_path=trained_model_file)

    assert isinstance(assessment, PriorityAssessment)
    assert assessment.priority in {"LOW", "MEDIUM", "HIGH"}
    assert 0.0 <= assessment.confidence <= 1.0
    assert len(assessment.action_insight) > 0
    assert len(assessment.top_drivers) > 0

    d = assessment.to_dict()
    assert "action_insight" in d
    assert "top_drivers" in d
    assert "label_provenance_notice" in d


def test_assess_priority_missing_model_raises(sample_features: EnvironmentalFeatures, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Trained Priority Model not found"):
        assess_priority(sample_features, model_path=tmp_path / "nonexistent.json")
