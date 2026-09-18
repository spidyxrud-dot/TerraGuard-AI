"""Tests for priority dataset and label provenance (Phase 4 / Priority Model)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ml.priority.dataset import (
    CLASS_NAMES,
    LabelProvenance,
    PriorityClass,
    PriorityDataset,
    derive_prototype_priority_label,
    generate_prototype_dataset,
    stratified_split,
)


def test_derive_prototype_priority_label_low() -> None:
    features = {
        "changed_area_ha": 0.2,
        "change_fraction_of_valid": 0.002,
        "veg_loss_ha_02": 0.0,
        "veg_loss_ha_03": 0.0,
        "large_patch_count": 0,
        "scl_veg_loss_ha": 0.0,
    }
    priority, details = derive_prototype_priority_label(features)
    assert priority == PriorityClass.LOW
    assert details["priority"] == "LOW"
    assert details["composite_score"] < 25.0


def test_derive_prototype_priority_label_medium() -> None:
    features = {
        "changed_area_ha": 4.5,
        "change_fraction_of_valid": 0.05,
        "veg_loss_ha_02": 2.5,
        "veg_loss_ha_03": 0.2,
        "large_patch_count": 1,
        "scl_veg_loss_ha": 0.8,
    }
    priority, details = derive_prototype_priority_label(features)
    assert priority == PriorityClass.MEDIUM
    assert details["priority"] == "MEDIUM"


def test_derive_prototype_priority_label_high() -> None:
    features = {
        "changed_area_ha": 15.0,
        "change_fraction_of_valid": 0.18,
        "veg_loss_ha_02": 12.0,
        "veg_loss_ha_03": 5.0,
        "large_patch_count": 4,
        "scl_veg_loss_ha": 6.5,
    }
    priority, details = derive_prototype_priority_label(features)
    assert priority == PriorityClass.HIGH
    assert details["priority"] == "HIGH"
    assert details["composite_score"] >= 55.0


def test_generate_prototype_dataset_contract() -> None:
    ds = generate_prototype_dataset(n_samples=100, seed=7)
    assert len(ds) == 100
    assert ds.X.shape == (100, 25)
    assert ds.y.shape == (100,)
    assert ds.label_source == LabelProvenance.RULE_DERIVED_PROTOTYPE

    d = ds.to_dict()
    assert d["is_prototype_rule_derived"] is True
    assert set(d["class_counts"].keys()) == set(CLASS_NAMES)
    assert all(count > 0 for count in d["class_counts"].values())


def test_stratified_split() -> None:
    ds = generate_prototype_dataset(n_samples=150, seed=42)
    train_ds, val_ds, test_ds = stratified_split(ds, val_fraction=0.2, test_fraction=0.2, seed=42)

    assert len(train_ds) + len(val_ds) + len(test_ds) == 150
    assert len(val_ds) == 30
    assert len(test_ds) == 30
    assert len(train_ds) == 90

    # Ensure all classes exist in all splits
    for s in (train_ds, val_ds, test_ds):
        assert set(np.unique(s.y)) == {0, 1, 2}


def test_csv_round_trip(tmp_path: Path) -> None:
    csv_file = tmp_path / "dataset.csv"
    original = generate_prototype_dataset(n_samples=50, seed=12)
    original.save_csv(csv_file)

    assert csv_file.is_file()
    restored = PriorityDataset.load_csv(csv_file)

    assert len(restored) == len(original)
    assert np.allclose(restored.X, original.X, atol=1e-5)
    assert np.array_equal(restored.y, original.y)
