"""Tests for environmental feature extraction (Phase 4).

Covers:
- Area calculation and valid pixel accounting
- Vegetation dynamics (NDVI before/after, vegetation loss thresholds at -0.1, -0.2, -0.3)
- Spatial clustering & connected components (patch count, mean/max area, large patches)
- Spectral band deltas (B02, B03, B04, B08)
- SCL land-cover transition statistics
- ML feature vector serialization and feature names alignment
- Integration with preprocessed Sentinel-2 pairs
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from app.services.features import (
    FEATURE_NAMES,
    EnvironmentalFeatures,
    analyze_change_patches,
    extract_environmental_features,
    extract_features_from_processed,
)


def test_analyze_change_patches_empty() -> None:
    mask = np.zeros((50, 50), dtype=np.uint8)
    count, mean_ha, max_ha, large_count, sizes = analyze_change_patches(mask)
    assert count == 0
    assert mean_ha == 0.0
    assert max_ha == 0.0
    assert large_count == 0
    assert sizes == []


def test_analyze_change_patches_known_blocks() -> None:
    mask = np.zeros((100, 100), dtype=np.uint8)
    # Patch 1: 10x10 = 100 px = 10,000 m2 = 1.0 ha
    mask[10:20, 10:20] = 1
    # Patch 2: 20x20 = 400 px = 40,000 m2 = 4.0 ha
    mask[40:60, 40:60] = 1

    count, mean_ha, max_ha, large_count, sizes = analyze_change_patches(mask, pixel_area_m2=100.0)
    assert count == 2
    assert max_ha == 4.0
    assert mean_ha == 2.5
    assert large_count == 2
    assert sizes == [4.0, 1.0]


def test_extract_environmental_features_zero_change() -> None:
    shape = (50, 50)
    change_mask = np.zeros(shape, dtype=np.uint8)
    ndvi_before = np.full(shape, 0.6, dtype=np.float32)
    ndvi_after = np.full(shape, 0.6, dtype=np.float32)
    valid_mask = np.ones(shape, dtype=bool)

    features = extract_environmental_features(
        change_mask=change_mask,
        ndvi_before=ndvi_before,
        ndvi_after=ndvi_after,
        valid_mask=valid_mask,
        pixel_resolution_m=10.0,
    )

    assert features.total_pixels == 2500
    assert features.valid_pixels == 2500
    assert features.changed_pixels == 0
    assert features.changed_area_ha == 0.0
    assert features.patch_count == 0
    assert features.ndvi_before_mean == pytest.approx(0.6, abs=1e-5)
    assert features.ndvi_after_mean == pytest.approx(0.6, abs=1e-5)
    assert features.ndvi_diff_mean == pytest.approx(0.0, abs=1e-5)


def test_extract_environmental_features_deforestation_signal() -> None:
    shape = (100, 100)
    change_mask = np.zeros(shape, dtype=np.uint8)
    # 20x20 change block (400 px = 4.0 ha)
    change_mask[20:40, 20:40] = 1

    ndvi_before = np.full(shape, 0.65, dtype=np.float32)
    ndvi_after = np.full(shape, 0.65, dtype=np.float32)
    # Severe NDVI drop in change zone
    ndvi_after[20:40, 20:40] = 0.20

    # Reflectance (4 bands: B02, B03, B04, B08)
    before_ref = np.zeros((4, 100, 100), dtype=np.float32)
    after_ref = np.zeros((4, 100, 100), dtype=np.float32)
    before_ref[2] = 0.05  # Red before (low in forest)
    before_ref[3] = 0.40  # NIR before (high in forest)
    after_ref[2] = 0.20   # Red after (soil exposed)
    after_ref[3] = 0.15   # NIR after (biomass lost)

    # SCL: 4 (vegetation) before -> 5 (not vegetated) after
    scl_before = np.full(shape, 4, dtype=np.uint8)
    scl_after = np.full(shape, 4, dtype=np.uint8)
    scl_after[20:40, 20:40] = 5

    valid_mask = np.ones(shape, dtype=bool)

    features = extract_environmental_features(
        change_mask=change_mask,
        ndvi_before=ndvi_before,
        ndvi_after=ndvi_after,
        before_reflectance=before_ref,
        after_reflectance=after_ref,
        scl_before=scl_before,
        scl_after=scl_after,
        valid_mask=valid_mask,
        pixel_resolution_m=10.0,
    )

    assert features.changed_pixels == 400
    assert features.changed_area_ha == 4.0
    assert features.patch_count == 1
    assert features.max_patch_area_ha == 4.0
    assert features.large_patch_count == 1

    # Check NDVI drop inside change
    assert features.change_ndvi_diff_mean == pytest.approx(-0.45, abs=1e-3)
    assert features.veg_loss_ha_02 == 4.0
    assert features.veg_loss_ha_03 == 4.0

    # Check band deltas: Red increased, NIR decreased
    assert features.delta_b04_mean == pytest.approx(0.15, abs=1e-3)
    assert features.delta_b08_mean == pytest.approx(-0.25, abs=1e-3)

    # Check SCL transition: 400 px = 4.0 ha veg loss
    assert features.scl_veg_to_nonveg_pixels == 400
    assert features.scl_veg_loss_ha == 4.0


def test_feature_vector_serialization() -> None:
    shape = (50, 50)
    change_mask = np.zeros(shape, dtype=np.uint8)
    change_mask[10:20, 10:20] = 1
    features = extract_environmental_features(
        change_mask=change_mask,
        ndvi_before=np.full(shape, 0.5, dtype=np.float32),
        ndvi_after=np.full(shape, 0.3, dtype=np.float32),
    )

    vec, names = features.to_feature_vector()
    assert len(vec) == len(FEATURE_NAMES)
    assert names == list(FEATURE_NAMES)
    assert vec.dtype == np.float32
    assert np.all(np.isfinite(vec)), "Feature vector must be completely finite (no NaNs/Infs)"

    d = features.to_dict()
    assert "changed_area_ha" in d
    assert "patch_sizes_ha_summary" in d


def test_extract_features_from_processed_directory(tmp_path: Path) -> None:
    proc_dir = tmp_path / "processed"
    proc_dir.mkdir(parents=True, exist_ok=True)

    height, width = 40, 40
    transform = from_origin(500_000.0, 2_000_000.0, 10.0, 10.0)
    crs = "EPSG:32643"

    prof = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
    }

    # Write change mask
    mask_prof = dict(prof, dtype="uint8", nodata=255)
    mask_arr = np.zeros((height, width), dtype=np.uint8)
    mask_arr[10:20, 10:20] = 1
    with rasterio.open(proc_dir / "change_mask.tif", "w", **mask_prof) as dst:
        dst.write(mask_arr, 1)

    # Write ndvi rasters
    with rasterio.open(proc_dir / "ndvi_before.tif", "w", **prof) as dst:
        dst.write(np.full((height, width), 0.5, dtype=np.float32), 1)
    with rasterio.open(proc_dir / "ndvi_after.tif", "w", **prof) as dst:
        dst.write(np.full((height, width), 0.2, dtype=np.float32), 1)

    features = extract_features_from_processed(proc_dir)
    assert isinstance(features, EnvironmentalFeatures)
    assert features.changed_pixels == 100
    assert features.changed_area_ha == 1.0
