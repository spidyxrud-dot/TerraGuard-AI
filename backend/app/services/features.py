"""Environmental feature extraction service (Phase 4).

Extracts quantitative environmental metrics, vegetation signals, landscape
fragmentation statistics, and spectral shift features from bi-temporal Sentinel-2
observations and Siamese U-Net change detection maps.

Feature Categories
------------------
1. Area & Scale: Changed area (m2, ha, km2), valid fraction, change fraction.
2. Vegetation & NDVI Dynamics: Baseline vs post NDVI, mean NDVI shift in change zones,
   vegetation loss/gain areas (loss at -0.1, -0.2, -0.3 thresholds).
3. Landscape Fragmentation & Spatial Clusters: Connected component analysis
   (patch count, mean/max patch size, patch density per km2, large patch alerts).
4. Land-Cover Transitions (SCL): Categorical transitions (vegetation -> soil/urban,
   vegetation -> water, water depletion).
5. Spectral Reflectance Shifts: Band deltas (B02 Blue, B03 Green, B04 Red, B08 NIR)
   inside change zones.
6. Machine Learning Feature Vector: Standardized 1D vector and feature names for
   downstream XGBoost priority classification and SHAP explainability.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rasterio

from app.services.cloud_mask import SCL_VALID_CLASSES
from app.services.ndvi import calculate_ndvi, ndvi_difference

FEATURE_NAMES: tuple[str, ...] = (
    "changed_area_ha",
    "change_fraction_of_valid",
    "ndvi_before_mean",
    "ndvi_after_mean",
    "ndvi_diff_mean",
    "change_ndvi_before_mean",
    "change_ndvi_after_mean",
    "change_ndvi_diff_mean",
    "change_ndvi_diff_std",
    "veg_loss_ha_01",
    "veg_loss_ha_02",
    "veg_loss_ha_03",
    "veg_gain_ha_02",
    "veg_loss_fraction",
    "patch_count",
    "mean_patch_area_ha",
    "max_patch_area_ha",
    "patch_density_per_km2",
    "large_patch_count",
    "delta_b02_mean",
    "delta_b03_mean",
    "delta_b04_mean",
    "delta_b08_mean",
    "brightness_diff_mean",
    "scl_veg_loss_ha",
)
"""Standardized order of features consumed by downstream XGBoost priority model."""


@dataclass
class EnvironmentalFeatures:
    """Quantitative environmental change intelligence extracted from an AOI."""

    # Area metrics
    total_pixels: int
    valid_pixels: int
    valid_fraction: float
    total_area_ha: float
    valid_area_ha: float
    total_area_km2: float
    valid_area_km2: float
    changed_pixels: int
    changed_area_m2: float
    changed_area_ha: float
    changed_area_km2: float
    change_fraction_of_valid: float

    # Vegetation dynamics
    ndvi_before_mean: float | None
    ndvi_after_mean: float | None
    ndvi_diff_mean: float | None
    change_ndvi_before_mean: float | None
    change_ndvi_after_mean: float | None
    change_ndvi_diff_mean: float | None
    change_ndvi_diff_std: float | None
    veg_loss_pixels_01: int
    veg_loss_pixels_02: int
    veg_loss_pixels_03: int
    veg_gain_pixels_02: int
    veg_loss_ha_01: float
    veg_loss_ha_02: float
    veg_loss_ha_03: float
    veg_gain_ha_02: float
    veg_loss_fraction: float

    # Spatial clustering & fragmentation
    patch_count: int
    mean_patch_area_ha: float
    max_patch_area_ha: float
    patch_density_per_km2: float
    large_patch_count: int  # patches > 1.0 ha
    patch_sizes_ha: list[float] = field(default_factory=list)

    # Spectral band shifts in change zone
    delta_b02_mean: float | None = None
    delta_b03_mean: float | None = None
    delta_b04_mean: float | None = None
    delta_b08_mean: float | None = None
    brightness_diff_mean: float | None = None

    # SCL land-cover transitions
    scl_veg_to_nonveg_pixels: int = 0
    scl_veg_to_water_pixels: int = 0
    scl_water_to_nonveg_pixels: int = 0
    scl_nonveg_to_veg_pixels: int = 0
    scl_veg_loss_ha: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert features to JSON-serializable dictionary."""
        d = asdict(self)
        # Omit potentially large list in summary views
        d["patch_sizes_ha_summary"] = {
            "count": len(self.patch_sizes_ha),
            "max": self.max_patch_area_ha,
            "mean": self.mean_patch_area_ha,
        }
        return d

    def to_feature_dict(self) -> dict[str, float]:
        """Extract flat numeric dictionary aligned with FEATURE_NAMES."""
        d: dict[str, float] = {}
        for name in FEATURE_NAMES:
            val = getattr(self, name, 0.0)
            d[name] = float(val) if val is not None and np.isfinite(val) else 0.0
        return d

    def to_feature_vector(self) -> tuple[np.ndarray, list[str]]:
        """1D float32 array aligned with FEATURE_NAMES for ML inference."""
        d = self.to_feature_dict()
        vec = np.array([d[name] for name in FEATURE_NAMES], dtype=np.float32)
        return vec, list(FEATURE_NAMES)


def analyze_change_patches(binary_change_mask: np.ndarray,
                           pixel_area_m2: float = 100.0,
                           min_patch_pixels: int = 4) -> tuple[int, float, float, int, list[float]]:
    """Connected-component analysis on binary change mask.

    Filters tiny 1-3 pixel noise artifacts and computes patch spatial metrics.
    """
    mask_u8 = (binary_change_mask > 0).astype(np.uint8)
    if mask_u8.sum() == 0:
        return 0, 0.0, 0.0, 0, []

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)

    patch_sizes_ha: list[float] = []
    large_patch_count = 0

    # Label 0 is background; evaluate labels 1 to num_labels - 1
    for label_idx in range(1, num_labels):
        area_px = stats[label_idx, cv2.CC_STAT_AREA]
        if area_px < min_patch_pixels:
            continue
        area_ha = (area_px * pixel_area_m2) / 10_000.0
        patch_sizes_ha.append(round(area_ha, 4))
        if area_ha >= 1.0:
            large_patch_count += 1

    patch_count = len(patch_sizes_ha)
    if patch_count == 0:
        return 0, 0.0, 0.0, 0, []

    mean_patch_ha = round(float(np.mean(patch_sizes_ha)), 4)
    max_patch_ha = round(float(np.max(patch_sizes_ha)), 4)

    return patch_count, mean_patch_ha, max_patch_ha, large_patch_count, sorted(patch_sizes_ha, reverse=True)


def extract_environmental_features(
    change_mask: np.ndarray,
    change_probability: np.ndarray | None = None,
    ndvi_before: np.ndarray | None = None,
    ndvi_after: np.ndarray | None = None,
    ndvi_diff: np.ndarray | None = None,
    before_reflectance: np.ndarray | None = None,
    after_reflectance: np.ndarray | None = None,
    scl_before: np.ndarray | None = None,
    scl_after: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
    pixel_resolution_m: float = 10.0,
) -> EnvironmentalFeatures:
    """Extract comprehensive environmental features across satellite pair rasters."""
    pixel_area_m2 = float(pixel_resolution_m ** 2)

    # 1. Spatial validity
    shape = change_mask.shape[-2:]
    total_pixels = int(np.prod(shape))

    finite_mask = np.ones(shape, dtype=bool)
    if valid_mask is not None:
        finite_mask &= (np.asarray(valid_mask).squeeze() > 0)

    valid_pixels = int(finite_mask.sum())
    valid_fraction = round(valid_pixels / total_pixels, 6) if total_pixels else 0.0

    total_area_ha = round((total_pixels * pixel_area_m2) / 10_000.0, 4)
    valid_area_ha = round((valid_pixels * pixel_area_m2) / 10_000.0, 4)
    total_area_km2 = round(total_area_ha / 100.0, 6)
    valid_area_km2 = round(valid_area_ha / 100.0, 6)

    # Consider 255 in change_mask as nodata
    clean_change_mask = (np.asarray(change_mask).squeeze() == 1) & finite_mask
    changed_pixels = int(clean_change_mask.sum())
    changed_area_m2 = round(changed_pixels * pixel_area_m2, 2)
    changed_area_ha = round(changed_area_m2 / 10_000.0, 4)
    changed_area_km2 = round(changed_area_ha / 100.0, 6)
    change_fraction_of_valid = round(changed_pixels / valid_pixels, 6) if valid_pixels else 0.0

    # 2. NDVI Dynamics
    if ndvi_diff is None and ndvi_before is not None and ndvi_after is not None:
        ndvi_diff = ndvi_difference(ndvi_before, ndvi_after)

    def _masked_mean(arr: np.ndarray | None, mask: np.ndarray) -> float | None:
        if arr is None:
            return None
        vals = np.asarray(arr).squeeze()[mask]
        usable = vals[np.isfinite(vals)]
        return round(float(usable.mean()), 6) if usable.size > 0 else None

    def _masked_std(arr: np.ndarray | None, mask: np.ndarray) -> float | None:
        if arr is None:
            return None
        vals = np.asarray(arr).squeeze()[mask]
        usable = vals[np.isfinite(vals)]
        return round(float(usable.std()), 6) if usable.size > 0 else None

    ndvi_before_mean = _masked_mean(ndvi_before, finite_mask)
    ndvi_after_mean = _masked_mean(ndvi_after, finite_mask)
    ndvi_diff_mean = _masked_mean(ndvi_diff, finite_mask)

    change_ndvi_before_mean = _masked_mean(ndvi_before, clean_change_mask)
    change_ndvi_after_mean = _masked_mean(ndvi_after, clean_change_mask)
    change_ndvi_diff_mean = _masked_mean(ndvi_diff, clean_change_mask)
    change_ndvi_diff_std = _masked_std(ndvi_diff, clean_change_mask)

    # Vegetation threshold metrics
    veg_loss_pixels_01 = 0
    veg_loss_pixels_02 = 0
    veg_loss_pixels_03 = 0
    veg_gain_pixels_02 = 0

    if ndvi_diff is not None:
        diff_valid = np.asarray(ndvi_diff).squeeze()[finite_mask]
        diff_finite = diff_valid[np.isfinite(diff_valid)]
        if diff_finite.size > 0:
            veg_loss_pixels_01 = int((diff_finite <= -0.1).sum())
            veg_loss_pixels_02 = int((diff_finite <= -0.2).sum())
            veg_loss_pixels_03 = int((diff_finite <= -0.3).sum())
            veg_gain_pixels_02 = int((diff_finite >= 0.2).sum())

    veg_loss_ha_01 = round((veg_loss_pixels_01 * pixel_area_m2) / 10_000.0, 4)
    veg_loss_ha_02 = round((veg_loss_pixels_02 * pixel_area_m2) / 10_000.0, 4)
    veg_loss_ha_03 = round((veg_loss_pixels_03 * pixel_area_m2) / 10_000.0, 4)
    veg_gain_ha_02 = round((veg_gain_pixels_02 * pixel_area_m2) / 10_000.0, 4)
    veg_loss_fraction = round(veg_loss_pixels_02 / valid_pixels, 6) if valid_pixels else 0.0

    # 3. Spatial clustering & fragmentation
    p_count, p_mean, p_max, large_p_count, p_sizes = analyze_change_patches(
        clean_change_mask, pixel_area_m2=pixel_area_m2
    )
    valid_area_km2 = (valid_pixels * pixel_area_m2) / 1_000_000.0
    patch_density = round(p_count / valid_area_km2, 4) if valid_area_km2 > 0 else 0.0

    # 4. Spectral Band Shifts in Change Region
    delta_b02 = None
    delta_b03 = None
    delta_b04 = None
    delta_b08 = None
    brightness_diff = None

    if before_reflectance is not None and after_reflectance is not None:
        b_ref = np.asarray(before_reflectance)
        a_ref = np.asarray(after_reflectance)
        if b_ref.ndim == 3 and a_ref.ndim == 3 and b_ref.shape[0] >= 4:
            delta_b02 = _masked_mean(a_ref[0] - b_ref[0], clean_change_mask)
            delta_b03 = _masked_mean(a_ref[1] - b_ref[1], clean_change_mask)
            delta_b04 = _masked_mean(a_ref[2] - b_ref[2], clean_change_mask)
            delta_b08 = _masked_mean(a_ref[3] - b_ref[3], clean_change_mask)

            b_vis = b_ref[:3].mean(axis=0)
            a_vis = a_ref[:3].mean(axis=0)
            brightness_diff = _masked_mean(a_vis - b_vis, clean_change_mask)

    # 5. SCL Land-Cover Transitions
    scl_veg_to_nonveg = 0
    scl_veg_to_water = 0
    scl_water_to_nonveg = 0
    scl_nonveg_to_veg = 0
    scl_veg_loss_ha = 0.0

    if scl_before is not None and scl_after is not None:
        scl_b = np.asarray(scl_before).squeeze()
        scl_a = np.asarray(scl_after).squeeze()

        # Vegetation (4), Not Vegetated (5), Water (6)
        chg = clean_change_mask
        scl_veg_to_nonveg = int(((scl_b == 4) & (scl_a == 5) & chg).sum())
        scl_veg_to_water = int(((scl_b == 4) & (scl_a == 6) & chg).sum())
        scl_water_to_nonveg = int(((scl_b == 6) & (scl_a == 5) & chg).sum())
        scl_nonveg_to_veg = int(((scl_b == 5) & (scl_a == 4) & chg).sum())

        scl_veg_loss_pixels = scl_veg_to_nonveg + scl_veg_to_water
        scl_veg_loss_ha = round((scl_veg_loss_pixels * pixel_area_m2) / 10_000.0, 4)

    return EnvironmentalFeatures(
        total_pixels=total_pixels,
        valid_pixels=valid_pixels,
        valid_fraction=valid_fraction,
        total_area_ha=total_area_ha,
        valid_area_ha=valid_area_ha,
        total_area_km2=total_area_km2,
        valid_area_km2=valid_area_km2,
        changed_pixels=changed_pixels,
        changed_area_m2=changed_area_m2,
        changed_area_ha=changed_area_ha,
        changed_area_km2=changed_area_km2,
        change_fraction_of_valid=change_fraction_of_valid,
        ndvi_before_mean=ndvi_before_mean,
        ndvi_after_mean=ndvi_after_mean,
        ndvi_diff_mean=ndvi_diff_mean,
        change_ndvi_before_mean=change_ndvi_before_mean,
        change_ndvi_after_mean=change_ndvi_after_mean,
        change_ndvi_diff_mean=change_ndvi_diff_mean,
        change_ndvi_diff_std=change_ndvi_diff_std,
        veg_loss_pixels_01=veg_loss_pixels_01,
        veg_loss_pixels_02=veg_loss_pixels_02,
        veg_loss_pixels_03=veg_loss_pixels_03,
        veg_gain_pixels_02=veg_gain_pixels_02,
        veg_loss_ha_01=veg_loss_ha_01,
        veg_loss_ha_02=veg_loss_ha_02,
        veg_loss_ha_03=veg_loss_ha_03,
        veg_gain_ha_02=veg_gain_ha_02,
        veg_loss_fraction=veg_loss_fraction,
        patch_count=p_count,
        mean_patch_area_ha=p_mean,
        max_patch_area_ha=p_max,
        patch_density_per_km2=patch_density,
        large_patch_count=large_p_count,
        patch_sizes_ha=p_sizes,
        delta_b02_mean=delta_b02,
        delta_b03_mean=delta_b03,
        delta_b04_mean=delta_b04,
        delta_b08_mean=delta_b08,
        brightness_diff_mean=brightness_diff,
        scl_veg_to_nonveg_pixels=scl_veg_to_nonveg,
        scl_veg_to_water_pixels=scl_veg_to_water,
        scl_water_to_nonveg_pixels=scl_water_to_nonveg,
        scl_nonveg_to_veg_pixels=scl_nonveg_to_veg,
        scl_veg_loss_ha=scl_veg_loss_ha,
    )


def extract_features_from_processed(
    processed_dir: str | Path,
    predictions_dir: str | Path | None = None,
    change_mask_path: str | Path | None = None,
) -> EnvironmentalFeatures:
    """Extract features by reading rasters and predictions from disk."""
    proc_path = Path(processed_dir)
    if not proc_path.is_dir():
        raise FileNotFoundError(f"Processed directory not found: {proc_path}")

    # Locate change mask
    if change_mask_path is not None:
        mask_file = Path(change_mask_path)
    elif predictions_dir is not None:
        mask_file = Path(predictions_dir) / "change_mask.tif"
    else:
        mask_file = proc_path / "change_mask.tif"

    if not mask_file.is_file():
        raise FileNotFoundError(f"Change mask raster not found at: {mask_file}")

    with rasterio.open(mask_file) as src:
        change_mask = src.read(1)
        res_m = abs(src.transform[0])

    def _read_optional_tif(name: str) -> np.ndarray | None:
        p = proc_path / name
        if p.is_file():
            with rasterio.open(p) as src:
                return src.read(1)
        return None

    def _read_optional_npy(name: str) -> np.ndarray | None:
        p = proc_path / name
        if p.is_file():
            return np.load(p)
        return None

    ndvi_before = _read_optional_tif("ndvi_before.tif")
    ndvi_after = _read_optional_tif("ndvi_after.tif")
    ndvi_diff = _read_optional_tif("ndvi_difference.tif")

    scl_before = _read_optional_tif("scl_before.tif")
    scl_after = _read_optional_tif("scl_after.tif")

    before_ref = _read_optional_npy("before.npy")
    after_ref = _read_optional_npy("after.npy")
    valid_mask = _read_optional_npy("valid_mask.npy")

    return extract_environmental_features(
        change_mask=change_mask,
        ndvi_before=ndvi_before,
        ndvi_after=ndvi_after,
        ndvi_diff=ndvi_diff,
        before_reflectance=before_ref,
        after_reflectance=after_ref,
        scl_before=scl_before,
        scl_after=scl_after,
        valid_mask=valid_mask,
        pixel_resolution_m=res_m,
    )
