"""NDVI computation for the TerraGuard change-detection pipeline.

NDVI = (NIR - RED) / (NIR + RED) computed on surface reflectance (B08 = NIR,
B04 = RED for Sentinel-2). Invalid pixels carry NaN instead of a fabricated
number, so every downstream statistic or feature has to acknowledge them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio

from app.utils.geo import RasterGrid

NDVI_FORMULA = "(B08 - B04) / (B08 + B04)"
NDVI_RANGE = (-1.0, 1.0)

CHANGE_THRESHOLDS = (0.1, 0.2, 0.3)
"""Absolute NDVI deltas reported as meaningful vegetation change."""

NODATA = float("nan")


def calculate_ndvi(nir: np.ndarray, red: np.ndarray, valid_mask: np.ndarray | None = None,
                   eps: float = 1e-6) -> np.ndarray:
    """Compute NDVI with NaN propagation for invalid pixels.

    A pixel is invalid when it is masked out, non-finite, or has a reflectance sum
    near zero (which would otherwise produce a division artifact).
    """
    nir = np.asarray(nir, dtype="float32")
    red = np.asarray(red, dtype="float32")
    if nir.shape != red.shape:
        raise ValueError(f"nir and red must have the same shape, got {nir.shape} and {red.shape}")

    usable = np.isfinite(nir) & np.isfinite(red)
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != nir.shape:
            raise ValueError(f"valid_mask shape {mask.shape} does not match bands {nir.shape}")
        usable &= mask

    denominator = nir + red
    usable &= np.abs(denominator) > eps

    ndvi = np.full(nir.shape, np.float32(np.nan), dtype="float32")
    with np.errstate(invalid="ignore", divide="ignore"):
        np.divide(nir - red, denominator, out=ndvi, where=usable)
    return np.clip(ndvi, NDVI_RANGE[0], NDVI_RANGE[1]).astype("float32")


def ndvi_difference(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """Per-pixel NDVI change (after - before); NaN propagates from either input."""
    before = np.asarray(before, dtype="float32")
    after = np.asarray(after, dtype="float32")
    if before.shape != after.shape:
        raise ValueError(f"NDVI rasters must share a shape, got {before.shape} and {after.shape}")
    difference = np.full(before.shape, np.float32(np.nan), dtype="float32")
    usable = np.isfinite(before) & np.isfinite(after)
    difference[usable] = after[usable] - before[usable]
    return difference


def describe(array: np.ndarray, prefix: str = "") -> dict:
    """Summary statistics that ignore NaN pixels."""
    values = np.asarray(array, dtype="float32")
    total = int(values.size)
    finite = np.isfinite(values)
    valid = int(finite.sum())
    stats: dict = {
        f"{prefix}valid_pixels" if prefix else "valid_pixels": valid,
        f"{prefix}invalid_pixels" if prefix else "invalid_pixels": total - valid,
        f"{prefix}valid_fraction" if prefix else "valid_fraction":
            round(valid / total, 6) if total else 0.0,
    }
    if not valid:
        stats.update({(prefix + key if prefix else key): None for key in ("min", "max", "mean", "std", "median")})
        return stats
    sample = values[finite]
    for key, value in (
        ("min", float(sample.min())),
        ("max", float(sample.max())),
        ("mean", float(sample.mean())),
        ("std", float(sample.std())),
        ("median", float(np.median(sample))),
    ):
        stats[f"{prefix}{key}" if prefix else key] = round(value, 6)
    return stats


def change_area_fractions(difference: np.ndarray, thresholds=CHANGE_THRESHOLDS) -> dict:
    """Fraction of valid pixels that lost or gained at least ``threshold`` NDVI."""
    difference = np.asarray(difference, dtype="float32")
    finite = np.isfinite(difference)
    valid = int(finite.sum())
    report: dict = {"valid_pixels": valid}
    if not valid:
        return report
    values = difference[finite]
    for threshold in thresholds:
        loss = int((values <= -threshold).sum())
        gain = int((values >= threshold).sum())
        key = f"{threshold:.1f}"
        report[f"loss_at_{key}_pixels"] = loss
        report[f"gain_at_{key}_pixels"] = gain
        report[f"loss_at_{key}_fraction"] = round(loss / valid, 6)
        report[f"gain_at_{key}_fraction"] = round(gain / valid, 6)
    return report


def write_ndvi_raster(path: str | Path, values: np.ndarray, grid: RasterGrid, description: str = "") -> None:
    """Write a float32 NDVI raster (NaNs preserved as nodata) onto the analysis grid."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = grid.profile(count=1, dtype="float32", nodata=NODATA)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.asarray(values, dtype="float32"), 1)
        if description:
            dst.update_tags(description=description, formula=NDVI_FORMULA)
