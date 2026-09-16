"""Shared raster/CRS helpers for the TerraGuard geospatial pipeline.

Everything downstream of acquisition (preprocessing, feature extraction, model
inference) assumes a single :class:`RasterGrid`: a north-up, axis-aligned grid
with one CRS, one resolution and a transform shared by every band and by both
observations of the pair. Pixels with the same ``(row, col)`` therefore refer to
the same geographic location in the before and after image.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine, from_origin
from rasterio.warp import reproject, transform_bounds

DEFAULT_RESOLUTION_M = 10.0
"""Sentinel-2 B02/B03/B04/B08 native ground sample distance."""

NAN = float("nan")


@dataclass(frozen=True)
class RasterGrid:
    """A north-up, axis-aligned raster grid (CRS + transform + size)."""

    crs: str
    transform: Affine
    width: int
    height: int

    @property
    def resolution(self) -> tuple[float, float]:
        return (abs(self.transform.a), abs(self.transform.e))

    @property
    def resolution_m(self) -> float:
        return abs(self.transform.a)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        min_x = self.transform.c
        max_y = self.transform.f
        max_x = min_x + self.transform.a * self.width
        min_y = max_y + self.transform.e * self.height
        return (min_x, min_y, max_x, max_y)

    @property
    def pixel_area_m2(self) -> float:
        return self.resolution_m * abs(self.transform.e)

    def to_dict(self) -> dict:
        return {
            "crs": self.crs,
            "width": int(self.width),
            "height": int(self.height),
            "resolution_m": float(self.resolution_m),
            "bounds": [float(v) for v in self.bounds],
            "transform": [float(v) for v in tuple(self.transform)[:6]],
            "pixel_area_m2": float(self.pixel_area_m2),
            "area_km2": round(self.pixel_area_m2 * self.width * self.height / 1e6, 4),
        }

    def profile(self, *, count: int, dtype: str, nodata: float | None = None, compress: str = "deflate") -> dict:
        profile = {
            "driver": "GTiff",
            "height": int(self.height),
            "width": int(self.width),
            "count": int(count),
            "dtype": dtype,
            "crs": self.crs,
            "transform": self.transform,
            "compress": compress,
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
        }
        if nodata is not None:
            profile["nodata"] = nodata
        return profile


def grid_from_bounds(bounds, resolution: float, crs) -> RasterGrid:
    """Build a grid covering ``bounds`` with pixel edges snapped to ``resolution``."""
    if resolution <= 0:
        raise ValueError(f"resolution must be positive, got {resolution}")
    min_x, min_y, max_x, max_y = (float(v) for v in bounds)
    snapped_min_x = math.floor(min_x / resolution) * resolution
    snapped_min_y = math.floor(min_y / resolution) * resolution
    snapped_max_x = math.ceil(max_x / resolution) * resolution
    snapped_max_y = math.ceil(max_y / resolution) * resolution
    width = int(round((snapped_max_x - snapped_min_x) / resolution))
    height = int(round((snapped_max_y - snapped_min_y) / resolution))
    if width <= 0 or height <= 0:
        raise ValueError(f"degenerate grid for bounds {bounds} at {resolution} m")
    transform = from_origin(snapped_min_x, snapped_max_y, resolution, resolution)
    return RasterGrid(crs=rasterio.crs.CRS.from_user_input(crs).to_string(), transform=transform,
                      width=width, height=height)


def raster_profile(path: str | Path) -> dict:
    """Return a JSON-friendly metadata summary of a raster band file."""
    with rasterio.open(path) as src:
        return {
            "path": str(path),
            "driver": src.driver,
            "count": int(src.count),
            "width": int(src.width),
            "height": int(src.height),
            "dtype": src.dtypes[0],
            "crs": src.crs.to_string() if src.crs else None,
            "resolution": [float(src.res[0]), float(src.res[1])],
            "bounds": [float(v) for v in src.bounds],
            "transform": [float(v) for v in tuple(src.transform)[:6]],
            "nodata": None if src.nodata is None else float(src.nodata),
        }


def reproject_bounds(bounds, src_crs, dst_crs) -> tuple[float, float, float, float]:
    """Transform ``bounds`` from ``src_crs`` into ``dst_crs``."""
    if rasterio.crs.CRS.from_user_input(src_crs) == rasterio.crs.CRS.from_user_input(dst_crs):
        return tuple(float(v) for v in bounds)
    return tuple(float(v) for v in transform_bounds(src_crs, dst_crs, *bounds))


def intersect_bounds(a, b) -> tuple[float, float, float, float] | None:
    """Geographic intersection of two bounds, or ``None`` when they do not overlap."""
    min_x = max(float(a[0]), float(b[0]))
    min_y = max(float(a[1]), float(b[1]))
    max_x = min(float(a[2]), float(b[2]))
    max_y = min(float(a[3]), float(b[3]))
    if min_x >= max_x or min_y >= max_y:
        return None
    return (min_x, min_y, max_x, max_y)


def common_grid(paths, resolution: float | None = None, crs=None) -> RasterGrid:
    """Build the shared analysis grid for one or more rasters.

    All raster bounds are reprojected into the target CRS, intersected, and snapped
    to ``resolution`` so the resulting grid has clean pixel edges. The resolution
    defaults to the finest resolution among the inputs.
    """
    paths = [Path(p) for p in paths]
    if not paths:
        raise ValueError("common_grid requires at least one raster path")

    profiles = [raster_profile(p) for p in paths]
    target_crs = rasterio.crs.CRS.from_user_input(crs) if crs else rasterio.crs.CRS.from_user_input(profiles[0]["crs"])
    if target_crs is None:
        raise ValueError("rasters have no CRS; a common grid cannot be defined")

    if resolution is None:
        resolution = min(min(p["resolution"]) for p in profiles)

    intersection = None
    for profile in profiles:
        bounds = reproject_bounds(profile["bounds"], profile["crs"], target_crs)
        intersection = bounds if intersection is None else intersect_bounds(intersection, bounds)
        if intersection is None:
            raise ValueError(f"rasters do not overlap in {target_crs}: {[str(p) for p in paths]}")
    return grid_from_bounds(intersection, resolution, target_crs)


def read_band(path: str | Path, band: int = 1, masked: bool = True) -> np.ndarray:
    """Read a single band as float32. With ``masked`` the nodata value becomes NaN."""
    with rasterio.open(path) as src:
        data = src.read(band).astype("float32")
        if masked and src.nodata is not None:
            data = np.where(data == np.float32(src.nodata), np.float32(NAN), data)
    return data


def valid_mask(array: np.ndarray) -> np.ndarray:
    """Boolean mask of finite values."""
    return np.isfinite(array)


def resample_to_grid(path: str | Path, grid: RasterGrid, band: int = 1,
                     resampling: Resampling = Resampling.bilinear,
                     dst_nodata: float | None = NAN) -> np.ndarray:
    """Reproject/resample one band of ``path`` onto ``grid`` (float32, nodata NaN)."""
    destination = np.full((grid.height, grid.width), dst_nodata, dtype="float32")
    with rasterio.open(path) as src:
        reproject(
            source=rasterio.band(src, band),
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=grid.transform,
            dst_crs=grid.crs,
            dst_nodata=dst_nodata,
            resampling=resampling,
        )
    return destination


def resample_mask_to_grid(path: str | Path, grid: RasterGrid, band: int = 1) -> np.ndarray:
    """Nearest-neighbour resample of a raster's validity mask onto ``grid``."""
    destination = np.zeros((grid.height, grid.width), dtype="uint8")
    with rasterio.open(path) as src:
        source = (src.read(band) != src.nodata).astype("uint8") if src.nodata is not None \
            else np.ones((src.height, src.width), dtype="uint8")
        reproject(
            source=source,
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=None,
            dst_transform=grid.transform,
            dst_crs=grid.crs,
            dst_nodata=0,
            resampling=Resampling.nearest,
        )
    return destination.astype(bool)

