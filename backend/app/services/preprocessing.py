"""Sentinel-2 pair preprocessing: raw bands -> aligned, normalized tensors.

Contract with the rest of TerraGuard
------------------------------------
The Siamese U-Net never sees a file path, a CRS or an acquisition date. It sees

    before : Tensor[4, H, W]   float32 surface reflectance in [0, 1]
    after  : Tensor[4, H, W]

in band order ``B02, B03, B04, B08`` on one shared grid, where ``(row, col)`` means
the same geographic location in both images. Every function in this module exists to
make that statement true, explicit and reproducible - regardless of whether the
imagery came from a local GeoTIFF, Copernicus, Sentinel Hub or Google Earth Engine.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import numpy as np
import rasterio

from app.services.cloud_mask import (
    SCL_BAND,
    SCL_CLASSES,
    SCL_MASKED_CLASSES,
    SCL_VALID_CLASSES,
    SCL_VALID_CLASSES_LENIENT,
    CloudMask,
    load_scl,
    mask_observation,
    pair_valid_mask,
    scl_mask_report,
    write_scl_raster,
)
from app.services.ndvi import (
    CHANGE_THRESHOLDS,
    NDVI_FORMULA,
    calculate_ndvi,
    change_area_fractions,
    describe,
    ndvi_difference,
    write_ndvi_raster,
)
from app.utils.geo import (
    DEFAULT_RESOLUTION_M,
    RasterGrid,
    common_grid,
    intersect_bounds,
    raster_profile,
    resample_mask_to_grid,
    resample_to_grid,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW_DIR = REPO_ROOT / "data" / "raw" / "demo_area"
DEFAULT_PROCESSED_DIR = REPO_ROOT / "data" / "processed" / "demo_area"

BAND_ORDER: tuple[str, ...] = ("B02", "B03", "B04", "B08")
"""Band order of the tensor channels (all 10 m native Sentinel-2 bands)."""

BAND_ROLES: dict[str, str] = {"B02": "blue", "B03": "green", "B04": "red", "B08": "nir"}
RED_BAND = "B04"
NIR_BAND = "B08"
AUX_BANDS: tuple[str, ...] = (SCL_BAND,)
"""Auxiliary raw assets (not tensor channels) staged next to the bands.

``SCL`` drives the per-pixel cloud / invalid masking in ``app.services.cloud_mask``. It is
optional: when it is absent (OSCD, LEVIR-CD, non-Sentinel sources) masking degrades to the
band nodata and footprint coverage rules only.
"""

REFLECTANCE_SCALE = 10_000.0
"""Sentinel-2 L2A digital numbers are surface reflectance x 10000."""

REFLECTANCE_RANGE = (0.0, 1.0)
NODATA_VALUE = 0.0
"""Written to disk for invalid pixels; the authoritative validity lives in the masks."""

NORMALIZATION_DESCRIPTION = "surface_reflectance = clip(DN / 10000, 0, 1)"


@dataclass(frozen=True)
class BandRaster:
    """One raw band file plus its geospatial metadata."""

    band: str
    path: Path
    profile: dict

    @property
    def crs(self) -> str | None:
        return self.profile["crs"]

    @property
    def resolution(self) -> tuple[float, float]:
        return tuple(self.profile["resolution"])  # type: ignore[return-value]

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return tuple(self.profile["bounds"])  # type: ignore[return-value]

    @property
    def shape(self) -> tuple[int, int]:
        return (self.profile["height"], self.profile["width"])


@dataclass
class Observation:
    """A single acquisition (before or after) made of raw per-band GeoTIFFs."""

    label: str
    directory: Path
    bands: dict[str, BandRaster]
    aux: dict[str, BandRaster] = field(default_factory=dict)
    acquisition: dict = field(default_factory=dict)

    @property
    def date(self) -> str | None:
        return self.acquisition.get("acquisition_date")

    @property
    def item_id(self) -> str | None:
        return self.acquisition.get("item_id")

    @property
    def cloud_cover(self) -> float | None:
        value = self.acquisition.get("cloud_cover_percent")
        return None if value is None else float(value)

    @property
    def mgrs_tile(self) -> str | None:
        return self.acquisition.get("mgrs_grid")

    @property
    def paths(self) -> list[Path]:
        return [self.bands[band].path for band in BAND_ORDER]

    def summary(self) -> dict:
        return {
            "label": self.label,
            "directory": str(self.directory),
            "acquisition_date": self.date,
            "item_id": self.item_id,
            "mgrs_tile": self.mgrs_tile,
            "cloud_cover_percent": self.cloud_cover,
            "bands": {
                band: {**raster.profile, "path": str(raster.path), "band": raster.band}
                for band, raster in self.bands.items()
            },
            "aux": {band: {**raster.profile, "path": str(raster.path), "band": raster.band}
                    for band, raster in self.aux.items()},
        }


@dataclass
class ValidationResult:
    """Outcome of a set of named checks."""

    label: str
    checks: dict[str, bool] = field(default_factory=dict)
    details: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(self.checks.values())

    @property
    def problems(self) -> list[str]:
        return [name for name, ok in self.checks.items() if not ok]

    def to_dict(self) -> dict:
        return {"label": self.label, "passed": self.passed, "checks": self.checks, "details": self.details}


def load_sentinel_pair(raw_dir: str | Path = DEFAULT_RAW_DIR) -> tuple[Observation, Observation]:
    """Load ``before``/``after`` observations from ``<raw_dir>/{before,after}/<band>.tif``.

    Raw files are only read, never modified. Missing bands raise immediately so a
    broken pair can never silently reach the model.
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.is_dir():
        raise FileNotFoundError(
            f"raw data directory not found: {raw_dir}\n"
            "Fetch the demo pair first: backend/.venv/Scripts/python.exe scripts/fetch_sentinel_pair.py"
        )

    provenance: dict = {}
    provenance_path = raw_dir / "acquisition.json"
    if provenance_path.is_file():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))

    observations: list[Observation] = []
    for label in ("before", "after"):
        directory = raw_dir / label
        if not directory.is_dir():
            raise FileNotFoundError(f"missing observation directory: {directory}")

        bands: dict[str, BandRaster] = {}
        missing: list[str] = []
        for band in BAND_ORDER:
            path = directory / f"{band}.tif"
            if not path.is_file():
                missing.append(str(path))
                continue
            bands[band] = BandRaster(band=band, path=path, profile=raster_profile(path))
        if missing:
            raise FileNotFoundError("missing required band(s):\n  " + "\n  ".join(missing))

        aux: dict[str, BandRaster] = {}
        for band in AUX_BANDS:
            path = directory / f"{band}.tif"
            if path.is_file():
                aux[band] = BandRaster(band=band, path=path, profile=raster_profile(path))

        observations.append(Observation(label=label, directory=directory, bands=bands, aux=aux,
                                        acquisition=provenance.get(label, {})))
    return observations[0], observations[1]


def aux_scl_checks(observation: Observation) -> dict | None:
    """Sanity-check the raw SCL asset of an observation (``None`` when it has none).

    The SCL band decides which pixels reach the model, so a corrupt or unexpected SCL band
    must fail validation loudly rather than silently mask nonsense.
    """
    raster = observation.aux.get(SCL_BAND)
    if raster is None:
        return None

    checks: dict[str, bool] = {"scl_present": True, "scl_readable": False,
                               "scl_classes_known": False, "scl_has_usable_pixels": False}
    details: dict = {"path": str(raster.path), "resolution": list(raster.resolution),
                     "shape": [int(raster.shape[0]), int(raster.shape[1])]}
    try:
        classes = load_scl(raster.path)
    except Exception as error:  # pragma: no cover - depends on corrupt input
        details["error"] = f"{type(error).__name__}: {error}"
        return {"checks": checks, "details": details}

    report = scl_mask_report(classes)
    checks["scl_readable"] = True
    checks["scl_classes_known"] = not report["unknown_classes"]
    checks["scl_has_usable_pixels"] = report["usable_pixels"] > 0
    details.update({key: report[key] for key in ("usable_pixels", "masked_pixels", "usable_fraction",
                                                 "masked_fraction", "unknown_classes", "histogram",
                                                 "mask_reasons")})
    return {"checks": checks, "details": details}


def validate_raster(observation: Observation) -> ValidationResult:
    """Validate one observation: readability, bands, CRS, resolution, values, nodata."""
    result = ValidationResult(label=observation.label)
    details: dict = {"bands": {}}

    readable = True
    for band, raster in observation.bands.items():
        try:
            with rasterio.open(raster.path) as src:
                data = src.read(1)
        except Exception as error:  # pragma: no cover - depends on corrupt input
            readable = False
            result.checks[f"band_readable_{band}"] = False
            details["bands"][band] = {"error": f"{type(error).__name__}: {error}"}
            continue

        result.checks[f"band_readable_{band}"] = True
        nodata = raster.profile["nodata"]
        finite = np.isfinite(data.astype("float64"))
        usable = data[finite & (data != nodata)] if nodata is not None else data[finite]
        details["bands"][band] = {
            "dtype": raster.profile["dtype"],
            "shape": [int(data.shape[0]), int(data.shape[1])],
            "nodata": nodata,
            "non_finite_pixels": int((~finite).sum()),
            "nodata_pixels": int((data == nodata).sum()) if nodata is not None else 0,
            "min": float(usable.min()) if usable.size else None,
            "max": float(usable.max()) if usable.size else None,
            "mean": round(float(usable.mean()), 3) if usable.size else None,
            "std": round(float(usable.std()), 3) if usable.size else None,
            "valid_fraction": round(float(usable.size / data.size), 6) if data.size else 0.0,
        }
        result.checks[f"band_has_variation_{band}"] = bool(usable.size and float(usable.std()) > 0.0)
        result.checks[f"band_values_in_sensor_range_{band}"] = bool(
            usable.size and float(usable.min()) >= 0.0 and float(usable.max()) <= 65535.0
        )

    result.checks["readable"] = readable and len(observation.bands) == len(BAND_ORDER)
    result.checks["bands_present"] = all(band in observation.bands for band in BAND_ORDER)
    result.checks["no_non_finite_pixels"] = all(
        stats.get("non_finite_pixels", 1) == 0 for stats in details["bands"].values() if "error" not in stats
    )

    crs_values = {raster.crs for raster in observation.bands.values()}
    result.checks["crs_valid"] = all(value is not None for value in crs_values)
    result.checks["crs_consistent_across_bands"] = len(crs_values) == 1

    resolutions = {raster.resolution for raster in observation.bands.values()}
    result.checks["resolution_is_10m"] = bool(resolutions) and all(
        abs(res[0] - DEFAULT_RESOLUTION_M) <= 1e-6 and abs(res[1] - DEFAULT_RESOLUTION_M) <= 1e-6
        for res in resolutions
    )

    shapes = {raster.shape for raster in observation.bands.values()}
    transforms = {tuple(raster.profile["transform"]) for raster in observation.bands.values()}
    result.checks["bands_share_grid"] = len(shapes) == 1 and len(transforms) == 1

    valid_fractions = [stats.get("valid_fraction") or 0.0 for stats in details["bands"].values()]
    result.checks["has_valid_pixels"] = bool(valid_fractions) and min(valid_fractions) > 0.0
    result.checks["mostly_valid_pixels"] = bool(valid_fractions) and min(valid_fractions) >= 0.99

    first = next(iter(observation.bands.values()))
    scl = aux_scl_checks(observation)
    if scl is not None:
        details["scl"] = scl["details"]
        result.checks.update(scl["checks"])

    details.update({
        "crs": first.crs,
        "resolution": list(first.resolution),
        "width": int(first.profile["width"]),
        "height": int(first.profile["height"]),
        "bounds": [float(v) for v in first.bounds],
        "transform": [float(v) for v in first.profile["transform"]],
        "nodata": first.profile["nodata"],
        "area_km2": round(first.resolution[0] * first.resolution[1]
                          * first.profile["width"] * first.profile["height"] / 1e6, 3),
    })
    result.details = details
    return result


def check_crs(before: Observation, after: Observation, target_crs: str | None = None) -> ValidationResult:
    """Verify both observations carry a usable metric CRS that can share one analysis CRS."""
    result = ValidationResult(label="crs")
    observations = [before, after]
    target = rasterio.crs.CRS.from_user_input(target_crs or before.bands[BAND_ORDER[0]].crs)

    present = True
    transformable = True
    identical = True
    units: dict[str, str | None] = {}
    for observation in observations:
        crs_value = observation.bands[BAND_ORDER[0]].crs
        present &= crs_value is not None
        if crs_value is None:
            transformable = False
            identical = False
            continue
        crs = rasterio.crs.CRS.from_user_input(crs_value)
        identical &= crs == target
        try:
            rasterio.warp.transform_bounds(crs, target, *observation.bands[BAND_ORDER[0]].bounds)
        except Exception:  # pragma: no cover - defensive
            transformable = False
        units[observation.label] = crs.linear_units

    result.checks["crs_present"] = present
    result.checks["crs_transformable_to_analysis_crs"] = transformable
    result.checks["crs_units_metric"] = bool(units) and all(
        unit is not None and unit.lower() in {"metre", "meter"} for unit in units.values()
    )
    result.details = {
        "analysis_crs": target.to_string(),
        "before_crs": before.bands[BAND_ORDER[0]].crs,
        "after_crs": after.bands[BAND_ORDER[0]].crs,
        "crs_identical": identical,
        "linear_units": units,
    }
    return result


def check_bounds(before: Observation, after: Observation, target_crs: str | None = None) -> ValidationResult:
    """Verify the two footprints overlap and quantify the shared area."""
    result = ValidationResult(label="bounds")
    target = rasterio.crs.CRS.from_user_input(target_crs or before.bands[BAND_ORDER[0]].crs)

    reprojected = {}
    for observation in (before, after):
        raster = observation.bands[BAND_ORDER[0]]
        reprojected[observation.label] = tuple(
            rasterio.warp.transform_bounds(raster.crs, target, *raster.bounds)
        )
    before_bounds = reprojected["before"]
    after_bounds = reprojected["after"]

    intersection = intersect_bounds(before_bounds, after_bounds)
    areas = {
        label: (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
        for label, bounds in reprojected.items()
    }
    if intersection is None:
        overlap_area = 0.0
    else:
        overlap_area = (intersection[2] - intersection[0]) * (intersection[3] - intersection[1])

    smallest = min(areas.values()) if areas else 0.0
    overlap_fraction = overlap_area / smallest if smallest > 0 else 0.0
    result.checks["bounds_overlap"] = intersection is not None and overlap_area > 0
    result.checks["overlap_covers_smaller_footprint"] = overlap_fraction >= 0.25
    result.details = {
        "analysis_crs": target.to_string(),
        "before_bounds": [float(v) for v in before_bounds],
        "after_bounds": [float(v) for v in after_bounds],
        "intersection_bounds": None if intersection is None else [float(v) for v in intersection],
        "overlap_fraction_of_smaller": round(overlap_fraction, 6),
        "overlap_area_km2": round(overlap_area / 1e6, 3),
    }
    return result


def read_bands(observation: Observation, masked: bool = True) -> dict[str, np.ndarray]:
    """Read every tensor band at native resolution as float32 (NaN at nodata)."""
    return {band: read_band(observation.bands[band].path, masked=masked) for band in BAND_ORDER}


def handle_nodata(array: np.ndarray, nodata: float | None = NODATA_VALUE) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(array_with_nan, valid_mask)`` for a single band.

    A pixel is invalid when it equals the nodata value, is non-finite, or is below
    zero (Sentinel-2 L2A surface reflectance is non-negative).
    """
    data = np.asarray(array, dtype="float32")
    invalid = ~np.isfinite(data) | (data < 0)
    if nodata is not None:
        invalid |= data == np.float32(nodata)
    cleaned = np.where(invalid, np.float32(np.nan), data).astype("float32")
    return cleaned, ~invalid


def normalize_bands(array: np.ndarray, scale: float = REFLECTANCE_SCALE,
                    value_range: tuple[float, float] = REFLECTANCE_RANGE) -> np.ndarray:
    """Convert digital numbers to surface reflectance and clip to ``value_range``.

    Fixed and reproducible: no per-image statistics, so before/after remain directly
    comparable. NaNs (invalid pixels) propagate untouched.
    """
    data = np.asarray(array, dtype="float32")
    with np.errstate(invalid="ignore"):
        reflectance = (data / np.float32(scale)).astype("float32")
    return np.clip(reflectance, value_range[0], value_range[1]).astype("float32")


@dataclass
class AlignedObservation:
    """One observation resampled onto the shared grid, in tensor-ready form."""

    label: str
    grid: RasterGrid
    data: np.ndarray
    """``[4, H, W]`` float32 reflectance; NaN marks invalid pixels."""

    band_valid: dict[str, np.ndarray]
    """Per-band ``[H, W]`` boolean validity masks."""

    @property
    def valid(self) -> np.ndarray:
        """``[H, W]`` pixels valid in every band: safe to use for change detection.

        Already includes the SCL cloud / invalid-pixel policy applied in :func:`align_rasters`.
        """
        mask = np.ones((self.grid.height, self.grid.width), dtype=bool)
        for band_mask in self.band_valid.values():
            mask &= band_mask
        return mask

    @property
    def metadata(self) -> dict:
        return {
            "label": self.label,
            "shape": [int(dimension) for dimension in self.data.shape],
            "dtype": self.data.dtype.name,
            "bands": list(BAND_ORDER),
            "valid_fraction": round(float(self.valid.mean()), 6),
            "band_stats": {band: describe(self.data[index]) for index, band in enumerate(BAND_ORDER)},
        }


def align_rasters(observation: Observation, grid: RasterGrid,
                  resampling: str = "bilinear", cloud: CloudMask | None = None,
                  valid_classes: tuple[int, ...] = SCL_VALID_CLASSES) -> AlignedObservation:
    """Resample every band of ``observation`` onto ``grid`` and normalize to reflectance.

    The same ``grid`` for before and after guarantees that ``data[:, row, col]``
    describes the same geographic location in both images, which is what makes a
    per-pixel comparison (and the Siamese U-Net input) meaningful.

    A pixel is kept only when it is valid in the band itself (nodata, non-finite, negative),
    covered by the source footprint, *and* classified as usable surface by SCL. Passing a
    pre-computed ``cloud`` mask reuses it instead of reading the SCL band again; when the
    observation has no SCL the mask is a no-op.
    """
    from rasterio.enums import Resampling

    cloud_mask = cloud if cloud is not None else mask_observation(observation, grid, valid_classes)
    if cloud_mask.valid.shape != (grid.height, grid.width):
        raise ValueError(f"cloud mask shape {cloud_mask.valid.shape} does not match grid "
                         f"{(grid.height, grid.width)}")

    method = Resampling[resampling]
    band_arrays: list[np.ndarray] = []
    band_valid: dict[str, np.ndarray] = {}

    for band in BAND_ORDER:
        raster = observation.bands[band]
        resampled = resample_to_grid(raster.path, grid, resampling=method)
        coverage = resample_mask_to_grid(raster.path, grid)
        cleaned, in_band_valid = handle_nodata(resampled, nodata=raster.profile["nodata"])
        mask = in_band_valid & coverage & cloud_mask.valid
        cleaned = np.where(mask, cleaned, np.float32(np.nan))
        band_arrays.append(normalize_bands(cleaned))
        band_valid[band] = mask

    return AlignedObservation(label=observation.label, grid=grid,
                              data=np.stack(band_arrays, axis=0).astype("float32"),
                              band_valid=band_valid)


def convert_to_tensor(array: np.ndarray, fill_value: float = NODATA_VALUE):
    """Convert an aligned ``[C, H, W]`` reflectance array into a float32 torch tensor.

    Invalid (NaN) pixels are filled with ``fill_value`` because the network cannot
    consume NaN; the authoritative validity is kept in the mask arrays alongside it.
    """
    import torch  # imported lazily: raster-only tooling must work without torch DLLs

    data = np.asarray(array, dtype="float32")
    filled = np.where(np.isfinite(data), data, np.float32(fill_value)).astype("float32")
    return torch.from_numpy(filled)


def write_stack(path: str | Path, array: np.ndarray, grid: RasterGrid,
                band_names: tuple[str, ...] = BAND_ORDER) -> None:
    """Write a ``[C, H, W]`` float32 stack; NaN becomes ``NODATA_VALUE`` on disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(array, dtype="float32")
    data = np.where(np.isfinite(data), data, np.float32(NODATA_VALUE))
    profile = grid.profile(count=data.shape[0], dtype="float32", nodata=NODATA_VALUE)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
        for index, name in enumerate(band_names, start=1):
            dst.set_band_description(index, f"{name} ({BAND_ROLES.get(name, 'band')})")


def file_record(path: str | Path) -> dict:
    """Size + SHA-256 of an output file, for provenance in ``metadata.json``."""
    path = Path(path)
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _observation_metadata(observation: Observation, aligned: AlignedObservation) -> dict:
    return {
        "label": observation.label,
        "acquisition_date": observation.date,
        "item_id": observation.item_id,
        "mgrs_tile": observation.mgrs_tile,
        "cloud_cover_percent": observation.cloud_cover,
        "aux_bands": {band: str(raster.path) for band, raster in observation.aux.items()},
        "raw_bands": {band: str(raster.path) for band, raster in observation.bands.items()},
        "valid_fraction": round(float(aligned.valid.mean()), 6),
        "band_stats_reflectance": {
            band: describe(aligned.data[index]) for index, band in enumerate(BAND_ORDER)
        },
    }


def prepare_pair(raw_dir: str | Path = DEFAULT_RAW_DIR, processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
                 resolution: float = DEFAULT_RESOLUTION_M, crs: str | None = None,
                 resampling: str = "bilinear", validate: bool = True,
                 build_tensors: bool = True,
                 valid_classes: tuple[int, ...] = SCL_VALID_CLASSES) -> dict:
    """Turn a raw Sentinel-2 band pair into the frozen input contract for the AI stage.

    Pipeline: validate -> common grid -> SCL cloud / invalid masking -> resample -> normalize
    to reflectance -> NDVI. Writes ``before.tif`` / ``after.tif`` (4-band reflectance), the
    ``*.npy`` arrays, validity masks (band + cloud), the SCL rasters, the NDVI rasters and a
    ``metadata.json`` provenance record, and returns that metadata.
    """
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    before, after = load_sentinel_pair(raw_dir)

    reports = {
        "before": validate_raster(before),
        "after": validate_raster(after),
        "crs": check_crs(before, after, crs),
        "bounds": check_bounds(before, after, crs),
    }
    if validate:
        failed = {name: report.problems for name, report in reports.items() if not report.passed}
        if failed:
            raise ValueError(f"raw Sentinel-2 pair failed validation: {failed}")

    grid = common_grid([*before.paths, *after.paths], resolution=resolution, crs=crs)
    cloud_before = mask_observation(before, grid, valid_classes)
    cloud_after = mask_observation(after, grid, valid_classes)
    aligned_before = align_rasters(before, grid, resampling=resampling, cloud=cloud_before)
    aligned_after = align_rasters(after, grid, resampling=resampling, cloud=cloud_after)
    shared_valid = aligned_before.valid & aligned_after.valid
    cloud_valid = pair_valid_mask(cloud_before, cloud_after)
    cloud_masked_pixels = int((~cloud_valid).sum())

    channel = {band: index for index, band in enumerate(BAND_ORDER)}
    ndvi_before = calculate_ndvi(aligned_before.data[channel[NIR_BAND]],
                                 aligned_before.data[channel[RED_BAND]], aligned_before.valid)
    ndvi_after = calculate_ndvi(aligned_after.data[channel[NIR_BAND]],
                                aligned_after.data[channel[RED_BAND]], aligned_after.valid)
    ndvi_delta = ndvi_difference(ndvi_before, ndvi_after)

    outputs: dict[str, Path] = {
        "before_tif": processed_dir / "before.tif",
        "after_tif": processed_dir / "after.tif",
        "before_npy": processed_dir / "before.npy",
        "after_npy": processed_dir / "after.npy",
        "before_valid_mask_npy": processed_dir / "before_valid_mask.npy",
        "after_valid_mask_npy": processed_dir / "after_valid_mask.npy",
        "valid_mask_npy": processed_dir / "valid_mask.npy",
        "cloud_valid_mask_npy": processed_dir / "cloud_valid_mask.npy",
        "scl_before_tif": processed_dir / "scl_before.tif",
        "scl_after_tif": processed_dir / "scl_after.tif",
        "ndvi_before_tif": processed_dir / "ndvi_before.tif",
        "ndvi_after_tif": processed_dir / "ndvi_after.tif",
        "ndvi_difference_tif": processed_dir / "ndvi_difference.tif",
    }

    write_stack(outputs["before_tif"], aligned_before.data, grid)
    write_stack(outputs["after_tif"], aligned_after.data, grid)
    for key, array in (("before_npy", aligned_before.data), ("after_npy", aligned_after.data)):
        np.save(outputs[key], np.where(np.isfinite(array), array, np.float32(NODATA_VALUE)).astype("float32"))
    np.save(outputs["before_valid_mask_npy"], aligned_before.valid)
    np.save(outputs["after_valid_mask_npy"], aligned_after.valid)
    np.save(outputs["valid_mask_npy"], shared_valid)
    np.save(outputs["cloud_valid_mask_npy"], cloud_valid)
    write_scl_raster(outputs["scl_before_tif"], cloud_before.class_array, grid,
                     f"SCL {before.date or 'before'}")
    write_scl_raster(outputs["scl_after_tif"], cloud_after.class_array, grid,
                     f"SCL {after.date or 'after'}")
    write_ndvi_raster(outputs["ndvi_before_tif"], ndvi_before, grid, f"NDVI {before.date or 'before'}")
    write_ndvi_raster(outputs["ndvi_after_tif"], ndvi_after, grid, f"NDVI {after.date or 'after'}")
    write_ndvi_raster(outputs["ndvi_difference_tif"], ndvi_delta, grid,
                      f"NDVI change {after.date or 'after'} - {before.date or 'before'}")

    tensor_info: dict = {"shape": [int(v) for v in aligned_before.data.shape], "dtype": "float32",
                         "layout": "C,H,W", "band_order": list(BAND_ORDER)}
    if build_tensors:
        before_tensor = convert_to_tensor(aligned_before.data)
        after_tensor = convert_to_tensor(aligned_after.data)
        tensor_info.update({
            "before_shape": [int(v) for v in before_tensor.shape],
            "after_shape": [int(v) for v in after_tensor.shape],
            "torch_dtype": str(before_tensor.dtype),
            "min": round(float(before_tensor.min()), 6),
            "max": round(float(before_tensor.max()), 6),
            "mean_before": round(float(before_tensor.mean()), 6),
            "mean_after": round(float(after_tensor.mean()), 6),
            "tensor_nan_free": bool(np.isfinite(before_tensor.numpy()).all()
                                    and np.isfinite(after_tensor.numpy()).all()),
        })

    metadata: dict = {
        "artifact": "terraguard.ai/processed_sentinel_pair",
        "aoi": raw_dir.name,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pipeline": {
            "module": "backend/app/services/preprocessing.py::prepare_pair",
            "resampling": resampling,
            "grid_source": "intersection of both footprints, snapped to resolution",
        },
        "crs": grid.crs,
        "resolution_m": float(grid.resolution_m),
        "width": int(grid.width),
        "height": int(grid.height),
        "bands": list(BAND_ORDER),
        "band_roles": BAND_ROLES,
        "grid": grid.to_dict(),
        "normalization": {
            "description": NORMALIZATION_DESCRIPTION,
            "scale": REFLECTANCE_SCALE,
            "clip": list(REFLECTANCE_RANGE),
        },
        "nodata_policy": {
            "disk_nodata_value": NODATA_VALUE,
            "note": "invalid pixels are stored as 0 in .tif/.npy; the *_valid_mask.npy "
                    "arrays are authoritative",
            "shared_valid_fraction": round(float(shared_valid.mean()), 6),
        },
        "cloud_mask": {
            "module": "backend/app/services/cloud_mask.py::mask_observation",
            "policy": "class 4 VEGETATION, 5 NOT_VEGETATED, 6 WATER are usable surface; "
                      "0 NO_DATA, 1 SATURATED, 2 DARK_AREA_OR_SHADOW, 3 CLOUD_SHADOW, "
                      "7 UNCLASSIFIED, 8/9 CLOUD, 10 THIN_CIRRUS, 11 SNOW_OR_ICE are masked",
            "valid_classes": list(valid_classes),
            "masked_classes": list(SCL_MASKED_CLASSES),
            "scl_classes": {str(code): name for code, name in SCL_CLASSES.items()},
            "applied": bool(cloud_before.available or cloud_after.available),
            "shared_valid_fraction": round(float(cloud_valid.mean()), 6),
            "masked_pixels": cloud_masked_pixels,
            "masked_fraction": round(cloud_masked_pixels / cloud_valid.size, 6),
            "before": cloud_before.to_dict(),
            "after": cloud_after.to_dict(),
            "note": "the '_valid_mask' arrays and the tensors already exclude these pixels; "
                    "masked pixels are NaN in memory and nodata on disk",
        },
        "observations": {
            "before": _observation_metadata(before, aligned_before),
            "after": _observation_metadata(after, aligned_after),
        },
        "ndvi": {
            "formula": NDVI_FORMULA,
            "before": describe(ndvi_before),
            "after": describe(ndvi_after),
            "difference": describe(ndvi_delta),
            "change_areas": change_area_fractions(ndvi_delta, CHANGE_THRESHOLDS),
        },
        "tensor": tensor_info,
        "validation": {name: report.to_dict() for name, report in reports.items()},
    }

    metadata["outputs"] = {name: {**file_record(path), "file": path.name} for name, path in outputs.items()}
    metadata_path = processed_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TerraGuard AI - Sentinel-2 pair preprocessing")
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--processed", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--resolution", type=float, default=DEFAULT_RESOLUTION_M)
    parser.add_argument("--crs", default=None, help="analysis CRS (default: CRS of the raw rasters)")
    parser.add_argument("--resampling", default="bilinear",
                        choices=["nearest", "bilinear", "cubic", "cubic_spline", "lanczos", "average"])
    parser.add_argument("--no-validate", action="store_true", help="skip hard validation failures")
    parser.add_argument("--lenient-scl", action="store_true",
                        help="also treat SCL class 7 (UNCLASSIFIED) as usable surface")
    args = parser.parse_args(argv)

    metadata = prepare_pair(raw_dir=args.raw, processed_dir=args.processed, resolution=args.resolution,
                            crs=args.crs, resampling=args.resampling, validate=not args.no_validate,
                            valid_classes=(SCL_VALID_CLASSES_LENIENT if args.lenient_scl
                                           else SCL_VALID_CLASSES))

    cloud = metadata["cloud_mask"]
    print(f"Processed pair written to {args.processed}")
    print(f"  crs          : {metadata['crs']}")
    print(f"  grid         : {metadata['width']}x{metadata['height']} @ {metadata['resolution_m']} m")
    print(f"  bands        : {', '.join(metadata['bands'])}")
    print(f"  before       : {metadata['observations']['before']['acquisition_date']} "
          f"({metadata['observations']['before']['item_id']})")
    print(f"  after        : {metadata['observations']['after']['acquisition_date']} "
          f"({metadata['observations']['after']['item_id']})")
    print(f"  cloud mask   : {'applied' if cloud['applied'] else 'no SCL asset'} | "
          f"valid classes {cloud['valid_classes']} | "
          f"{cloud['masked_pixels']} px masked ({cloud['masked_fraction']:.4%})")
    print(f"  valid pixels : {metadata['nodata_policy']['shared_valid_fraction']:.4%} "
          f"(band + footprint + cloud)")
    print(f"  NDVI delta   : mean {metadata['ndvi']['difference']['mean']}, "
          f"browning>0.2 {metadata['ndvi']['change_areas']['loss_at_0.2_fraction']:.2%}, "
          f"greening>0.2 {metadata['ndvi']['change_areas']['gain_at_0.2_fraction']:.2%}")
    print(f"  tensor       : {metadata['tensor']['before_shape']} {metadata['tensor']['torch_dtype']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
