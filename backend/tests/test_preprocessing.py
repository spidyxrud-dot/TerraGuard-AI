"""Tests for the Sentinel-2 preprocessing contract.

The synthetic scenes below are built so that the *same geographic* feature is written
into rasters with different origins (and, in one test, a different CRS). If the
alignment step ever regressed, the before/after feature would land in different
pixels and the change-detection assertions would fail.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from app.services.ndvi import calculate_ndvi, change_area_fractions, ndvi_difference
from app.services.preprocessing import (
    BAND_ORDER,
    align_rasters,
    convert_to_tensor,
    handle_nodata,
    load_sentinel_pair,
    normalize_bands,
    prepare_pair,
    validate_raster,
)
from app.utils.geo import common_grid, reproject_bounds

UTM_CRS = "EPSG:32643"
GEO_CRS = "EPSG:4326"
RESOLUTION_M = 10.0
SIZE = 64
ORIGIN = (500_000.0, 2_000_000.0)
SHIFT = (50.0, -30.0)

BAND_BASE = {"B02": 300, "B03": 400, "B04": 500, "B08": 3000}

# Geographic bounds (UTM) of the synthetic "vegetation loss" square.
LOSS_BOUNDS = (500_200.0, 1_999_680.0, 500_320.0, 1_999_800.0)


def write_band(path: Path, array: np.ndarray, transform, crs: str = UTM_CRS, nodata: int = 0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=array.shape[0], width=array.shape[1],
                       count=1, dtype=array.dtype.name, crs=crs, transform=transform, nodata=nodata) as dst:
        dst.write(array, 1)
    return path


def local_window(transform, bounds) -> tuple[int, int, int, int]:
    """Pixel window ``(row0, row1, col0, col1)`` of geographic ``bounds`` in a raster."""
    col0 = int(round((bounds[0] - transform.c) / abs(transform.a)))
    col1 = int(round((bounds[2] - transform.c) / abs(transform.a)))
    row0 = int(round((transform.f - bounds[3]) / abs(transform.e)))
    row1 = int(round((transform.f - bounds[1]) / abs(transform.e)))
    return row0, row1, col0, col1


def build_scene(directory: Path, transform, *, browning: bool = False, crs: str = UTM_CRS,
                size: int = SIZE) -> None:
    """Write a 4-band synthetic scene; with ``browning`` the square loses NIR and gains red.

    Every band carries a mild ramp so the validation checks see real variation, and the
    loss square is placed by *geographic* bounds so a shifted (or reprojected) raster
    still contains the same feature in the same place on the ground.
    """
    window = local_window(transform, reproject_bounds(LOSS_BOUNDS, UTM_CRS, crs))
    rows, cols = np.indices((size, size))
    texture = (rows + cols).astype("uint16")
    for band, base in BAND_BASE.items():
        data = (base + texture).astype("uint16")
        if browning:
            row0, row1, col0, col1 = window
            if band == "B08":
                data[row0:row1, col0:col1] = (800 + texture[row0:row1, col0:col1]).astype("uint16")
            elif band == "B04":
                data[row0:row1, col0:col1] = (1200 + texture[row0:row1, col0:col1]).astype("uint16")
        write_band(directory / f"{band}.tif", data, transform, crs=crs)


def utm_transform(offset: tuple[float, float] = (0.0, 0.0)):
    return from_origin(ORIGIN[0] + offset[0], ORIGIN[1] + offset[1], RESOLUTION_M, RESOLUTION_M)


def scene_transform(crs: str, offset: tuple[float, float] = (0.0, 0.0)):
    """Transform for a ``SIZE``-pixel scene covering the same ground area in any CRS."""
    if crs == UTM_CRS:
        return utm_transform(offset)
    min_x = ORIGIN[0] + offset[0]
    max_y = ORIGIN[1] + offset[1]
    bounds = reproject_bounds((min_x, max_y - SIZE * RESOLUTION_M, min_x + SIZE * RESOLUTION_M, max_y),
                              UTM_CRS, crs)
    x_resolution = (bounds[2] - bounds[0]) / SIZE
    y_resolution = (bounds[3] - bounds[1]) / SIZE
    return from_origin(bounds[0], bounds[3], x_resolution, y_resolution)


def build_raw_pair(raw_dir: Path, *, shifted: bool = False, after_crs: str = UTM_CRS) -> None:
    build_scene(raw_dir / "before", scene_transform(UTM_CRS))
    offset = SHIFT if shifted else (0.0, 0.0)
    build_scene(raw_dir / "after", scene_transform(after_crs, offset), browning=True, crs=after_crs)


def change_metrics(difference: np.ndarray, grid, threshold: float = 0.2) -> tuple[float, float, float]:
    """Return (loss fraction inside the true feature, loss fraction outside, max |change| outside).

    The inside metric uses the *exact* feature window so it is not diluted by padding;
    the outside metric excludes a 1-pixel halo, which bilinear resampling legitimately mixes.
    """
    row0, row1, col0, col1 = local_window(grid.transform, LOSS_BOUNDS)
    inner = difference[row0:row1, col0:col1]
    outer = difference.copy()
    outer[max(row0 - 1, 0):row1 + 1, max(col0 - 1, 0):col1 + 1] = np.nan

    def loss_fraction(values: np.ndarray) -> float:
        finite = np.isfinite(values)
        if not finite.any():
            return 0.0
        return float((values[finite] < -threshold).mean())

    with np.errstate(invalid="ignore"):
        max_outside = float(np.nanmax(np.abs(outer)))
    return loss_fraction(inner), loss_fraction(outer), max_outside


def ndvi_difference_of(aligned_before, aligned_after) -> np.ndarray:
    before = calculate_ndvi(aligned_before.data[3], aligned_before.data[2], aligned_before.valid)
    after = calculate_ndvi(aligned_after.data[3], aligned_after.data[2], aligned_after.valid)
    return ndvi_difference(before, after)


@pytest.fixture()
def raw_pair(tmp_path: Path) -> Path:
    raw_dir = tmp_path / "raw" / "demo_area"
    build_raw_pair(raw_dir)
    return raw_dir


def test_load_pairs_and_validate_raster(raw_pair: Path) -> None:
    before, after = load_sentinel_pair(raw_pair)
    assert set(before.bands) == set(BAND_ORDER)
    assert before.date is None and after.date is None  # no acquisition.json in synthetic data

    for observation in (before, after):
        report = validate_raster(observation)
        assert report.passed, report.problems
        assert report.details["crs"] == UTM_CRS
        assert report.details["resolution"] == [RESOLUTION_M, RESOLUTION_M]


def test_load_raises_when_a_band_is_missing(raw_pair: Path) -> None:
    (raw_pair / "after" / "B08.tif").unlink()
    with pytest.raises(FileNotFoundError, match="B08"):
        load_sentinel_pair(raw_pair)


def test_common_grid_snaps_to_intersection(raw_pair: Path) -> None:
    before, after = load_sentinel_pair(raw_pair)
    grid = common_grid([*before.paths, *after.paths], resolution=RESOLUTION_M)
    assert grid.crs == UTM_CRS
    assert (grid.width, grid.height) == (SIZE, SIZE)
    assert grid.bounds == (ORIGIN[0], ORIGIN[1] - SIZE * RESOLUTION_M, ORIGIN[0] + SIZE * RESOLUTION_M, ORIGIN[1])


def test_common_grid_handles_different_origins(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw" / "demo_area"
    build_raw_pair(raw_dir, shifted=True)
    before, after = load_sentinel_pair(raw_dir)
    grid = common_grid([*before.paths, *after.paths], resolution=RESOLUTION_M)
    # before spans 500000..500640 / 1999360..2000000; after is shifted by +50 m, -30 m
    assert (grid.width, grid.height) == (59, 61)
    assert grid.bounds[0] == pytest.approx(500_050.0)
    assert grid.bounds[3] == pytest.approx(1_999_970.0)


def test_align_rasters_co_registers_a_shifted_pair(tmp_path: Path) -> None:
    """A +50 m / -30 m scene offset must not masquerade as environmental change."""
    raw_dir = tmp_path / "raw" / "demo_area"
    build_raw_pair(raw_dir, shifted=True)
    before, after = load_sentinel_pair(raw_dir)
    grid = common_grid([*before.paths, *after.paths], resolution=RESOLUTION_M)

    aligned_before = align_rasters(before, grid)
    aligned_after = align_rasters(after, grid)
    assert aligned_before.data.shape == aligned_after.data.shape == (4, grid.height, grid.width)
    assert aligned_before.valid.mean() > 0.99 and aligned_after.valid.mean() > 0.99

    difference = ndvi_difference_of(aligned_before, aligned_after)
    inside, outside, max_outside = change_metrics(difference, grid)

    assert inside > 0.95, f"loss square not co-registered (inside fraction {inside:.2f})"
    assert outside < 0.01, f"spurious change outside the square ({outside:.2f})"
    assert max_outside < 0.05, f"misalignment leaked {max_outside:.3f} NDVI outside the square"


def test_align_rasters_reprojects_a_different_crs(tmp_path: Path) -> None:
    """An after scene delivered in WGS84 must land on the same pixel as before."""
    raw_dir = tmp_path / "raw" / "demo_area"
    build_raw_pair(raw_dir, shifted=True, after_crs=GEO_CRS)
    before, after = load_sentinel_pair(raw_dir)
    assert after.bands["B08"].crs == GEO_CRS

    grid = common_grid([*before.paths, *after.paths], resolution=RESOLUTION_M)
    aligned_before = align_rasters(before, grid)
    aligned_after = align_rasters(after, grid)
    difference = ndvi_difference_of(aligned_before, aligned_after)
    inside, outside, max_outside = change_metrics(difference, grid)

    assert inside > 0.8, f"reprojected loss square not co-located (inside fraction {inside:.2f})"
    assert outside < 0.05, f"spurious change after reprojection ({outside:.2f})"
    assert max_outside < 0.15, f"reprojection leaked {max_outside:.3f} NDVI outside the square"


def test_align_rasters_outputs_reflectance_in_unit_range(raw_pair: Path) -> None:
    before, after = load_sentinel_pair(raw_pair)
    grid = common_grid([*before.paths, *after.paths], resolution=RESOLUTION_M)
    aligned = align_rasters(before, grid)
    blue = aligned.data[0]
    finite = np.isfinite(blue)
    assert aligned.data.dtype == np.float32
    assert float(aligned.data[np.isfinite(aligned.data)].min()) >= 0.0
    assert float(aligned.data[np.isfinite(aligned.data)].max()) <= 1.0
    # B02 base 300 + pixel ramp (<=126) -> about 0.030..0.043 reflectance
    assert 0.02 < float(blue[finite].mean()) < 0.05


def test_normalize_bands_scales_and_clips() -> None:
    digital_numbers = np.array([[0.0, 5000.0], [10000.0, 20000.0]], dtype="float32")
    reflectance = normalize_bands(digital_numbers)
    assert reflectance.tolist() == [[0.0, 0.5], [1.0, 1.0]]
    assert reflectance.dtype == np.float32

    with_nan = normalize_bands(np.array([[np.nan, 2500.0]], dtype="float32"))
    assert np.isnan(with_nan[0, 0]) and with_nan[0, 1] == pytest.approx(0.25)


def test_handle_nodata_flags_invalid_pixels() -> None:
    data = np.array([[0.0, 100.0], [np.nan, -5.0]], dtype="float32")
    cleaned, valid = handle_nodata(data)
    assert valid.tolist() == [[False, True], [False, False]]
    assert np.isnan(cleaned[0, 0]) and np.isnan(cleaned[1, 0]) and np.isnan(cleaned[1, 1])
    assert cleaned[0, 1] == 100.0


def test_calculate_ndvi_values_and_invalids() -> None:
    nir = np.array([[3000.0, 500.0], [0.0, np.nan]], dtype="float32")
    red = np.array([[1000.0, 500.0], [0.0, 100.0]], dtype="float32")
    ndvi = calculate_ndvi(nir, red)
    assert ndvi[0, 0] == pytest.approx(0.5)
    assert ndvi[0, 1] == pytest.approx(0.0)
    assert np.isnan(ndvi[1, 0])  # zero reflectance sum cannot be divided
    assert np.isnan(ndvi[1, 1])  # NaN propagates

    masked = calculate_ndvi(nir, red, valid_mask=np.array([[False, True], [True, True]]))
    assert np.isnan(masked[0, 0]) and masked[0, 1] == pytest.approx(0.0)

    with pytest.raises(ValueError):
        calculate_ndvi(nir, red[:1])


def test_ndvi_difference_and_change_area_fractions() -> None:
    before = np.array([[0.8, 0.5, 0.2, np.nan]], dtype="float32")
    after = np.array([[0.5, 0.45, 0.55, 0.9]], dtype="float32")
    difference = ndvi_difference(before, after)
    assert difference[0, 0] == pytest.approx(-0.3)
    assert difference[0, 1] == pytest.approx(-0.05)
    assert difference[0, 2] == pytest.approx(0.35)
    assert np.isnan(difference[0, 3])

    report = change_area_fractions(difference, thresholds=(0.1, 0.3))
    assert report["valid_pixels"] == 3
    assert report["loss_at_0.1_pixels"] == 1          # -0.30 counts as >= 0.1 loss
    assert report["loss_at_0.3_pixels"] == 1          # inclusive threshold
    assert report["gain_at_0.1_pixels"] == 1          # +0.35 gain
    assert report["gain_at_0.3_pixels"] == 1
    assert report["loss_at_0.1_fraction"] == pytest.approx(1 / 3)
    assert report["gain_at_0.3_fraction"] == pytest.approx(1 / 3)


def test_convert_to_tensor_shape_dtype_and_nan_fill(raw_pair: Path) -> None:
    before, _ = load_sentinel_pair(raw_pair)
    grid = common_grid(before.paths, resolution=RESOLUTION_M)
    aligned = align_rasters(before, grid)
    aligned.data[:, 0, 0] = np.nan
    tensor = convert_to_tensor(aligned.data)
    assert tuple(tensor.shape) == (4, grid.height, grid.width)
    assert str(tensor.dtype) == "torch.float32"
    assert bool(np.isfinite(tensor.numpy()).all())
    assert float(tensor[:, 0, 0].sum()) == 0.0


def test_prepare_pair_writes_the_input_contract(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw" / "demo_area"
    processed_dir = tmp_path / "processed" / "demo_area"
    build_raw_pair(raw_dir, shifted=True)

    metadata = prepare_pair(raw_dir=raw_dir, processed_dir=processed_dir)

    for name in ("before.tif", "after.tif", "before.npy", "after.npy",
                 "before_valid_mask.npy", "after_valid_mask.npy", "valid_mask.npy",
                 "ndvi_before.tif", "ndvi_after.tif", "ndvi_difference.tif", "metadata.json"):
        assert (processed_dir / name).is_file(), f"missing processed artifact {name}"

    assert metadata["crs"] == UTM_CRS
    assert metadata["bands"] == list(BAND_ORDER)
    assert metadata["resolution_m"] == RESOLUTION_M
    assert metadata["tensor"]["before_shape"] == [4, metadata["height"], metadata["width"]]
    assert metadata["tensor"]["tensor_nan_free"] is True
    assert metadata["normalization"]["scale"] == 10_000.0

    with rasterio.open(processed_dir / "before.tif") as src:
        assert src.count == 4 and src.dtypes[0] == "float32"
        assert (src.height, src.width) == (metadata["height"], metadata["width"])
        assert src.crs.to_string() == UTM_CRS

    mask = np.load(processed_dir / "valid_mask.npy")
    assert mask.shape == (metadata["height"], metadata["width"])
    assert mask.any() and np.array_equal(mask, np.load(processed_dir / "before_valid_mask.npy")
                                         & np.load(processed_dir / "after_valid_mask.npy"))

    arrays = np.load(processed_dir / "before.npy")
    assert arrays.shape == (4, metadata["height"], metadata["width"])
    assert np.isfinite(arrays).all()

    # the synthetic square browned: a small area fraction but a strong per-pixel signal
    change_areas = metadata["ndvi"]["change_areas"]
    assert change_areas["loss_at_0.3_fraction"] > 0.02
    assert change_areas["gain_at_0.3_fraction"] < 0.001
    assert metadata["ndvi"]["difference"]["min"] < -0.5