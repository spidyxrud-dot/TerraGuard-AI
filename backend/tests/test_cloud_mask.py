"""Tests for Sentinel-2 SCL cloud / invalid-pixel masking.

The synthetic scenes below deliberately *fabricate* change that a change detector would
otherwise happily report: a bright "cloud" that turns one block into sudden vegetation,
and a dark "shadow" that does the same. Both blocks are declared in the SCL band, so the
correct outcome is that they never reach the tensors at all.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from app.services.cloud_mask import (
    MASK_REASONS,
    SCL_BAND,
    SCL_CLASSES,
    SCL_MASKED_CLASSES,
    SCL_VALID_CLASSES,
    SCL_VALID_CLASSES_LENIENT,
    apply_mask,
    class_histogram,
    class_name,
    load_scl,
    mask_observation,
    pair_valid_mask,
    resample_scl_to_grid,
    scl_mask_report,
    scl_valid_mask,
    write_scl_raster,
)
from app.services.cloud_mask import main as cloud_main
from app.services.ndvi import calculate_ndvi, ndvi_difference
from app.services.preprocessing import (
    BAND_ORDER,
    DEFAULT_RESOLUTION_M,
    align_rasters,
    convert_to_tensor,
    load_sentinel_pair,
    prepare_pair,
    validate_raster,
)
from app.utils.geo import common_grid

UTM_CRS = "EPSG:32643"
SIZE = 64
ORIGIN = (500_000.0, 2_000_000.0)
SCL_RESOLUTION_M = 20.0

BAND_BASE = {"B02": 300, "B03": 400, "B04": 500, "B08": 3000}

# All three blocks sit on the 20 m SCL pixel lattice, so the coarse SCL band and the 10 m
# bands agree exactly about where their edges are (no rounding slack in the assertions).
LOSS_BOUNDS = (500_160.0, 1_999_840.0, 500_320.0, 2_000_000.0)
CLOUD_BOUNDS = (500_340.0, 1_999_660.0, 500_500.0, 1_999_820.0)
SHADOW_BOUNDS = (500_000.0, 1_999_500.0, 500_160.0, 1_999_660.0)
BLOCK_PIXELS = 16 * 16
NATIVE_BLOCK_PIXELS = 8 * 8
"""Block size in native 20 m SCL pixels - the count ``scl_mask_report`` sees."""


def transform_of(origin: tuple[float, float] = ORIGIN, resolution: float = DEFAULT_RESOLUTION_M):
    return from_origin(origin[0], origin[1], resolution, resolution)


def coarse_transform():
    """The 20 m transform Sentinel-2 SCL actually arrives on."""
    return from_origin(ORIGIN[0], ORIGIN[1], SCL_RESOLUTION_M, SCL_RESOLUTION_M)


def local_window(transform, bounds) -> tuple[int, int, int, int]:
    """Pixel window ``(row0, row1, col0, col1)`` of geographic ``bounds`` in a raster."""
    col0 = int(round((bounds[0] - transform.c) / abs(transform.a)))
    col1 = int(round((bounds[2] - transform.c) / abs(transform.a)))
    row0 = int(round((transform.f - bounds[3]) / abs(transform.e)))
    row1 = int(round((transform.f - bounds[1]) / abs(transform.e)))
    return row0, row1, col0, col1


def block(array: np.ndarray, transform, bounds) -> tuple[slice, slice]:
    """Numpy view of the geographic ``bounds`` inside ``array``."""
    row0, row1, col0, col1 = local_window(transform, bounds)
    return (slice(row0, row1), slice(col0, col1))


def write_raster(path: Path, array: np.ndarray, transform, crs: str = UTM_CRS,
                 nodata: int | None = 0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=array.shape[0], width=array.shape[1],
                       count=1, dtype=array.dtype.name, crs=crs, transform=transform,
                       nodata=nodata) as dst:
        dst.write(array, 1)
    return path


def write_scl(path: Path, *, cloud: bool = False, shadow: bool = False) -> Path:
    """SCL at Sentinel-2 native 20 m: class 4 VEGETATION plus optional flagged blocks."""
    classes = np.full((SIZE // 2, SIZE // 2), 4, dtype="uint8")
    transform = coarse_transform()
    if cloud:
        classes[block(classes, transform, CLOUD_BOUNDS)] = 9
    if shadow:
        classes[block(classes, transform, SHADOW_BOUNDS)] = 3
    return write_raster(path, classes, transform)


def build_scene(directory: Path, *, browning: bool = False, cloud: bool = False,
                shadow: bool = False) -> None:
    """Write one 4-band scene plus its SCL band.

    ``browning`` applies the honest NIR-loss / red-gain inside :data:`LOSS_BOUNDS`.
    ``cloud`` fabricates a greening spike inside :data:`CLOUD_BOUNDS` and marks it SCL 9.
    ``shadow`` fabricates a greening spike inside :data:`SHADOW_BOUNDS` and marks it SCL 3.
    """
    rows, cols = np.indices((SIZE, SIZE))
    texture = (rows + cols).astype("uint16")
    transform = transform_of()
    loss = block(texture, transform, LOSS_BOUNDS)
    cloud_view = block(texture, transform, CLOUD_BOUNDS)
    shadow_view = block(texture, transform, SHADOW_BOUNDS)

    for band, base in BAND_BASE.items():
        data = (base + texture).astype("uint16")
        if browning:
            if band == "B08":
                data[loss] = (800 + texture[loss]).astype("uint16")
            elif band == "B04":
                data[loss] = (1200 + texture[loss]).astype("uint16")
        if cloud:
            data[cloud_view] = 9000 if band == "B08" else 200 if band == "B04" else base
        if shadow:
            data[shadow_view] = 800 if band == "B08" else 1400 if band == "B04" else base
        write_raster(directory / f"{band}.tif", data, transform)

    write_scl(directory / f"{SCL_BAND}.tif", cloud=cloud, shadow=shadow)


def build_raw_pair(raw_dir: Path, *, cloud: bool = True, shadow: bool = True,
                   scl: bool = True) -> None:
    """A pair whose only genuine change is the browning square in :data:`LOSS_BOUNDS`."""
    build_scene(raw_dir / "before", shadow=shadow)
    build_scene(raw_dir / "after", browning=True, cloud=cloud)
    if not scl:
        for label in ("before", "after"):
            (raw_dir / label / f"{SCL_BAND}.tif").unlink()


@pytest.fixture()
def raw_pair(tmp_path: Path) -> Path:
    raw_dir = tmp_path / "raw" / "demo_area"
    build_raw_pair(raw_dir)
    return raw_dir


def grid_of(raw_dir: Path, resolution: float = DEFAULT_RESOLUTION_M):
    before, after = load_sentinel_pair(raw_dir)
    return before, after, common_grid([*before.paths, *after.paths], resolution=resolution)


def ndvi_of(aligned) -> np.ndarray:
    return calculate_ndvi(aligned.data[3], aligned.data[2], aligned.valid)


# --------------------------------------------------------------------------- taxonomy


def test_taxonomy_covers_every_sentinel2_scl_class() -> None:
    assert sorted(SCL_CLASSES) == list(range(12))
    assert set(SCL_CLASSES) - set(SCL_VALID_CLASSES) == set(SCL_MASKED_CLASSES)
    assert class_name(4) == "vegetation"
    assert class_name(10) == "thin_cirrus"
    assert class_name(99) == "unknown_class_99"


def test_mask_reasons_cover_every_masked_class() -> None:
    grouped = {code for codes in MASK_REASONS.values() for code in codes}
    assert grouped == set(SCL_MASKED_CLASSES)
    assert MASK_REASONS["cloud"] == (8, 9)
    assert MASK_REASONS["shadow"] == (2, 3)
    assert MASK_REASONS["cirrus"] == (10,)


def test_lenient_policy_only_adds_unclassified() -> None:
    assert set(SCL_VALID_CLASSES_LENIENT) - set(SCL_VALID_CLASSES) == {7}


# ---------------------------------------------------------------------- classification


def test_scl_valid_mask_keeps_only_usable_surface_classes() -> None:
    values = np.array([[4, 5, 6], [7, 9, 0], [3, 10, 11]], dtype="uint8")
    expected = np.array([[True, True, True], [False, False, False], [False, False, False]])
    assert np.array_equal(scl_valid_mask(values), expected)


def test_scl_valid_mask_rejects_unknown_and_non_finite_codes() -> None:
    values = np.array([[4.0, 42.0], [np.nan, np.inf]])
    assert np.array_equal(scl_valid_mask(values), np.array([[True, False], [False, False]]))


def test_scl_valid_mask_is_strict_about_fractional_codes() -> None:
    assert np.array_equal(scl_valid_mask(np.array([4.0, 4.4, 5.6, 6.0])),
                          np.array([True, False, False, True]))


def test_lenient_policy_trusts_unclassified() -> None:
    values = np.array([7, 8])
    assert not scl_valid_mask(values).any()
    assert np.array_equal(scl_valid_mask(values, SCL_VALID_CLASSES_LENIENT),
                          np.array([True, False]))


def test_scl_valid_mask_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        scl_valid_mask(np.array([], dtype="uint8"))


# ---------------------------------------------------------------------------- resample


def test_load_scl_reads_native_classes_unchanged(tmp_path: Path) -> None:
    classes = np.array([[4, 5], [6, 9]], dtype="uint8")
    path = write_raster(tmp_path / "SCL.tif", classes, transform_of(resolution=10.0))
    assert np.array_equal(load_scl(path), classes)


def test_load_scl_treats_raster_nodata_as_no_data(tmp_path: Path) -> None:
    path = write_raster(tmp_path / "SCL.tif", np.array([[0, 4]], dtype="uint8"), transform_of())
    assert load_scl(path)[0, 0] == 0
    assert not scl_valid_mask(load_scl(path))[0, 0]


def test_resample_upsamples_20m_scl_without_inventing_classes(tmp_path: Path) -> None:
    """SCL is 20 m, the analysis grid is 10 m: nearest neighbour must keep codes exact."""
    classes = np.full((SIZE // 2, SIZE // 2), 4, dtype="uint8")
    classes[block(classes, coarse_transform(), CLOUD_BOUNDS)] = 9
    path = write_raster(tmp_path / "SCL.tif", classes, coarse_transform())
    grid = common_grid([write_raster(tmp_path / "band.tif",
                                     np.full((SIZE, SIZE), 500, dtype="uint16"), transform_of())],
                       resolution=DEFAULT_RESOLUTION_M)

    resampled = resample_scl_to_grid(path, grid)
    assert resampled.shape == (SIZE, SIZE)
    assert resampled.dtype == np.uint8
    assert set(np.unique(resampled)) <= set(np.unique(classes))
    # Every 20 m source pixel becomes a 2x2 block of identical 10 m classes.
    assert int((resampled == 9).sum()) == 4 * int((classes == 9).sum())
    assert np.array_equal(resampled[::2, ::2], classes)


def test_resample_marks_uncovered_grid_cells_as_no_data(tmp_path: Path) -> None:
    """A grid cell the SCL footprint does not reach is NO_DATA, hence invalid."""
    path = write_raster(tmp_path / "SCL.tif", np.full((4, 4), 4, dtype="uint8"),
                        coarse_transform())
    wide = common_grid([write_raster(tmp_path / "band.tif",
                                     np.full((SIZE, SIZE), 500, dtype="uint16"), transform_of())],
                       resolution=DEFAULT_RESOLUTION_M)
    resampled = resample_scl_to_grid(path, wide)

    assert resampled[0, 0] == 4, "the covered corner keeps its class"
    assert resampled[-1, -1] == 0, "the far corner is outside the SCL footprint"
    assert scl_valid_mask(resampled)[0, 0]
    assert not scl_valid_mask(resampled)[-1, -1]


def test_resample_keeps_class_boundaries_intact_on_a_shifted_grid(tmp_path: Path) -> None:
    """A resampled SCL must not blend a cloud edge into neighbouring surface pixels."""
    classes = np.full((SIZE // 2, SIZE // 2), 4, dtype="uint8")
    classes[:, :4] = 9
    path = write_raster(tmp_path / "SCL.tif", classes, coarse_transform())
    shifted = common_grid([write_raster(tmp_path / "band.tif",
                                        np.full((SIZE, SIZE), 500, dtype="uint16"),
                                        transform_of(origin=(ORIGIN[0] + 20.0, ORIGIN[1])))],
                          resolution=DEFAULT_RESOLUTION_M)
    resampled = resample_scl_to_grid(path, shifted)
    assert set(np.unique(resampled)) <= {0, 4, 9}
    assert int((resampled == 9).sum()) > 0


def test_write_scl_raster_round_trips(tmp_path: Path) -> None:
    classes = np.full((8, 8), 5, dtype="uint8")
    classes[0, 0] = 9
    grid = common_grid([write_raster(tmp_path / "band.tif", np.full((8, 8), 500, dtype="uint16"),
                                     transform_of(), nodata=None)], resolution=DEFAULT_RESOLUTION_M)
    path = tmp_path / "out" / "scl.tif"
    write_scl_raster(path, classes, grid, "SCL test")
    with rasterio.open(path) as src:
        assert src.count == 1 and src.dtypes[0] == "uint8" and src.nodata == 0
        assert src.crs.to_string() == grid.crs
        assert tuple(src.transform)[:6] == tuple(grid.transform)[:6]
        assert np.array_equal(src.read(1), classes)

    with pytest.raises(ValueError):
        write_scl_raster(tmp_path / "out" / "bad.tif", np.zeros((3, 3), dtype="uint8"), grid)


# ------------------------------------------------------ observation-level masking


def test_mask_observation_uses_the_scl_band(raw_pair: Path) -> None:
    before, after, grid = grid_of(raw_pair)
    cloud_before = mask_observation(before, grid)
    cloud_after = mask_observation(after, grid)

    assert cloud_before.available and cloud_after.available
    assert cloud_before.valid.shape == (grid.height, grid.width)
    assert cloud_before.classes.dtype == np.uint8

    cloud_view = block(cloud_after.valid, grid.transform, CLOUD_BOUNDS)
    shadow_view = block(cloud_before.valid, grid.transform, SHADOW_BOUNDS)
    assert not cloud_after.valid[cloud_view].any(), "the cloud block must be masked"
    assert not cloud_before.valid[shadow_view].any(), "the shadow block must be masked"
    # The cloud is an *after-only* feature: the before scene saw clean surface there.
    assert cloud_before.valid[cloud_view].all()
    assert cloud_after.valid[shadow_view].all()
    # Nothing outside the two flagged blocks is masked.
    assert cloud_before.masked_pixels + cloud_after.masked_pixels == 2 * BLOCK_PIXELS


def test_mask_observation_records_the_policy_and_source(raw_pair: Path) -> None:
    _, after, grid = grid_of(raw_pair)
    cloud = mask_observation(after, grid)
    report = cloud.report
    assert report["available"] is True
    assert report["valid_classes"] == list(SCL_VALID_CLASSES)
    assert report["source"]["resampling"] == "nearest"
    assert report["source"]["resolution_m"] == SCL_RESOLUTION_M
    assert report["source"]["target_resolution_m"] == DEFAULT_RESOLUTION_M
    assert report["mask_reasons"]["cloud"]["pixels"] == BLOCK_PIXELS
    assert cloud.to_dict()["masked_pixels"] == cloud.masked_pixels == BLOCK_PIXELS


def test_mask_observation_is_a_noop_without_scl(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw" / "no_scl"
    build_raw_pair(raw_dir, scl=False)
    _, after, grid = grid_of(raw_dir)

    cloud = mask_observation(after, grid)
    assert cloud.available is False
    assert cloud.classes is None
    assert cloud.valid.all(), "no SCL must not mask anything by cloud policy"
    assert cloud.masked_pixels == 0
    assert "no SCL asset" in cloud.report["note"]
    assert np.array_equal(cloud.class_array, np.zeros((grid.height, grid.width), dtype="uint8"))


def test_class_array_is_the_native_codes_when_available(raw_pair: Path) -> None:
    _, after, grid = grid_of(raw_pair)
    cloud = mask_observation(after, grid)
    assert np.array_equal(cloud.class_array, cloud.classes)
    assert cloud.class_array[block(cloud.class_array, grid.transform, CLOUD_BOUNDS)].max() == 9


def test_pair_valid_mask_intersects_and_validates_shape(raw_pair: Path) -> None:
    before, after, grid = grid_of(raw_pair)
    cloud_before = mask_observation(before, grid)
    cloud_after = mask_observation(after, grid)
    shared = pair_valid_mask(cloud_before, cloud_after)
    assert np.array_equal(shared, cloud_before.valid & cloud_after.valid)
    assert int((~shared).sum()) == 2 * BLOCK_PIXELS
    assert shared.sum() < cloud_before.valid.sum()

    class Other:
        valid = np.ones((3, 3), dtype=bool)

    with pytest.raises(ValueError):
        pair_valid_mask(cloud_before, Other())


def test_validate_raster_reports_scl_checks(raw_pair: Path) -> None:
    before, _, _ = grid_of(raw_pair)
    report = validate_raster(before)
    assert report.passed, report.problems
    for name in ("scl_present", "scl_readable", "scl_classes_known", "scl_has_usable_pixels"):
        assert report.checks[name] is True, name
    assert report.details["scl"]["masked_pixels"] == NATIVE_BLOCK_PIXELS
    assert report.details["scl"]["mask_reasons"]["shadow"]["classes"] == [2, 3]


def test_validate_raster_skips_scl_checks_without_scl(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw" / "no_scl"
    build_raw_pair(raw_dir, scl=False)
    before, _, _ = grid_of(raw_dir)
    report = validate_raster(before)
    assert report.passed, report.problems
    assert not any(name.startswith("scl_") for name in report.checks)


def test_cloud_mask_cli_reports_the_pair(raw_pair: Path, capsys: pytest.CaptureFixture) -> None:
    assert cloud_main(["--raw", str(raw_pair)]) == 0
    out = capsys.readouterr().out
    assert "BEFORE" in out and "AFTER" in out
    assert "usable in BOTH acquisitions" in out
    assert "shadow" in out, "the after scene's masked classes must be reported"


def test_cloud_mask_cli_lenient_policy_keeps_unclassified(raw_pair: Path,
                                                          capsys: pytest.CaptureFixture) -> None:
    assert cloud_main(["--raw", str(raw_pair), "--lenient"]) == 0
    out = capsys.readouterr().out
    assert "usable in BOTH acquisitions" in out



class _NoCloudMask:
    """Cloud mask that refuses to mask anything, standing in for 'SCL ignored'."""

    def __init__(self, grid):
        self.grid = grid
        self.valid = np.ones((grid.height, grid.width), dtype=bool)


# --------------------------------------------------------------- pipeline integration
# --------------------------------------------------------------- pipeline integration


def test_align_rasters_applies_the_cloud_mask(raw_pair: Path) -> None:
    before, _, grid = grid_of(raw_pair)
    aligned = align_rasters(before, grid, cloud=mask_observation(before, grid))

    shadow_view = block(aligned.data, grid.transform, SHADOW_BOUNDS)
    assert np.isnan(aligned.data[(Ellipsis, *shadow_view)]).all(), \
        "masked pixels must be NaN in every band"
    assert not aligned.valid[shadow_view].any()
    assert aligned.valid[0, 0] and np.isfinite(aligned.data[:, 0, 0]).all()


def test_align_rasters_masks_the_same_pixels_with_or_without_an_explicit_mask(raw_pair: Path) -> None:
    before, _, grid = grid_of(raw_pair)
    implicit = align_rasters(before, grid)
    explicit = align_rasters(before, grid, cloud=mask_observation(before, grid))
    assert np.array_equal(implicit.valid, explicit.valid)
    assert np.allclose(implicit.data, explicit.data, equal_nan=True)


def test_align_rasters_matches_the_band_and_cloud_masks(raw_pair: Path) -> None:
    before, _, grid = grid_of(raw_pair)
    cloud = mask_observation(before, grid)
    aligned = align_rasters(before, grid, cloud=cloud)
    for band in BAND_ORDER:
        assert np.array_equal(aligned.band_valid[band], cloud.valid), band
    assert np.array_equal(aligned.valid, cloud.valid)


def test_align_rasters_without_scl_keeps_every_pixel(raw_pair: Path) -> None:
    before, _, grid = grid_of(raw_pair)
    cloud = mask_observation(before, grid)
    cloud.valid = np.ones_like(cloud.valid)
    assert align_rasters(before, grid, cloud=cloud).valid.all()


def test_align_rasters_rejects_a_cloud_mask_of_the_wrong_shape(raw_pair: Path) -> None:
    before, _, grid = grid_of(raw_pair)
    cloud = mask_observation(before, grid)
    cloud.valid = np.ones((2, 2), dtype=bool)
    with pytest.raises(ValueError):
        align_rasters(before, grid, cloud=cloud)


def test_cloud_pixels_cannot_fabricate_change(raw_pair: Path) -> None:
    """The whole point of the mask: invented change must never reach the comparison."""
    before, after, grid = grid_of(raw_pair)
    unmasked = ndvi_difference(
        ndvi_of(align_rasters(before, grid, cloud=_NoCloudMask(grid))),
        ndvi_of(align_rasters(after, grid, cloud=_NoCloudMask(grid))),
    )

    for name, bounds in (("cloud", CLOUD_BOUNDS), ("shadow", SHADOW_BOUNDS)):
        fabricated = unmasked[block(unmasked, grid.transform, bounds)]
        assert np.isfinite(fabricated).all(), f"precondition: {name} block valid without SCL"
        assert float(fabricated.max()) > 0.2, f"precondition: {name} block fabricates greening"

    comparison = ndvi_difference(ndvi_of(align_rasters(before, grid)),
                                 ndvi_of(align_rasters(after, grid)))
    for name, bounds in (("cloud", CLOUD_BOUNDS), ("shadow", SHADOW_BOUNDS)):
        flagged = comparison[block(comparison, grid.transform, bounds)]
        assert np.isnan(flagged).all(), f"the {name} block must not be compared at all"


def test_only_the_real_change_survives_masking(raw_pair: Path) -> None:
    before, after, grid = grid_of(raw_pair)
    difference = ndvi_difference(ndvi_of(align_rasters(before, grid)),
                                 ndvi_of(align_rasters(after, grid)))
    loss_view = block(difference, grid.transform, LOSS_BOUNDS)
    assert (difference[loss_view] < -0.2).all(), "the genuine browning must survive"
    assert not bool((difference > 0.2).any()), "no fabricated greening may remain"
    finite = np.isfinite(difference)
    assert int((difference < -0.2).sum()) == BLOCK_PIXELS
    assert int(finite.sum()) == SIZE * SIZE - 2 * BLOCK_PIXELS


def test_tensors_stay_finite_and_cloud_aware(raw_pair: Path, tmp_path: Path) -> None:
    processed = tmp_path / "processed" / "demo_area"
    metadata = prepare_pair(raw_dir=raw_pair, processed_dir=processed, validate=True)
    before, after = load_sentinel_pair(raw_pair)
    grid = common_grid([*before.paths, *after.paths], resolution=DEFAULT_RESOLUTION_M)
    cloud_valid = np.load(processed / "cloud_valid_mask.npy")
    aligned = align_rasters(after, grid)
    tensor = convert_to_tensor(np.nan_to_num(aligned.data, nan=0.0))

    assert metadata["cloud_mask"]["applied"] is True
    assert metadata["cloud_mask"]["masked_pixels"] == 2 * BLOCK_PIXELS
    assert np.array_equal(cloud_valid, pair_valid_mask(mask_observation(before, grid),
                                                       mask_observation(after, grid)))
    assert int((~cloud_valid).sum()) == 2 * BLOCK_PIXELS
    assert tensor.shape == (len(BAND_ORDER), SIZE, SIZE)
    assert bool(np.isfinite(tensor.numpy()).all())


def test_apply_mask_nans_a_stack_and_leaves_valid_pixels_alone() -> None:
    masked = apply_mask(np.ones((4, 2, 2), dtype="float32"),
                        np.array([[True, False], [False, True]]))
    assert masked.dtype == np.float32
    assert np.isfinite(masked[:, 0, 0]).all() and np.isfinite(masked[:, 1, 1]).all()
    assert np.isnan(masked[:, 0, 1]).all() and np.isnan(masked[:, 1, 0]).all()


def test_apply_mask_accepts_a_single_band() -> None:
    masked = apply_mask(np.arange(4, dtype="float32").reshape(2, 2), np.eye(2, dtype=bool))
    assert np.isnan(masked[0, 1]) and masked[1, 1] == 3.0


def test_apply_mask_rejects_a_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        apply_mask(np.ones((1, 4, 4)), np.ones((2, 2), dtype=bool))


def test_class_histogram_counts_and_names_every_class() -> None:
    histogram = {record["class"]: record for record in class_histogram(np.array([4, 4, 5, 9, 0]))}
    assert histogram[4]["pixels"] == 2
    assert histogram[4]["name"] == "vegetation"
    assert histogram[4]["usable"] is True
    assert histogram[9]["usable"] is False
    assert histogram[0]["usable"] is False
    assert histogram[4]["fraction"] == pytest.approx(2 / 5, abs=1e-6)


def test_class_histogram_reports_non_finite_separately() -> None:
    histogram = class_histogram(np.array([4.0, np.nan]))
    assert any(record["class"] is None and record["pixels"] == 1 for record in histogram)


def test_scl_mask_report_totals_and_reasons() -> None:
    classes = np.array([[4, 5, 6, 7], [0, 1, 2, 3], [8, 9, 10, 11]], dtype="uint8")
    report = scl_mask_report(classes)
    assert report["pixels"] == 12
    assert report["usable_pixels"] == 3
    assert report["masked_pixels"] == 9
    assert report["usable_fraction"] == pytest.approx(0.25, abs=1e-6)
    assert report["usable_fraction"] + report["masked_fraction"] == pytest.approx(1.0, abs=1e-6)
    assert report["unknown_classes"] == []
    assert report["mask_reasons"]["cloud"]["pixels"] == 2
    assert report["mask_reasons"]["shadow"]["pixels"] == 2
    assert sum(item["pixels"] for item in report["mask_reasons"].values()) == 9


def test_scl_mask_report_flags_unknown_classes() -> None:
    report = scl_mask_report(np.array([4, 42], dtype="uint8"))
    assert report["unknown_classes"] == [42]
    assert report["histogram"][-1]["name"] == "unknown_class_42"
