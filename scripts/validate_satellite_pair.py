"""Validate the raw Sentinel-2 pair and the processed (aligned) outputs.

Produces a single PASS/FAIL report covering:
  1. the raw per-band GeoTIFFs (readability, CRS, bands, resolution, values, nodata),
  2. spatial alignment of the two acquisitions (CRS compatibility, footprint overlap),
  3. the processed artifacts written by ``app.services.preprocessing.prepare_pair``,
  4. the frozen tensor contract handed to the Siamese U-Net: [4, H, W] float32.

Usage
-----
    backend/.venv/Scripts/python.exe scripts/validate_satellite_pair.py
    backend/.venv/Scripts/python.exe scripts/validate_satellite_pair.py --skip-build
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND = REPO_ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.services.preprocessing import (  # noqa: E402  (import after sys.path bootstrap)
    BAND_ORDER,
    BAND_ROLES,
    DEFAULT_PROCESSED_DIR,
    DEFAULT_RAW_DIR,
    REFLECTANCE_RANGE,
    AlignedObservation,
    Observation,
    ValidationResult,
    align_rasters,
    check_bounds,
    check_crs,
    convert_to_tensor,
    load_sentinel_pair,
    prepare_pair,
    validate_raster,
)
from app.services.cloud_mask import SCL_CLASSES  # noqa: E402  (import after sys.path bootstrap)
from app.utils.geo import common_grid  # noqa: E402

WIDTH = 78
_CONSOLE_ENCODING = (getattr(sys.stdout, "encoding", "") or "").lower()
UNICODE_SAFE = "utf" in _CONSOLE_ENCODING
OK = "\u2713" if UNICODE_SAFE else "[PASS]"
BAD = "\u2717" if UNICODE_SAFE else "[FAIL]"


def line(ok: bool, text: str) -> str:
    return f"{OK if ok else BAD} {text}"


def print_check(ok: bool, text: str) -> None:
    print(f"  {line(ok, text)}")


def print_observation_block(report: ValidationResult, observation: Observation) -> None:
    """Raw-input section for one acquisition (the report block from the Phase 2 spec)."""
    checks = report.checks
    details = report.details
    print(f"\n{observation.label.upper()}")
    print(f"  source      : {details.get('width')}x{details.get('height')} px, "
          f"{details.get('area_km2')} km2, nodata={details.get('nodata')}")
    print(f"  acquisition : {observation.date or 'n/a'} | {observation.item_id or 'n/a'} | "
          f"tile {observation.mgrs_tile or 'n/a'} | cloud {observation.cloud_cover}%")
    print(f"  CRS         : {details.get('crs')} | resolution {details.get('resolution')}")
    print(f"  bands       : {', '.join(BAND_ORDER)} ({', '.join(BAND_ROLES[b] for b in BAND_ORDER)})")
    print(f"  bounds      : {[round(float(v), 1) for v in details.get('bounds', [])]}")
    print(f"  files       : {observation.directory}")
    for band in BAND_ORDER:
        stats = details.get("bands", {}).get(band, {})
        if "error" in stats:
            print(f"    {band}: ERROR {stats['error']}")
            continue
        print(f"    {band}: min={stats['min']} max={stats['max']} mean={stats['mean']} "
              f"std={stats['std']} nodata_px={stats['nodata_pixels']} "
              f"valid={stats['valid_fraction']:.4%}")
    primaries = [
        (checks.get("readable", False), "file readable"),
        (checks.get("crs_valid", False), "CRS valid"),
        (checks.get("crs_consistent_across_bands", False), "CRS consistent across bands"),
        *[(checks.get(f"band_readable_{band}", False), f"{band} found") for band in BAND_ORDER],
        (checks.get("resolution_is_10m", False), "10 m resolution"),
        (checks.get("bands_share_grid", False), "bands share one grid"),
        (checks.get("has_valid_pixels", False), "valid pixels present"),
        (checks.get("mostly_valid_pixels", False), "valid pixels >= 99%"),
        (checks.get("no_non_finite_pixels", False), "no NaN/Inf pixels"),
        (checks.get("band_has_variation_B08", False), "bands carry variation"),
        (checks.get("band_values_in_sensor_range_B08", False), "values inside uint16 sensor range"),
    ]
    for ok, text in primaries:
        print_check(ok, text)


def print_alignment_block(crs_report: ValidationResult, bounds_report: ValidationResult,
                          grid, aligned: dict[str, AlignedObservation]) -> None:
    print("\nSPATIAL ALIGNMENT")
    print(f"  analysis CRS    : {crs_report.details.get('analysis_crs')} "
          f"(before {crs_report.details.get('before_crs')}, after {crs_report.details.get('after_crs')})")
    print(f"  overlap         : {bounds_report.details.get('overlap_fraction_of_smaller'):.2%} "
          f"of the smaller footprint ({bounds_report.details.get('overlap_area_km2')} km2)")
    for ok, text in [
        (crs_report.checks.get("crs_present", False), "CRS valid on both observations"),
        (crs_report.checks.get("crs_transformable_to_analysis_crs", False),
         "both CRS transformable to the analysis CRS"),
        (crs_report.checks.get("crs_units_metric", False), "metric CRS units (metres)"),
        (crs_report.details.get("crs_identical", False), "identical acquisition CRS"),
        (bounds_report.checks.get("bounds_overlap", False), "geographic overlap"),
        (bounds_report.checks.get("overlap_covers_smaller_footprint", False), "overlap >= 25% of smaller footprint"),
        (all(a.data.shape[1:] == (grid.height, grid.width) for a in aligned.values()),
         f"common grid generated ({grid.width}x{grid.height} @ {grid.resolution_m} m)"),
        (aligned["before"].data.shape == aligned["after"].data.shape,
         f"dimensions match ({aligned['before'].data.shape} == {aligned['after'].data.shape})"),
        (grid.crs == crs_report.details.get("analysis_crs"), "grid CRS matches analysis CRS"),
    ]:
        print_check(ok, text)
    for label, observation in aligned.items():
        print(f"  {label} valid: {observation.valid.mean():.4%} of pixels")


def validate_processed(processed_dir: Path, metadata: dict) -> tuple[list[tuple[bool, str]], dict]:
    """Check every processed artifact on disk against ``metadata``."""
    expected = {
        "before.tif": "aligned 4-band reflectance (before)",
        "after.tif": "aligned 4-band reflectance (after)",
        "before.npy": "tensor array [4,H,W] (before)",
        "after.npy": "tensor array [4,H,W] (after)",
        "before_valid_mask.npy": "validity mask (before)",
        "after_valid_mask.npy": "validity mask (after)",
        "valid_mask.npy": "shared validity mask",
        "ndvi_before.tif": "NDVI before",
        "ndvi_after.tif": "NDVI after",
        "ndvi_difference.tif": "NDVI change map",
        "metadata.json": "provenance metadata",
    }
    checks: list[tuple[bool, str]] = []
    facts: dict = {}
    for name, description in expected.items():
        path = processed_dir / name
        checks.append((path.is_file() and path.stat().st_size > 0, f"{name} - {description}"))

    grid_height, grid_width = metadata["height"], metadata["width"]
    tif_facts = {}
    for label in ("before", "after"):
        with rasterio.open(processed_dir / f"{label}.tif") as src:
            tif_facts[label] = {
                "count": src.count, "dtype": src.dtypes[0], "crs": src.crs.to_string(),
                "shape": (src.height, src.width), "transform": tuple(src.transform)[:6],
            }
    facts["tifs"] = tif_facts
    before_tif, after_tif = tif_facts["before"], tif_facts["after"]

    checks.append((before_tif["count"] == len(BAND_ORDER) and after_tif["count"] == len(BAND_ORDER),
                   f"both tif files have {len(BAND_ORDER)} bands"))
    checks.append((before_tif["dtype"] == "float32" and after_tif["dtype"] == "float32",
                   "reflectance stored as float32"))
    checks.append((before_tif["shape"] == after_tif["shape"] == (grid_height, grid_width),
                   f"tif dimensions match the grid ({grid_width}x{grid_height})"))
    checks.append((before_tif["crs"] == after_tif["crs"] == metadata["crs"], "tif CRS matches metadata"))
    checks.append((before_tif["transform"] == after_tif["transform"], "tif transforms are identical"))

    before_array = np.load(processed_dir / "before.npy")
    after_array = np.load(processed_dir / "after.npy")
    before_mask = np.load(processed_dir / "before_valid_mask.npy")
    after_mask = np.load(processed_dir / "after_valid_mask.npy")
    shared_mask = np.load(processed_dir / "valid_mask.npy")
    facts["arrays"] = {
        "before": {"shape": list(before_array.shape), "dtype": before_array.dtype.name},
        "after": {"shape": list(after_array.shape), "dtype": after_array.dtype.name},
    }
    checks.append((before_array.shape == after_array.shape == (len(BAND_ORDER), grid_height, grid_width),
                   f"npy arrays are [4, {grid_height}, {grid_width}]"))
    checks.append((before_array.dtype == np.float32 and after_array.dtype == np.float32,
                   "npy arrays are float32"))
    checks.append((bool(np.isfinite(before_array).all() and np.isfinite(after_array).all()),
                   "npy arrays contain no NaN/Inf (invalid pixels filled with nodata)"))
    checks.append((before_mask.shape == after_mask.shape == shared_mask.shape == (grid_height, grid_width),
                   "validity masks match the grid shape"))
    checks.append((bool(np.array_equal(shared_mask, before_mask & after_mask)),
                   "shared mask equals before AND after masks"))
    checks.append((float(before_array.min()) >= REFLECTANCE_RANGE[0]
                   and float(after_array.max()) <= REFLECTANCE_RANGE[1],
                   f"reflectance within {list(REFLECTANCE_RANGE)}"))
    checks.append((float(shared_mask.mean()) > 0.99, f"shared valid fraction is {shared_mask.mean():.4%}"))

    ndvi_facts: dict = {}
    for name, boundary in (("ndvi_before", 1.0), ("ndvi_after", 1.0), ("ndvi_difference", 2.0)):
        with rasterio.open(processed_dir / f"{name}.tif") as src:
            values = src.read(1, masked=True)
            ndvi_facts[name] = {
                "shape": (src.height, src.width),
                "nodata": None if src.nodata is None else float(src.nodata),
                "min": round(float(values.min()), 6), "max": round(float(values.max()), 6),
                "mean": round(float(values.mean()), 6),
                "valid_pixels": int(values.count()),
            }
        checks.append((values.shape == (grid_height, grid_width), f"{name}.tif matches the grid"))
        checks.append((float(values.min()) >= -boundary and float(values.max()) <= boundary,
                       f"{name} values within [{-boundary:g}, {boundary:g}]"))
        checks.append((int(values.count()) > 0, f"{name} has valid pixels"))
    facts["ndvi"] = ndvi_facts
    checks.append((metadata["ndvi"]["difference"]["valid_pixels"] > 0, "NDVI statistics computed"))
    return checks, facts


def print_tensor_block(metadata: dict, before: np.ndarray, after: np.ndarray) -> list[tuple[bool, str]]:
    """Validate the frozen AI input contract: two [4, H, W] float32 tensors."""
    before_tensor = convert_to_tensor(before)
    after_tensor = convert_to_tensor(after)
    shape = (len(BAND_ORDER), metadata["height"], metadata["width"])
    print("\nTENSOR CONTRACT (input to the Siamese U-Net)")
    print(f"  before: {tuple(before_tensor.shape)} {before_tensor.dtype} "
          f"min={float(before_tensor.min()):.4f} max={float(before_tensor.max()):.4f} "
          f"mean={float(before_tensor.mean()):.4f}")
    print(f"  after : {tuple(after_tensor.shape)} {after_tensor.dtype} "
          f"min={float(after_tensor.min()):.4f} max={float(after_tensor.max()):.4f} "
          f"mean={float(after_tensor.mean()):.4f}")
    return [
        (tuple(before_tensor.shape) == shape == tuple(after_tensor.shape),
         f"shape is {list(shape)} for both observations"),
        (str(before_tensor.dtype) == "torch.float32" and str(after_tensor.dtype) == "torch.float32",
         "dtype is float32"),
        (bool(np.isfinite(before_tensor.numpy()).all() and np.isfinite(after_tensor.numpy()).all()),
         "no NaN/Inf reaches the model"),
        (0.0 <= float(before_tensor.min()) and float(before_tensor.max()) <= 1.0
         and 0.0 <= float(after_tensor.min()) and float(after_tensor.max()) <= 1.0,
         "values inside [0, 1] reflectance range"),
        (float(before_tensor.mean()) > 0.0 and float(after_tensor.mean()) > 0.0,
         "both tensors carry signal (mean > 0)"),
        (metadata["tensor"]["band_order"] == list(BAND_ORDER),
         f"band order fixed to {list(BAND_ORDER)}"),
    ]


def print_cloud_block(metadata: dict, processed_dir: Path) -> list[tuple[bool, str]]:
    """Validate the SCL cloud / invalid-pixel masking applied to both observations."""
    cloud = metadata.get("cloud_mask")
    print("\nCLOUD MASK (Sentinel-2 L2A SCL)")
    if not cloud:
        print_check(False, "cloud_mask metadata missing from metadata.json")
        return [(False, "metadata contains no cloud_mask block")]

    print(f"  policy      : usable classes {cloud['valid_classes']}; "
          f"masked {cloud['masked_classes']}")
    print(f"  grid        : {cloud['before']['pixels']} px, "
          f"{cloud['masked_pixels']} masked in either observation ({cloud['masked_fraction']:.4%})")
    for label in ("before", "after"):
        entry = cloud[label]
        report = entry["report"]
        if not entry["available"]:
            print(f"  {label:12s}: no SCL asset - no pixels masked by cloud policy")
            continue
        print(f"  {label:12s}: usable {report['usable_fraction']:.4%}, "
              f"masked {report['masked_pixels']} px ({report['masked_fraction']:.4%})")
        for reason, detail in report["mask_reasons"].items():
            print(f"                {reason:24s} {detail['pixels']:>7} px "
                  f"({detail['fraction_of_image']:.4%}) classes {detail['classes']}")

    artifacts = {
        "cloud_valid_mask.npy": "combined cloud validity",
        "scl_before.tif": "SCL classes on the analysis grid (before)",
        "scl_after.tif": "SCL classes on the analysis grid (after)",
    }
    checks: list[tuple[bool, str]] = []
    for name, description in artifacts.items():
        path = processed_dir / name
        checks.append((path.is_file() and path.stat().st_size > 0, f"{name} - {description}"))

    try:
        cloud_valid = np.load(processed_dir / "cloud_valid_mask.npy")
        shared_valid = np.load(processed_dir / "valid_mask.npy")
        before_valid = np.load(processed_dir / "before_valid_mask.npy")
        after_valid = np.load(processed_dir / "after_valid_mask.npy")
    except Exception as error:  # pragma: no cover - depends on missing artifacts
        return checks + [(False, f"validity masks unreadable: {error}")]

    checks.append((cloud_valid.shape == shared_valid.shape,
                   "cloud validity mask matches the grid shape"))
    checks.append((bool(np.array_equal(shared_valid, before_valid & after_valid)),
                   "shared validity mask equals before AND after band masks"))
    mask_applied = bool(np.all(shared_valid <= cloud_valid))
    checks.append((mask_applied, "band validity is a subset of cloud validity (mask enforced)"))
    checks.append((float(cloud_valid.mean()) >= 0.5,
                   f"usable in both acquisitions: {float(cloud_valid.mean()):.4%}"))

    for path in ("scl_before.tif", "scl_after.tif"):
        with rasterio.open(processed_dir / path) as src:
            classes = src.read(1, masked=False)
            checks.append((src.count == 1 and src.dtypes[0] == "uint8",
                           f"{path} is a single uint8 class raster"))
            checks.append((tuple(classes.shape) == (metadata["height"], metadata["width"]),
                           f"{path} matches the grid ({metadata['width']}x{metadata['height']})"))
            present = {int(code) for code in np.unique(classes)}
            checks.append((present <= set(SCL_CLASSES),
                           f"{path} carries only known SCL classes ({sorted(present)})"))

    checks.append((metadata["cloud_mask"]["policy"] and metadata["cloud_mask"]["module"],
                   "masking policy recorded in metadata"))
    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TerraGuard AI - Sentinel-2 pair validation")
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--processed", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--skip-build", action="store_true",
                        help="validate the existing processed artifacts instead of regenerating them")
    parser.add_argument("--report", type=Path, default=None, help="optional path for a JSON report")
    args = parser.parse_args(argv)

    try:  # make box-drawing/check glyphs safe on legacy Windows consoles
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # pragma: no cover - depends on stream type
        pass

    print("=" * WIDTH)
    print(" TerraGuard AI - Satellite Pair Validation")
    print("=" * WIDTH)
    print(f" raw       : {args.raw}")
    print(f" processed : {args.processed}")

    failures: list[str] = []
    before, after = load_sentinel_pair(args.raw)

    raw_reports = {"before": validate_raster(before), "after": validate_raster(after)}
    for label, report in raw_reports.items():
        print_observation_block(report, before if label == "before" else after)
        failures.extend(f"raw {label}: {problem}" for problem in report.problems)

    crs_report = check_crs(before, after)
    bounds_report = check_bounds(before, after)
    grid = common_grid([*before.paths, *after.paths], resolution=10.0)
    aligned = {"before": align_rasters(before, grid), "after": align_rasters(after, grid)}
    print_alignment_block(crs_report, bounds_report, grid, aligned)
    failures.extend(f"alignment: {problem}" for problem in crs_report.problems + bounds_report.problems)

    if args.skip_build:
        metadata = json.loads((args.processed / "metadata.json").read_text(encoding="utf-8"))
        print("\nBUILD")
        print_check(True, "skipped (--skip-build): validating existing artifacts")
    else:
        metadata = prepare_pair(raw_dir=args.raw, processed_dir=args.processed)
        print("\nBUILD")
        print_check(True, "prepare_pair() regenerated the processed artifacts")

    checks, facts = validate_processed(args.processed, metadata)
    print("\nOUTPUT")
    for ok, text in checks:
        print_check(ok, text)
    failures.extend(text for ok, text in checks if not ok)

    cloud_checks = print_cloud_block(metadata, args.processed)
    for ok, text in cloud_checks:
        print_check(ok, text)
    failures.extend(text for ok, text in cloud_checks if not ok)

    before_array = np.load(args.processed / "before.npy")
    after_array = np.load(args.processed / "after.npy")
    tensor_checks = print_tensor_block(metadata, before_array, after_array)
    for ok, text in tensor_checks:
        print_check(ok, text)
    failures.extend(text for ok, text in tensor_checks if not ok)

    ndvi = metadata["ndvi"]
    print("\nNDVI")
    print(f"  formula   : {ndvi['formula']}")
    print(f"  before    : mean {ndvi['before']['mean']} (min {ndvi['before']['min']}, max {ndvi['before']['max']})")
    print(f"  after     : mean {ndvi['after']['mean']} (min {ndvi['after']['min']}, max {ndvi['after']['max']})")
    print(f"  change    : mean {ndvi['difference']['mean']} "
          f"(min {ndvi['difference']['min']}, max {ndvi['difference']['max']})")
    print(f"  browning  : >0.1 {ndvi['change_areas']['loss_at_0.1_fraction']:.2%}, "
          f">0.2 {ndvi['change_areas']['loss_at_0.2_fraction']:.2%}, "
          f">0.3 {ndvi['change_areas']['loss_at_0.3_fraction']:.2%}")
    print(f"  greening  : >0.1 {ndvi['change_areas']['gain_at_0.1_fraction']:.2%}, "
          f">0.2 {ndvi['change_areas']['gain_at_0.2_fraction']:.2%}, "
          f">0.3 {ndvi['change_areas']['gain_at_0.3_fraction']:.2%}")
    print(f"  pixels    : {ndvi['difference']['valid_pixels']} valid of "
          f"{ndvi['difference']['valid_pixels'] + ndvi['difference']['invalid_pixels']}")

    print("\n" + "=" * WIDTH)
    if failures:
        print(f"RESULT: FAIL ({len(failures)} check(s) failed)")
        for failure in failures:
            print(f"  - {failure}")
    else:
        print("RESULT: PASS")
    print("=" * WIDTH)
    print(" REAL SENTINEL-2 -> PREPROCESSING -> [4,H,W] -> [4,H,W] -> READY FOR AI")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({
            "raw": {label: report.to_dict() for label, report in raw_reports.items()},
            "alignment": {"crs": crs_report.to_dict(), "bounds": bounds_report.to_dict(),
                          "grid": grid.to_dict()},
            "processed": {"checks": [{"check": text, "passed": ok} for ok, text in checks], "facts": facts},
            "tensor": {"checks": [{"check": text, "passed": ok} for ok, text in tensor_checks]},
            "metadata": metadata,
            "failures": failures,
            "passed": not failures,
        }, indent=2), encoding="utf-8")
        print(f"JSON report written to {args.report}")

    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())

