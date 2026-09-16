"""Sentinel-2 L2A Scene Classification (SCL) cloud / invalid-pixel masking.

Why this exists
---------------
Dividing digital numbers by 10000 (``preprocessing.normalize_bands``) makes reflectance
*comparable*; it does not make it *meaningful*. A pixel under a cloud, in a cloud shadow,
in thin cirrus, saturated, unclassified or simply absent can carry a perfectly plausible
reflectance and still fabricate change: the same field looks browner simply because a
cloud edge moved between the two acquisitions. A Siamese network cannot tell that apart
from a real environmental change, so the mask has to be applied before the model - not
learned by it.

Sentinel-2 L2A ships a per-pixel ``SCL`` band that classifies exactly this, so the rule is:

    usable surface = {4 VEGETATION, 5 NOT_VEGETATED, 6 WATER}

Everything else is masked and turned into NaN in the reflectance stack:

    reason           classes
    ---------------  --------------------------------------------------------
    no_data          0  NO_DATA
    saturated        1  SATURATED_OR_DEFECTIVE
    shadow           2  DARK_AREA_OR_CAST_SHADOW, 3 CLOUD_SHADOW
    unclassified     7  UNCLASSIFIED
    cloud            8  CLOUD_MEDIUM_PROBABILITY, 9 CLOUD_HIGH_PROBABILITY
    cirrus           10 THIN_CIRRUS
    snow_or_ice      11 SNOW_OR_ICE

Class codes are categorical, never continuous: the SCL band is always resampled onto the
analysis grid with nearest-neighbour (bilinear would invent classes such as "7.5") and a
grid cell the source does not cover becomes NO_DATA, hence invalid.

Validity is per observation and then intersected: a pixel is usable for change detection
only when it is a usable surface class in *both* acquisitions. Like the reflectance
normalization, this is fixed and reproducible - no per-image threshold is fitted.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.utils.geo import (
    DEFAULT_RESOLUTION_M,
    NAN,
    RasterGrid,
    common_grid,
    read_band,
    resample_to_grid,
)

SCL_BAND = "SCL"
"""Auxiliary band name used on disk (``<observation>/SCL.tif``)."""

SCL_NATIVE_RESOLUTION_M = 20.0
"""Sentinel-2 L2A classifies at 20 m even when the 10 m bands are used."""

SCL_NO_DATA = 0

SCL_CLASSES: dict[int, str] = {
    0: "no_data",
    1: "saturated_or_defective",
    2: "dark_area_or_cast_shadow",
    3: "cloud_shadow",
    4: "vegetation",
    5: "not_vegetated",
    6: "water",
    7: "unclassified",
    8: "cloud_medium_probability",
    9: "cloud_high_probability",
    10: "thin_cirrus",
    11: "snow_or_ice",
}
"""Full Sentinel-2 L2A Scene Classification Layer taxonomy (ESA sen2cor)."""

SCL_VALID_CLASSES: tuple[int, ...] = (4, 5, 6)
"""Classes treated as a usable ground surface. Strict by default."""

SCL_VALID_CLASSES_LENIENT: tuple[int, ...] = (4, 5, 6, 7)
"""Opt-in variant that also trusts class 7 UNCLASSIFIED (sen2cor's low-confidence bin).

Not the default: unclassified pixels are exactly where a classifier was unsure, and
change detection would read their uncertainty as change.
"""

SCL_MASKED_CLASSES: tuple[int, ...] = tuple(
    code for code in sorted(SCL_CLASSES) if code not in SCL_VALID_CLASSES
)
"""Every class the default policy refuses to trust, derived from the taxonomy above."""

MASK_REASONS: dict[str, tuple[int, ...]] = {
    "no_data": (0,),
    "saturated_or_defective": (1,),
    "shadow": (2, 3),
    "unclassified": (7,),
    "cloud": (8, 9),
    "cirrus": (10,),
    "snow_or_ice": (11,),
}
"""Human-readable grouping of the masked classes, used in reports."""

SCL_RESAMPLING = "nearest"
MASK_METHOD = (
    "SCL classes {4,5,6} kept as usable surface; "
    f"{len(SCL_MASKED_CLASSES)} other classes masked; SCL resampled 20 m -> analysis grid with "
    "nearest neighbour; masked pixels become NaN in every band of both observations"
)


def class_name(code: int) -> str:
    """Human-readable name of an SCL class code."""
    return SCL_CLASSES.get(int(code), f"unknown_class_{int(code)}")


def load_scl(path: str | Path) -> np.ndarray:
    """Read an SCL raster at native resolution as uint8 class codes (NO_DATA kept as 0)."""
    values = np.nan_to_num(read_band(path, masked=False), nan=float(SCL_NO_DATA))
    return np.clip(np.rint(values), 0, 255).astype("uint8")


def resample_scl_to_grid(path: str | Path, grid: RasterGrid,
                         no_data_class: int = SCL_NO_DATA) -> np.ndarray:
    """Resample SCL classes onto the analysis grid, nearest-neighbour, as uint8.

    Nearest neighbour is mandatory: class codes are labels, so interpolation is
    meaningless. Cells the source footprint does not cover become ``no_data_class``
    (NO_DATA) and are therefore invalid downstream.
    """
    from rasterio.enums import Resampling

    values = resample_to_grid(path, grid, band=1, resampling=Resampling.nearest, dst_nodata=NAN)
    filled = np.where(np.isfinite(values), np.rint(values), float(no_data_class))
    return np.clip(filled, 0.0, 255.0).astype("uint8")


def scl_valid_mask(scl: np.ndarray, valid_classes: tuple[int, ...] = SCL_VALID_CLASSES) -> np.ndarray:
    """Boolean mask of usable-surface pixels for an SCL array of any shape.

    Strict set membership: an unknown code, a fractional code, NaN or Inf is invalid.
    """
    values = np.asarray(scl, dtype="float64")
    if values.size == 0:
        raise ValueError("scl array is empty")
    return np.isin(values, np.asarray(sorted(set(valid_classes)), dtype="float64"))


def class_histogram(scl: np.ndarray, valid_classes: tuple[int, ...] = SCL_VALID_CLASSES) -> list[dict]:
    """Per-class pixel counts, ordered by class code, names resolved, validity flagged."""
    values = np.asarray(scl, dtype="float64")
    total = int(values.size)
    finite = np.isfinite(values)
    records: list[dict] = []
    if finite.any():
        codes, counts = np.unique(values[finite], return_counts=True)
        for code, count in zip(codes, counts):
            integer = int(round(float(code)))
            records.append({
                "class": integer,
                "name": class_name(integer),
                "pixels": int(count),
                "fraction": round(int(count) / total, 6) if total else 0.0,
                "usable": integer in set(valid_classes),
            })
    non_finite = int((~finite).sum())
    if non_finite:
        records.append({"class": None, "name": "non_finite", "pixels": non_finite,
                        "fraction": round(non_finite / total, 6) if total else 0.0, "usable": False})
    return records


def scl_mask_report(scl: np.ndarray, valid_classes: tuple[int, ...] = SCL_VALID_CLASSES) -> dict:
    """JSON-serializable summary of an SCL array: histogram, masked fraction, reasons."""
    values = np.asarray(scl, dtype="float64")
    total = int(values.size)
    valid = int(scl_valid_mask(values, valid_classes).sum())
    histogram = class_histogram(values, valid_classes)

    unknown = sorted({
        int(round(float(record["class"]))) for record in histogram
        if record["class"] is not None and int(round(float(record["class"]))) not in SCL_CLASSES
    })
    reasons: dict[str, dict] = {}
    for reason, codes in MASK_REASONS.items():
        pixels = sum(record["pixels"] for record in histogram if record["class"] in codes)
        if pixels:
            reasons[reason] = {
                "classes": list(codes),
                "pixels": pixels,
                "fraction_of_image": round(pixels / total, 6) if total else 0.0,
            }

    return {
        "method": MASK_METHOD,
        "valid_classes": list(valid_classes),
        "valid_class_names": [class_name(code) for code in valid_classes],
        "masked_classes": list(SCL_MASKED_CLASSES),
        "pixels": total,
        "usable_pixels": valid,
        "masked_pixels": total - valid,
        "usable_fraction": round(valid / total, 6) if total else 0.0,
        "masked_fraction": round((total - valid) / total, 6) if total else 0.0,
        "unknown_classes": unknown,
        "histogram": histogram,
        "mask_reasons": reasons,
    }


def apply_mask(array: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return ``array`` as float32 with masked pixels set to NaN.

    Accepts a single band ``[H, W]`` or a stack ``[C, H, W]``; the mask is always ``[H, W]``.
    """
    data = np.asarray(array, dtype="float32")
    valid = np.asarray(mask, dtype=bool)
    if data.shape[-2:] != valid.shape:
        raise ValueError(f"mask shape {valid.shape} does not match data {data.shape}")
    return np.where(valid, data, np.float32(NAN)).astype("float32")


def write_scl_raster(path: str | Path, classes: np.ndarray, grid: RasterGrid, label: str = "") -> None:
    """Write SCL class codes on the analysis grid as uint8 (NO_DATA = 0)."""
    import rasterio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(classes, dtype="uint8")
    if data.shape != (grid.height, grid.width):
        raise ValueError(f"SCL array {data.shape} does not match grid {(grid.height, grid.width)}")
    with rasterio.open(path, "w", **grid.profile(count=1, dtype="uint8", nodata=SCL_NO_DATA)) as dst:
        dst.write(data, 1)
        dst.set_band_description(1, "SCL (Sentinel-2 L2A scene classification)")
        dst.update_tags(classes=json.dumps({str(code): name for code, name in SCL_CLASSES.items()}),
                        label=label or "SCL")


@dataclass
class CloudMask:
    """SCL-derived validity for one observation on the shared analysis grid."""

    label: str
    grid: RasterGrid
    available: bool
    valid: np.ndarray
    """``[H, W]`` boolean: usable surface in this acquisition."""

    classes: np.ndarray | None = None
    """``[H, W]`` uint8 SCL codes on the analysis grid; ``None`` when SCL is absent."""

    report: dict = field(default_factory=dict)

    @property
    def masked(self) -> np.ndarray:
        return ~self.valid

    @property
    def class_array(self) -> np.ndarray:
        """``[H, W]`` uint8 SCL codes on the grid; NO_DATA when the observation has no SCL."""
        if self.classes is not None:
            return self.classes
        return np.full((self.grid.height, self.grid.width), SCL_NO_DATA, dtype="uint8")

    @property
    def masked_pixels(self) -> int:
        return int(self.masked.sum())

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "available": self.available,
            "pixels": int(self.valid.size),
            "masked_pixels": self.masked_pixels,
            "valid_fraction": round(float(self.valid.mean()), 6),
            "report": self.report,
        }


def mask_observation(observation, grid: RasterGrid,
                     valid_classes: tuple[int, ...] = SCL_VALID_CLASSES) -> CloudMask:
    """Build the cloud / invalid-pixel mask for one observation on ``grid``.

    ``observation`` is duck-typed on ``.aux[SCL_BAND].path`` so this module stays free of
    an import cycle with ``preprocessing``. When no SCL band is present - OSCD, LEVIR-CD
    and any non-Sentinel source - the mask is a no-op: every pixel is usable and the
    report says so instead of pretending masking happened.
    """
    label = getattr(observation, "label", "observation")
    everything_usable = np.ones((grid.height, grid.width), dtype=bool)

    aux = getattr(observation, "aux", None) or {}
    path = getattr(aux.get(SCL_BAND), "path", None)
    if path is None or not Path(path).is_file():
        return CloudMask(label=label, grid=grid, available=False, valid=everything_usable,
                         classes=None, report={
                             "method": MASK_METHOD,
                             "scl_band": None,
                             "available": False,
                             "valid_classes": list(valid_classes),
                             "note": "no SCL asset for this observation; no pixel is masked by "
                                     "cloud policy (band nodata and footprint coverage still apply)",
                         })

    classes = resample_scl_to_grid(path, grid)
    valid = scl_valid_mask(classes, valid_classes)
    report = scl_mask_report(classes, valid_classes)
    report.update({
        "scl_band": str(path),
        "available": True,
        "source": {
            "resolution_m": SCL_NATIVE_RESOLUTION_M,
            "resampling": SCL_RESAMPLING,
            "target_resolution_m": grid.resolution_m,
            "target_shape": [int(grid.height), int(grid.width)],
        },
    })
    return CloudMask(label=label, grid=grid, available=True, valid=valid, classes=classes,
                     report=report)


def pair_valid_mask(before: CloudMask, after: CloudMask) -> np.ndarray:
    """Pixels usable in *both* acquisitions - the only ones a comparison may use."""
    if before.valid.shape != after.valid.shape:
        raise ValueError(f"cloud masks must share a shape, got {before.valid.shape} "
                         f"and {after.valid.shape}")
    return before.valid & after.valid


def main(argv: list[str] | None = None) -> int:
    """Print the SCL masking report for the raw demo pair."""
    parser = argparse.ArgumentParser(description="TerraGuard AI - Sentinel-2 SCL cloud masking")
    parser.add_argument("--raw", type=Path,
                        default=Path(__file__).resolve().parents[3] / "data" / "raw" / "demo_area")
    parser.add_argument("--resolution", type=float, default=DEFAULT_RESOLUTION_M)
    parser.add_argument("--lenient", action="store_true",
                        help="also trust SCL class 7 (UNCLASSIFIED)")
    args = parser.parse_args(argv)

    from app.services.preprocessing import load_sentinel_pair

    valid_classes = SCL_VALID_CLASSES_LENIENT if args.lenient else SCL_VALID_CLASSES
    before, after = load_sentinel_pair(args.raw)
    grid = common_grid([*before.paths, *after.paths], resolution=args.resolution)

    print(f"SCL masking on grid {grid.width}x{grid.height} @ {grid.resolution_m} m ({grid.crs})")
    print(f"usable classes: {valid_classes}")
    shared = None
    for observation in (before, after):
        cloud = mask_observation(observation, grid, valid_classes)
        shared = cloud.valid if shared is None else shared & cloud.valid
        print(f"\n{cloud.label.upper()}")
        if not cloud.available:
            print("  no SCL asset - nothing masked")
            continue
        report = cloud.report
        print(f"  scl source : {report['scl_band']}")
        print(f"               {SCL_NATIVE_RESOLUTION_M} m -> {grid.resolution_m} m nearest neighbour")
        print(f"  usable     : {report['usable_fraction']:.4%} of grid pixels")
        print(f"  masked     : {report['masked_fraction']:.4%} ({report['masked_pixels']} pixels)")
        for reason, detail in report["mask_reasons"].items():
            print(f"    {reason:22s} {detail['pixels']:>7} px "
                  f"({detail['fraction_of_image']:.4%}) classes {detail['classes']}")
        print("  histogram  : " + ", ".join(
            f"{record['class']}={record['pixels']}" for record in report["histogram"]
            if record["class"] is not None))

    if shared is not None:
        print(f"\nusable in BOTH acquisitions: {shared.mean():.4%} "
              f"({int((~shared).sum())} pixels excluded from the comparison)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
