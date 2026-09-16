"""Dataset loaders for supervised change detection (Step 3.2).

Every loader in this module returns the same sample interface, so the Siamese U-Net
never needs to know where a pair came from:

    {
        "before": torch.Tensor  # [C, H, W] float32, normalized to [0, 1]
        "after":  torch.Tensor  # [C, H, W] float32, normalized to [0, 1]
        "mask":   torch.Tensor  # [1, H, W] float32, 1.0 = change (ground truth)
    }

Datasets
--------
OSCD (primary) - Onera Satellite Change Detection, real Sentinel-2 L1C/L2A tiles with
per-band GeoTIFFs and human-annotated change masks. With ``B02/B03/B04/B08`` requested
it satisfies the 4-band TerraGuard contract natively. Regions are split by the official
``train.txt`` / ``test.txt`` files, which keeps every pixel of a geographic scene on one
side of the split (no patch-level leakage).

LEVIR-CD (secondary) - 256x256 RGB pairs of Google-Earth imagery with building-change
labels. It has **no multispectral bands**: the loader maps RGB onto the blue/green/red
contract channels and zero-fills the NIR channel. Zeros are an explicit absence marker,
documented in ``channel_names`` - NIR is never synthesized from RGB, and a model trained
on LEVIR data must be interpreted as an RGB (structural-change) specialist.

Both loaders normalize to ``[0, 1]`` with fixed, dataset-appropriate scales (Sentinel-2
DN / 10000 mirrors ``backend/app/services/preprocessing.py`` exactly; 8-bit imagery /
255). No per-sample statistics are fitted, so before/after stay directly comparable.

Cropping / augmentation is deterministic per sample index (seeded PRNG), so a given
index always yields the same window - reproducible training and leak-free shuffling.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from torch.utils.data import Dataset

BAND_ORDER: tuple[str, ...] = ("B02", "B03", "B04", "B08")
"""TerraGuard channel order: blue, green, red, nir (matches the backend contract)."""

REFLECTANCE_SCALE = 10_000.0
"""Sentinel-2 DN -> reflectance scale (identical to the Phase 2 pipeline)."""

REFLECTANCE_RANGE: tuple[float, float] = (0.0, 1.0)

NORMALIZATION_DESCRIPTION = "surface_reflectance = clip(DN / 10000, 0, 1)"

EIGHT_BIT_SCALE = 255.0
"""LEVIR-CD ships 8-bit RGB PNGs; /255 lands them in the same [0, 1] range."""

MASK_THRESHOLD = 127
"""Label pixel values above this count as change (OSCD/LEVIR store 0 / 255)."""


def normalize_reflectance(dn: np.ndarray) -> np.ndarray:
    """Sentinel-2 digital numbers -> float32 surface reflectance, clipped to [0, 1].

    Fixed scale, identical to ``backend/app/services/preprocessing.normalize_bands``:
    no per-image statistics, so two dates (and two datasets) stay comparable.
    """
    values = np.asarray(dn, dtype="float32") / REFLECTANCE_SCALE
    return np.clip(values, REFLECTANCE_RANGE[0], REFLECTANCE_RANGE[1]).astype("float32")


def normalize_eight_bit(rgb: np.ndarray) -> np.ndarray:
    """8-bit imagery -> float32 in [0, 1], per channel, same clip policy."""
    values = np.asarray(rgb, dtype="float32") / EIGHT_BIT_SCALE
    return np.clip(values, REFLECTANCE_RANGE[0], REFLECTANCE_RANGE[1]).astype("float32")


def read_change_mask(path: str | Path) -> np.ndarray:
    """Read a 0/255 change annotation as a ``[H, W]`` uint8 mask of 0/1."""
    with rasterio.open(path) as src:
        values = np.asarray(src.read(1))
    return (values > MASK_THRESHOLD).astype("uint8")


def _find_file(directory: Path, filename: str) -> Path:
    """Case-insensitive file lookup (OSCD zips differ in band-name casing)."""
    exact = directory / filename
    if exact.is_file():
        return exact
    lowered = filename.lower()
    for candidate in sorted(directory.iterdir()):
        if candidate.is_file() and candidate.name.lower() == lowered:
            return candidate
    raise FileNotFoundError(f"expected {filename} in {directory}")


def _read_band_on_grid(path: Path, crs, transform, width: int, height: int,
                       resampling: Resampling = Resampling.bilinear) -> np.ndarray:
    """Read one band resampled onto an explicit reference grid.

    OSCD ships bands at their native Sentinel-2 resolutions; a Siamese pair needs every
    channel on one grid. ``WarpedVRT`` does the alignment in-memory without touching the
    source files. Bands already on the grid are read directly.
    """
    with rasterio.open(path) as src:
        if (src.crs == crs and src.transform == transform
                and src.width == width and src.height == height):
            return src.read(1)
        with WarpedVRT(src, crs=crs, transform=transform,
                       width=width, height=height, resampling=resampling) as vrt:
            return vrt.read(1)


class ChangePairDataset(Dataset):
    """Shared behaviour for all change-detection loaders.

    Subclasses fill ``self._pairs`` with ``(identifier, before, after, mask)`` numpy
    arrays (``before``/``after`` already normalized ``[C, H, W]`` float32, ``mask``
    ``[H, W]`` uint8 0/1) and implement ``__len__``.

    Cropping and flips are deterministic per sample index: the PRNG is re-seeded from
    ``(seed, index)`` on every access, so the same index always yields the same window
    regardless of DataLoader worker count or epoch - reproducibility by construction.
    """

    #: names shown in summaries / stored in model metadata, in channel order
    channel_names: tuple[str, ...] = BAND_ORDER
    #: per-channel availability flags (False = filler channel, e.g. LEVIR's absent NIR)
    channel_availability: tuple[bool, ...] = (True, True, True, True)
    #: human-readable normalization description for model metadata
    normalization: str = NORMALIZATION_DESCRIPTION
    has_georeference: bool = False

    def __init__(self, patch_size: int | None = None, augment: bool = False,
                 seed: int = 0) -> None:
        super().__init__()
        if patch_size is not None and patch_size <= 0:
            raise ValueError(f"patch_size must be a positive int or None, got {patch_size}")
        self.patch_size = patch_size
        self.augment = augment
        self.seed = seed
        self._pairs: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []

    # -- required Dataset interface ------------------------------------------
    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, index: int) -> dict:
        identifier, before, after, mask = self._pairs[index]
        channels, height, width = before.shape
        if self.patch_size is not None and (height, width) != (self.patch_size, self.patch_size):
            rng = random.Random(f"{self.seed}/{index}")
            row = rng.randint(0, height - self.patch_size)
            col = rng.randint(0, width - self.patch_size)
            window = np.s_[row:row + self.patch_size, col:col + self.patch_size]
            before = before[(slice(None), *window)]
            after = after[(slice(None), *window)]
            mask = mask[window]
        if self.augment:
            rng = random.Random(f"{self.seed}/{index}/augment")
            if rng.random() < 0.5:
                before, after = before[:, :, ::-1], after[:, :, ::-1]
            if rng.random() < 0.5:
                before, after = before[:, ::-1, :], after[:, ::-1, :]
        return {
            "before": torch.from_numpy(np.ascontiguousarray(before, dtype="float32")),
            "after": torch.from_numpy(np.ascontiguousarray(after, dtype="float32")),
            "mask": torch.from_numpy(
                np.ascontiguousarray(mask, dtype="float32")[np.newaxis]),
        }

    # -- helpers --------------------------------------------------------------
    @property
    def identifiers(self) -> list[str]:
        return [identifier for identifier, *_ in self._pairs]

    def _validate_pair(self, identifier: str, before: np.ndarray, after: np.ndarray,
                       mask: np.ndarray) -> None:
        if before.shape != after.shape:
            raise ValueError(f"{identifier}: before {before.shape} and after {after.shape} "
                             "must share shape - the Siamese comparison is pixelwise")
        if mask.shape != before.shape[1:]:
            raise ValueError(f"{identifier}: mask {mask.shape} does not match the image "
                             f"grid {before.shape[1:]}")
        for name, array in (("before", before), ("after", after)):
            if not np.isfinite(array).all():
                raise ValueError(f"{identifier}: {name} contains non-finite pixels")
            if array.min() < REFLECTANCE_RANGE[0] or array.max() > REFLECTANCE_RANGE[1]:
                raise ValueError(f"{identifier}: {name} outside "
                                 f"{list(REFLECTANCE_RANGE)} (max {array.max():.4f})")

    def _register(self, identifier: str, before: np.ndarray, after: np.ndarray,
                  mask: np.ndarray) -> None:
        self._validate_pair(identifier, before, after, mask)
        self._pairs.append((identifier, before, after, mask))

    def change_fractions(self) -> dict[str, float]:
        """Ground-truth change fraction per sample, for dataset-level reporting."""
        return {identifier: float(mask.mean())
                for identifier, _, _, mask in self._pairs}

    def summary(self) -> dict:
        """JSON-serializable dataset facts (CLI / experiment logs)."""
        fractions = self.change_fractions()
        first = self._pairs[0] if self._pairs else None
        return {
            "dataset": type(self).__name__,
            "samples": len(self._pairs),
            "identifiers": self.identifiers,
            "channels": list(self.channel_names),
            "channel_availability": list(self.channel_availability),
            "normalization": self.normalization,
            "patch_size": self.patch_size,
            "augment": self.augment,
            "shape": None if first is None else [int(first[1].shape[0]),
                                                 int(first[1].shape[1]),
                                                 int(first[1].shape[2])],
            "change_fraction": {
                "mean": round(float(np.mean(list(fractions.values()))) if fractions else 0.0, 6),
                "min": round(float(min(fractions.values())) if fractions else 0.0, 6),
                "max": round(float(max(fractions.values())) if fractions else 0.0, 6),
            },
            "change_fractions": {key: round(value, 6) for key, value in fractions.items()},
        }


OSCD_LAYOUT_HINT = """expected OSCD layout (official 'Onera Satellite Change Detection'
train/test zips):
    <root>/images/<region>/imgs_1_rect/{B02,B03,B04,B08,...}.tif   (before)
    <root>/images/<region>/imgs_2_rect/...                          (after)
    <root>/target/<region>/cm/cm.png                                (change mask)
    <root>/train.txt, <root>/test.txt                               (region split)
'imgs_1'/'imgs_2' and masks beside the images are also recognized."""

OSCD_IMAGE_DIRS: tuple[tuple[str, str], ...] = (("imgs_1_rect", "imgs_2_rect"),
                                                ("imgs_1", "imgs_2"))
OSCD_MASK_LOCATIONS: tuple[str, ...] = (
    "target/{region}/cm/cm.png", "{region}/cm/cm.png", "{region}/target/cm/cm.png")


def oscd_split_regions(root: str | Path) -> dict[str, list[str]]:
    """Region names from the official ``train.txt`` / ``test.txt`` files.

    These files are the dataset's location-level split: every pixel of a scene stays on
    one side, so patch-level train/validation leakage is impossible by construction.
    """
    root = Path(root)
    split: dict[str, list[str]] = {}
    for name in ("train", "test"):
        path = root / f"{name}.txt"
        if not path.is_file():
            raise FileNotFoundError(f"{name}.txt not found in {root}\n{OSCD_LAYOUT_HINT}")
        regions = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
                   if line.strip() and not line.strip().startswith("#")]
        if not regions:
            raise ValueError(f"{path} lists no regions")
        split[name] = regions
    overlap = sorted(set(split["train"]) & set(split["test"]))
    if overlap:
        raise ValueError(f"train.txt and test.txt both list regions: {overlap} - refusing "
                         "a split with geographic leakage")
    return split


class OscdDataset(ChangePairDataset):
    """OSCD pairs as ``{before, after, mask}`` on the TerraGuard 4-band contract.

    One sample = one annotated region (full tile, or a deterministic patch window when
    ``patch_size`` is set). Bands are read per date and aligned onto the first requested
    band's grid (10 m), so ``before[:, y, x]`` and ``after[:, y, x]`` always describe the
    same point. Regions without a change annotation (OSCD ships one) get an all-zero
    mask and ``annotated=False`` in :attr:`region_report` - honest zeros, not missing
    data smuggled past the loader.
    """

    has_georeference = True

    def __init__(self, root: str | Path, split: str | None = None,
                 regions: list[str] | None = None, bands: tuple[str, ...] = BAND_ORDER,
                 patch_size: int | None = None, augment: bool = False,
                 seed: int = 0) -> None:
        super().__init__(patch_size=patch_size, augment=augment, seed=seed)
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"OSCD root not found: {self.root}\n{OSCD_LAYOUT_HINT}")
        unknown = [band for band in bands if band not in BAND_ORDER]
        if unknown:
            raise ValueError(f"bands {unknown} are outside the TerraGuard contract "
                             f"{list(BAND_ORDER)}")
        self.bands = tuple(bands)
        self.channel_names = self.bands
        self.channel_availability = tuple(band in self.bands for band in BAND_ORDER)
        self.split = split

        images_dir = self.root / "images" if (self.root / "images").is_dir() else self.root
        wanted = self._resolve_regions(images_dir, split, regions)
        if not wanted:
            raise FileNotFoundError(f"no OSCD regions found under {images_dir}\n"
                                    f"{OSCD_LAYOUT_HINT}")

        self.region_report: dict[str, dict] = {}
        for region in wanted:
            self._register(*self._load_region(images_dir, region))

    # -- discovery -------------------------------------------------------------
    def _resolve_regions(self, images_dir: Path, split: str | None,
                         regions: list[str] | None) -> list[str]:
        if regions is not None:
            return regions
        if split is None:
            return sorted({entry.name for entry in images_dir.iterdir() if entry.is_dir()})
        known = oscd_split_regions(self.root).get(split)
        if known is None:
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")
        missing = [region for region in known
                   if not self._region_dir(images_dir, region).is_dir()]
        if missing:
            raise FileNotFoundError(f"regions listed in {split}.txt but absent on disk: "
                                    f"{missing}\n{OSCD_LAYOUT_HINT}")
        return known

    @staticmethod
    def _region_dir(images_dir: Path, region: str) -> Path:
        exact = images_dir / region
        if exact.is_dir():
            return exact
        for entry in sorted(images_dir.iterdir()):
            if entry.is_dir() and entry.name.lower() == region.lower():
                return entry
        return exact

    @staticmethod
    def _date_dir(region_dir: Path, index: int) -> Path:
        for first, second in OSCD_IMAGE_DIRS:
            candidate = region_dir / (first if index == 1 else second)
            if candidate.is_dir():
                return candidate
        raise FileNotFoundError(
            f"no image folder for date {index} in {region_dir}\n{OSCD_LAYOUT_HINT}")

    def _mask_path(self, region: str, region_dir: Path) -> Path | None:
        for template in OSCD_MASK_LOCATIONS:
            candidate = self.root / template.format(region=region)
            if candidate.is_file():
                return candidate
        return region_dir / "cm" / "cm.png" if (region_dir / "cm" / "cm.png").is_file() else None

    # -- loading ---------------------------------------------------------------
    def _load_region(self, images_dir: Path, region: str) -> tuple[str, np.ndarray,
                                                                    np.ndarray, np.ndarray]:
        region_dir = self._region_dir(images_dir, region)
        before = self._read_date(self._date_dir(region_dir, 1), f"{region}:before")
        after = self._read_date(self._date_dir(region_dir, 2), f"{region}:after")

        mask_path = self._mask_path(region, region_dir)
        if mask_path is None:
            mask = np.zeros(before.shape[1:], dtype="uint8")
            self.region_report[region] = {"annotated": False, "mask": None}
        else:
            mask = read_change_mask(mask_path)
            self.region_report[region] = {"annotated": True, "mask": str(mask_path)}
        return region, before, after, mask

    def _read_date(self, date_dir: Path, label: str) -> np.ndarray:
        reference_path = _find_file(date_dir, f"{self.bands[0]}.tif")
        with rasterio.open(reference_path) as src:
            grid = (src.crs, src.transform, src.width, src.height)
        channels = [normalize_reflectance(
            _read_band_on_grid(_find_file(date_dir, f"{band}.tif"), *grid))
            for band in self.bands]
        stack = np.stack(channels, axis=0).astype("float32")
        if stack.shape[1:] != (grid[3], grid[2]):
            raise ValueError(f"{label}: bands did not align onto the {grid[2]}x{grid[3]} "
                             f"reference grid, got {stack.shape[1:]}")
        return stack


LEVIR_LAYOUT_HINT = """expected LEVIR-CD layout (official zip):
    <root>/<split>/A/*.png      before image, 256x256 RGB, 8-bit
    <root>/<split>/B/*.png      after image, same stems as A/
    <root>/<split>/label/*.png  binary change label, same stems as A/
with <split> one of 'train', 'val', 'test' (LEVIR-CD+ layouts with the split folders
directly under <root> are recognized)."""


class LevirDataset(ChangePairDataset):
    """LEVIR-CD pairs on the TerraGuard channel order - with one honest caveat.

    LEVIR-CD is 8-bit RGB aerial/Google-Earth imagery: it has blue/green/red but **no
    NIR**. The loader maps R->B04, G->B03, B->B02 and, by default, zero-fills the B08
    channel. ``channel_availability`` records that fill: a zero NIR channel is an
    explicit absence marker, not data. Train models on it for structural (RGB) change;
    interpret them accordingly - or set ``add_nir_zero_channel=False`` for a 3-channel
    contract, in which case ``in_channels`` must match at model build time.
    """

    def __init__(self, root: str | Path, split: str = "train",
                 patch_size: int | None = None, augment: bool = False, seed: int = 0,
                 add_nir_zero_channel: bool = True) -> None:
        super().__init__(patch_size=patch_size, augment=augment, seed=seed)
        self.root = Path(root)
        self.split = split
        base = self.root / split
        images_a, images_b = base / "A", base / "B"
        labels = base / "label"
        if not images_a.is_dir() or not images_b.is_dir():
            raise FileNotFoundError(f"LEVIR-CD split '{split}' not found under {self.root}"
                                    f"\n{LEVIR_LAYOUT_HINT}")
        if not labels.is_dir():
            raise FileNotFoundError(f"no label folder at {labels}\n{LEVIR_LAYOUT_HINT}")

        self.add_nir_zero_channel = add_nir_zero_channel
        self.channel_names = BAND_ORDER if add_nir_zero_channel else ("B02", "B03", "B04")
        self.channel_availability = (True, True, True, False) if add_nir_zero_channel \
            else (True, True, True)
        self.normalization = "rgb_8bit / 255 clipped to [0, 1] (no NIR band in LEVIR-CD)"
        self.has_georeference = False

        stems = self._paired_stems(images_a, images_b, labels)
        for stem in stems:
            before = self._read_rgb(images_a / f"{stem}.png", stem)
            after = self._read_rgb(images_b / f"{stem}.png", stem)
            mask = read_change_mask(labels / f"{stem}.png")
            if add_nir_zero_channel:
                zero = np.zeros_like(before[:1])
                before = np.concatenate([before, zero], axis=0)
                after = np.concatenate([after, zero], axis=0)
            self._register(f"{split}/{stem}", before, after, mask)

    @staticmethod
    def _paired_stems(images_a: Path, images_b: Path, labels: Path) -> list[str]:
        def stems(directory: Path) -> set[str]:
            return {path.stem for path in directory.glob("*.png")}
        common = sorted(stems(images_a) & stems(images_b) & stems(labels))
        if not common:
            raise FileNotFoundError(f"A/, B/ and label/ share no matching stems under "
                                    f"{images_a.parent}\n{LEVIR_LAYOUT_HINT}")
        for name, directory in (("A", images_a), ("B", images_b), ("label", labels)):
            orphans = stems(directory) - set(common)
            if orphans:
                raise FileNotFoundError(f"{name}/ has {len(orphans)} file(s) without a "
                                        f"matching pair/label: {sorted(orphans)[:3]}...")
        return common

    def _read_rgb(self, path: Path, stem: str) -> np.ndarray:
        with rasterio.open(path) as src:
            if src.count < 3:
                raise ValueError(f"{stem}: {path} has {src.count} band(s), expected RGB")
            rgb = src.read([1, 2, 3])  # rasterio band order == R, G, B for LEVIR PNGs
        channels = np.stack([
            normalize_eight_bit(rgb[2]),  # B02 slot: blue
            normalize_eight_bit(rgb[1]),  # B03 slot: green
            normalize_eight_bit(rgb[0]),  # B04 slot: red
        ], axis=0).astype("float32")
        return channels


def main(argv: list[str] | None = None) -> int:
    """Summarize available change-detection datasets (no data -> prints layouts)."""
    parser = argparse.ArgumentParser(description="TerraGuard AI - change-detection datasets")
    parser.add_argument("--oscd-root", type=Path, default=None,
                        help="unzipped OSCD root (images/, target/, train.txt, test.txt)")
    parser.add_argument("--levir-root", type=Path, default=None,
                        help="unzipped LEVIR-CD root (train/ val/ test/ with A, B, label)")
    parser.add_argument("--split", default="train")
    parser.add_argument("--patch-size", type=int, default=None)
    args = parser.parse_args(argv)

    if args.oscd_root is None and args.levir_root is None:
        print("No dataset roots given. Point --oscd-root / --levir-root at an unzipped "
              "download to inspect it.\n")
        print("OSCD (primary, Sentinel-2):")
        print(OSCD_LAYOUT_HINT)
        print("\nLEVIR-CD (secondary, RGB):")
        print(LEVIR_LAYOUT_HINT)
        return 0

    reports: list[dict] = []
    if args.oscd_root is not None:
        dataset = OscdDataset(args.oscd_root, split=args.split, patch_size=args.patch_size)
        reports.append(dataset.summary())
    if args.levir_root is not None:
        dataset = LevirDataset(args.levir_root, split=args.split, patch_size=args.patch_size)
        reports.append(dataset.summary())

    print(json.dumps({"datasets": reports}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

