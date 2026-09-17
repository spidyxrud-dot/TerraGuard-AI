"""Make ``app.*`` and ``ml.*`` importable when tests run from the repository root."""

import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND = REPO_ROOT / "backend"
for entry in (BACKEND, REPO_ROOT):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

BAND_BASE = {"B02": 300, "B03": 400, "B04": 500, "B08": 3000}


def make_oscd_region(root: Path, region: str, size: int = 64, change: bool = True,
                     coarse_nir: bool = False) -> None:
    """Write one tiny OSCD-style region (per-band GeoTIFFs + optional cm.png)."""
    rows, cols = np.indices((size, size))
    coarse_rows, coarse_cols = np.indices((size // 2, size // 2))
    for date_index, date in enumerate(("imgs_1_rect", "imgs_2_rect")):
        directory = root / "images" / region / date
        offset = 0 if date_index == 0 else 100
        for band, base in BAND_BASE.items():
            if band == "B08" and coarse_nir:
                values = (1500 + offset + 5 * coarse_cols).astype("uint16")
                transform = from_origin(500_000.0, 2_000_000.0, 20.0, 20.0)
                shape = (size // 2, size // 2)
            else:
                values = (base + offset + 3 * rows + 7 * cols).astype("uint16")
                transform = from_origin(500_000.0, 2_000_000.0, 10.0, 10.0)
                shape = (size, size)
            directory.mkdir(parents=True, exist_ok=True)
            with rasterio.open(directory / f"{band}.tif", "w", driver="GTiff",
                               height=shape[0], width=shape[1], count=1, dtype="uint16",
                               crs="EPSG:32643", transform=transform) as dst:
                dst.write(values, 1)
    if change:
        mask = np.zeros((size, size), dtype="uint8")
        mask[size // 4:size // 2, size // 4:size // 2] = 255
        mask_path = root / "target" / region / "cm" / "cm.png"
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(mask_path, "w", driver="PNG", height=size, width=size, count=1,
                           dtype="uint8") as dst:
            dst.write(mask, 1)


def make_oscd_split(root: Path, train_regions: list[str], test_regions: list[str]) -> Path:
    (root / "train.txt").write_text("\n".join(train_regions) + "\n", encoding="utf-8")
    (root / "test.txt").write_text("\n".join(test_regions) + "\n", encoding="utf-8")
    return root


def make_levir_pair(root: Path, split: str, stem: str, size: int = 64,
                    change: bool = True) -> None:
    red, green, blue = 40, 90, 130
    before = np.zeros((3, size, size), dtype="uint8")
    after = np.zeros((3, size, size), dtype="uint8")
    before[0], before[1], before[2] = red, green, blue
    after[0], after[1], after[2] = red + 50, green, blue
    label = np.zeros((size, size), dtype="uint8")
    if change:
        label[size // 4:size // 2, size // 4:size // 2] = 255
    for name, array in (("A", before), ("B", after), ("label", label)):
        path = root / split / name / f"{stem}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        count = 3 if name != "label" else 1
        with rasterio.open(path, "w", driver="PNG", height=size, width=size, count=count,
                           dtype="uint8") as dst:
            dst.write(array[None] if array.ndim == 2 else array)


@pytest.fixture()
def oscd_root(tmp_path: Path) -> Path:
    """OSCD tree: 'alpha' (lowercase band name), 'beta' (20 m NIR), 'gamma' (no mask)."""
    root = tmp_path / "oscd"
    make_oscd_region(root, "alpha")
    (root / "images" / "alpha" / "imgs_1_rect" / "B03.tif").rename(
        root / "images" / "alpha" / "imgs_1_rect" / "b03.tif")
    make_oscd_region(root, "beta", coarse_nir=True)
    make_oscd_region(root, "gamma", change=False)
    return make_oscd_split(root, ["alpha", "beta"], ["gamma"])


@pytest.fixture()
def levir_root(tmp_path: Path) -> Path:
    """LEVIR-CD tree with official train/val/test folders."""
    root = tmp_path / "levir"
    for index in range(3):
        make_levir_pair(root, "train", f"train_{index:04d}", change=index != 1)
    for index in range(2):
        make_levir_pair(root, "val", f"val_{index:04d}", change=index == 0)
    for index in range(2):
        make_levir_pair(root, "test", f"test_{index:04d}", change=index == 1)
    return root


@pytest.fixture()
def train_oscd_root(tmp_path: Path) -> Path:
    """OSCD tree with enough regions for a location-level train/validation split."""
    root = tmp_path / "oscd_train"
    for region in ("r01", "r02", "r03", "r04"):
        make_oscd_region(root, region)
    for region in ("t01", "t02"):
        make_oscd_region(root, region)
    return make_oscd_split(root, ["r01", "r02", "r03", "r04"], ["t01", "t02"])
