"""Tests for the change-detection dataset loaders (Step 3.2).

The fixtures build miniature OSCD/LEVIR-CD trees with the same layout, formats and
traps as the real downloads: per-band GeoTIFFs (including a 20 m band that must be
warped onto the 10 m reference grid), lowercase band names, an unannotated region,
and A/B/label PNG pairs. They prove the loaders deliver the frozen sample contract
``before [C,H,W] / after [C,H,W] / mask [1,H,W]`` without leaking the two datasets'
differences into the model interface.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import rasterio
import torch
from rasterio.transform import from_origin
from torch.utils.data import DataLoader

from app.services import preprocessing as backend_preprocessing
from ml.change_detection.dataset import (
    BAND_ORDER,
    NORMALIZATION_DESCRIPTION,
    REFLECTANCE_RANGE,
    REFLECTANCE_SCALE,
    LevirDataset,
    OscdDataset,
    normalize_eight_bit,
    normalize_reflectance,
    oscd_split_regions,
)
from ml.change_detection.dataset import main as dataset_main

SIZE = 64
SIZE_COARSE = 32
ORIGIN = (500_000.0, 2_000_000.0)
CRS = "EPSG:32643"

BAND_DN = {"B02": 300, "B03": 420, "B04": 520, "B08": 2600}
DATE_OFFSET = {"before": 0, "after": 100}
CHANGE_ROWS = slice(24, 40)
CHANGE_COLS = slice(24, 40)
CHANGE_PIXELS = 16 * 16



def _dn(band: str, rows, cols, date: str) -> np.ndarray:
    texture = ((3 * rows + 7 * cols) % 900 + 100).astype("uint16")
    return (BAND_DN[band] + DATE_OFFSET[date] + texture).astype("uint16")


def _write_tif(path: Path, array: np.ndarray, resolution: float = 10.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    transform = from_origin(ORIGIN[0], ORIGIN[1], resolution, resolution)
    with rasterio.open(path, "w", driver="GTiff", height=array.shape[0],
                       width=array.shape[1], count=1, dtype=array.dtype.name,
                       crs=CRS, transform=transform) as dst:
        dst.write(array, 1)
    return path


def _write_png(path: Path, array: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    planes = array[None] if array.ndim == 2 else array
    with rasterio.open(path, "w", driver="PNG", height=planes.shape[1],
                       width=planes.shape[2], count=planes.shape[0], dtype="uint8") as dst:
        dst.write(planes)
    return path


def _change_mask(size: int = SIZE, filled: bool = True) -> np.ndarray:
    mask = np.zeros((size, size), dtype="uint8")
    if filled:
        mask[CHANGE_ROWS, CHANGE_COLS] = 255
    return mask


def _write_oscd_date(directory: Path, date: str, *, coarse_nir: bool = False) -> None:
    rows, cols = np.indices((SIZE, SIZE))
    for band in BAND_ORDER:
        if band == "B08" and coarse_nir:
            continue
        _write_tif(directory / f"{band}.tif", _dn(band, rows, cols, date))
    if coarse_nir:
        coarse = np.indices((SIZE_COARSE, SIZE_COARSE))[1]
        _write_tif(directory / "B08.tif",
                   (1000 + 10 * coarse).astype("uint16"), resolution=20.0)


def make_oscd(root: Path) -> Path:
    """Three regions: normal, coarse 20 m NIR band, and one without annotations."""
    images = root / "images"
    target = root / "target"

    _write_oscd_date(images / "alpha" / "imgs_1_rect", "before")
    _write_oscd_date(images / "alpha" / "imgs_2_rect", "after")
    # alpha ships one band with lowercase naming, as some OSCD zips do
    (images / "alpha" / "imgs_1_rect" / "B03.tif").rename(
        images / "alpha" / "imgs_1_rect" / "b03.tif")
    _write_png(target / "alpha" / "cm" / "cm.png", _change_mask())

    _write_oscd_date(images / "beta" / "imgs_1_rect", "before", coarse_nir=True)
    _write_oscd_date(images / "beta" / "imgs_2_rect", "after", coarse_nir=True)
    _write_png(target / "beta" / "cm" / "cm.png", _change_mask())

    _write_oscd_date(images / "gamma" / "imgs_1_rect", "before")
    _write_oscd_date(images / "gamma" / "imgs_2_rect", "after")
    # gamma has no change annotation at all (OSCD ships one such region)

    (root / "train.txt").write_text("alpha\nbeta\n", encoding="utf-8")
    (root / "test.txt").write_text("gamma\n", encoding="utf-8")
    return root


def make_levir(root: Path, *, pairs: int = 3, size: int = 64) -> Path:
    split = root / "train"
    label_boxes = {0: True, 1: False, 2: True}
    rgb = {pair: (10 + 30 * pair, 20 + 30 * pair, 30 + 30 * pair) for pair in range(pairs)}
    for pair in range(pairs):
        red, green, blue = rgb[pair]
        before = np.zeros((3, size, size), dtype="uint8")
        after = np.zeros((3, size, size), dtype="uint8")
        before[0], before[1], before[2] = red, green, blue          # R, G, B
        after[0], after[1], after[2] = red + 30, green, blue
        stem = f"train_{pair:04d}"
        _write_png(split / "A" / f"{stem}.png", before)
        _write_png(split / "B" / f"{stem}.png", after)
        _write_png(split / "label" / f"{stem}.png", _change_mask(size, label_boxes[pair]))
    return root


@pytest.fixture()
def oscd_root(tmp_path: Path) -> Path:
    return make_oscd(tmp_path / "oscd")


@pytest.fixture()
def levir_root(tmp_path: Path) -> Path:
    return make_levir(tmp_path / "levir")


# ------------------------------------------------------------------- normalization


def test_ml_constants_mirror_the_backend_contract() -> None:
    assert BAND_ORDER == backend_preprocessing.BAND_ORDER
    assert REFLECTANCE_SCALE == backend_preprocessing.REFLECTANCE_SCALE
    assert REFLECTANCE_RANGE == backend_preprocessing.REFLECTANCE_RANGE
    assert NORMALIZATION_DESCRIPTION == backend_preprocessing.NORMALIZATION_DESCRIPTION


def test_normalize_reflectance_is_fixed_scale_with_clip() -> None:
    values = np.array([[2000, 12000], [0, 65535]], dtype="uint16")
    normalized = normalize_reflectance(values)
    assert normalized.dtype == np.float32
    assert normalized[0, 0] == pytest.approx(0.2, abs=1e-6)
    assert normalized[0, 1] == 1.0, "DN above the scale clips to the range maximum"
    assert normalized[1, 0] == 0.0
    assert normalized[1, 1] == 1.0


def test_normalize_eight_bit_lands_in_the_same_range() -> None:
    values = np.array([[[10, 255], [0, 127]]], dtype="uint8")
    normalized = normalize_eight_bit(values)
    assert normalized.dtype == np.float32
    assert normalized.max() == pytest.approx(1.0, abs=1e-6)
    assert normalized[0, 1, 1] == pytest.approx(127 / 255, abs=1e-6)


# ------------------------------------------------------------------------- OSCD


def test_oscd_split_regions_reads_the_official_files(oscd_root: Path) -> None:
    split = oscd_split_regions(oscd_root)
    assert split == {"train": ["alpha", "beta"], "test": ["gamma"]}
    assert not set(split["train"]) & set(split["test"]), "location-level split must be disjoint"


def test_oscd_split_regions_rejects_leakage(tmp_path: Path) -> None:
    (tmp_path / "train.txt").write_text("alpha\nbeta\n", encoding="utf-8")
    (tmp_path / "test.txt").write_text("beta\ngamma\n", encoding="utf-8")
    with pytest.raises(ValueError, match="geographic leakage"):
        oscd_split_regions(tmp_path)


def test_oscd_split_regions_requires_both_files(tmp_path: Path) -> None:
    (tmp_path / "train.txt").write_text("alpha\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="test.txt"):
        oscd_split_regions(tmp_path)


def test_oscd_dataset_respects_the_official_split(oscd_root: Path) -> None:
    train = OscdDataset(oscd_root, split="train")
    test = OscdDataset(oscd_root, split="test")
    assert train.identifiers == ["alpha", "beta"]
    assert test.identifiers == ["gamma"]
    assert len(train) == 2 and len(test) == 1


def test_oscd_dataset_discovers_all_regions_without_split(oscd_root: Path) -> None:
    dataset = OscdDataset(oscd_root)
    assert dataset.identifiers == ["alpha", "beta", "gamma"]


def test_oscd_sample_contract(oscd_root: Path) -> None:
    sample = OscdDataset(oscd_root, split="train")[0]
    assert set(sample) == {"before", "after", "mask"}
    assert sample["before"].shape == (4, SIZE, SIZE)
    assert sample["after"].shape == (4, SIZE, SIZE)
    assert sample["mask"].shape == (1, SIZE, SIZE)
    assert sample["before"].dtype == torch.float32
    assert sample["mask"].dtype == torch.float32
    assert set(sample["mask"].unique().tolist()) <= {0.0, 1.0}
    for key in ("before", "after"):
        assert sample[key].min() >= REFLECTANCE_RANGE[0]
        assert sample[key].max() <= REFLECTANCE_RANGE[1]


def test_oscd_values_follow_the_dn_normalization(oscd_root: Path) -> None:
    sample = OscdDataset(oscd_root, split="train", regions=["alpha"])[0]
    rows, cols = 0, 0
    expected_before = (BAND_DN["B02"] + DATE_OFFSET["before"] + 100) / REFLECTANCE_SCALE
    expected_after = (BAND_DN["B02"] + DATE_OFFSET["after"] + 100) / REFLECTANCE_SCALE
    assert sample["before"][0, rows, cols].item() == pytest.approx(expected_before, abs=1e-6)
    assert sample["after"][0, rows, cols].item() == pytest.approx(expected_after, abs=1e-6)


def test_oscd_reads_the_ground_truth_mask(oscd_root: Path) -> None:
    sample = OscdDataset(oscd_root, split="train")[0]
    assert float(sample["mask"].mean()) == pytest.approx(CHANGE_PIXELS / (SIZE * SIZE), abs=1e-6)


def test_oscd_coarse_nir_band_is_warped_onto_the_reference_grid(oscd_root: Path) -> None:
    """beta ships B08 at 20 m: it must arrive aligned with the 10 m reference grid."""
    sample = OscdDataset(oscd_root, split="train", regions=["beta"])[0]
    nir = sample["before"][3]
    assert nir.shape == (SIZE, SIZE)
    for col, expected_dn in ((0, 1000.0), (10, 1047.5), (63, 1312.5)):
        assert nir[0, col].item() == pytest.approx(expected_dn / REFLECTANCE_SCALE, abs=5e-4)


def test_oscd_unannotated_region_gets_an_honest_zero_mask(oscd_root: Path) -> None:
    dataset = OscdDataset(oscd_root, split="test")
    assert dataset.region_report["gamma"]["annotated"] is False
    assert dataset[0]["mask"].sum().item() == 0.0


def test_oscd_missing_band_fails_with_the_expected_path(oscd_root: Path) -> None:
    (oscd_root / "images" / "beta" / "imgs_1_rect" / "B04.tif").unlink()
    with pytest.raises(FileNotFoundError, match="B04"):
        OscdDataset(oscd_root, split="train")


def test_oscd_split_listings_must_exist_on_disk(oscd_root: Path) -> None:
    (oscd_root / "images" / "beta").rename(oscd_root / "images" / "beta_moved")
    with pytest.raises(FileNotFoundError, match="beta"):
        OscdDataset(oscd_root, split="train")


def test_oscd_root_must_exist(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="OSCD root"):
        OscdDataset(tmp_path / "missing")


# ----------------------------------------------------------- cropping / augmenting


def test_patch_crop_is_deterministic_and_mask_aligned(oscd_root: Path) -> None:
    dataset = OscdDataset(oscd_root, split="train", patch_size=32)
    full = OscdDataset(oscd_root, split="train")

    first, again = dataset[0], dataset[0]
    assert torch.equal(first["before"], again["before"]), "same index must give same window"
    assert first["before"].shape == (4, 32, 32)

    rng = random.Random(f"{dataset.seed}/{0}")
    row, col = rng.randint(0, SIZE - 32), rng.randint(0, SIZE - 32)
    assert torch.equal(first["mask"][0], full[0]["mask"][0, row:row + 32, col:col + 32])
    assert torch.equal(first["before"][0], full[0]["before"][0, row:row + 32, col:col + 32])
    assert torch.equal(first["after"][0], full[0]["after"][0, row:row + 32, col:col + 32])


def test_flip_augmentation_is_deterministic_and_valid(oscd_root: Path) -> None:
    dataset = OscdDataset(oscd_root, split="train", augment=True)
    base = OscdDataset(oscd_root, split="train")
    first, again = dataset[0], dataset[0]
    assert torch.equal(first["before"], again["before"])

    original = base[0]["before"]
    flips = (original, torch.flip(original, [2]), torch.flip(original, [1]),
             torch.flip(original, [1, 2]))
    assert any(torch.equal(first["before"], candidate) for candidate in flips)


def test_invalid_patch_size_is_rejected(oscd_root: Path) -> None:
    with pytest.raises(ValueError, match="patch_size"):
        OscdDataset(oscd_root, split="train", patch_size=0)


# ----------------------------------------------------------------------- LEVIR-CD


def test_levir_contract_and_rgb_channel_mapping(levir_root: Path) -> None:
    dataset = LevirDataset(levir_root, split="train")
    assert len(dataset) == 3
    sample = dataset[0]
    assert sample["before"].shape == (4, 64, 64)
    assert sample["after"].shape == (4, 64, 64)
    assert sample["mask"].shape == (1, 64, 64)

    red, green, blue = 10, 20, 30
    assert float(sample["before"][0].max()) == pytest.approx(blue / 255, abs=1e-6)
    assert float(sample["before"][1].max()) == pytest.approx(green / 255, abs=1e-6)
    assert float(sample["before"][2].max()) == pytest.approx(red / 255, abs=1e-6)
    assert float(sample["before"][3].abs().max()) == 0.0, "NIR is an honest zero filler"
    assert float(sample["after"][3].abs().max()) == 0.0
    assert dataset.channel_availability == (True, True, True, False)
    assert "no NIR" in dataset.normalization


def test_levir_change_survives_into_the_mask(levir_root: Path) -> None:
    dataset = LevirDataset(levir_root, split="train")
    assert float(dataset[0]["mask"].mean()) == pytest.approx(CHANGE_PIXELS / (SIZE * SIZE), 1e-6)
    assert float(dataset[1]["mask"].sum()) == 0.0, "empty label must stay all-zero"
    assert float(dataset[2]["mask"].mean()) == pytest.approx(CHANGE_PIXELS / (SIZE * SIZE), 1e-6)


def test_levir_before_and_after_differ(levir_root: Path) -> None:
    sample = LevirDataset(levir_root, split="train")[0]
    assert not torch.equal(sample["before"][2], sample["after"][2]), \
        "red channel changes between dates - a swap bug would be silent otherwise"


def test_levir_three_channel_option(levir_root: Path) -> None:
    dataset = LevirDataset(levir_root, split="train", add_nir_zero_channel=False)
    sample = dataset[0]
    assert sample["before"].shape == (3, 64, 64)
    assert dataset.channel_availability == (True, True, True)


def test_levir_rejects_unpaired_files(levir_root: Path) -> None:
    _write_png(levir_root / "train" / "A" / "train_9999.png",
               np.zeros((3, 64, 64), dtype="uint8"))
    with pytest.raises(FileNotFoundError, match="train_9999"):
        LevirDataset(levir_root, split="train")


def test_levir_requires_labels(levir_root: Path) -> None:
    (levir_root / "train" / "label" / "train_0001.png").unlink()
    with pytest.raises(FileNotFoundError, match="train_0001"):
        LevirDataset(levir_root, split="train")


def test_levir_unknown_split_fails_with_layout_hint(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="LEVIR-CD split 'val'"):
        LevirDataset(tmp_path, split="val")


# ------------------------------------------------------------ DataLoader + CLI


def test_dataloader_batches_both_datasets(oscd_root: Path, levir_root: Path) -> None:
    oscd_batch = next(iter(DataLoader(OscdDataset(oscd_root, split="train"), batch_size=2)))
    assert oscd_batch["before"].shape == (2, 4, SIZE, SIZE)
    assert oscd_batch["mask"].shape == (2, 1, SIZE, SIZE)

    levir_batch = next(iter(DataLoader(LevirDataset(levir_root, split="train"), batch_size=2)))
    assert levir_batch["before"].shape == (2, 4, 64, 64)


def test_cli_summarizes_datasets(oscd_root: Path, levir_root: Path,
                                 capsys: pytest.CaptureFixture) -> None:
    assert dataset_main(["--oscd-root", str(oscd_root), "--levir-root", str(levir_root),
                         "--split", "train"]) == 0
    out = capsys.readouterr().out
    assert "OscdDataset" in out and "LevirDataset" in out
    assert "channel_availability" in out
    assert "no NIR" in out


def test_cli_without_roots_prints_expected_layouts(capsys: pytest.CaptureFixture) -> None:
    assert dataset_main([]) == 0
    out = capsys.readouterr().out
    assert "imgs_1_rect" in out and "label" in out and "train.txt" in out
