"""Tests for Siamese U-Net inference pipeline (Step 3.6).

Covers:
- predict_pair output shapes, datatypes, and validity mask masking
- Threshold sensitivity and NaN input handling
- GeoTIFF writer preserving CRS, transform, dimensions, and nodata tags
- End-to-end inference on mock processed pairs and real Sentinel-2 pair if available
- CLI execution and error handling
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
import torch
from rasterio.transform import from_origin

from ml.change_detection.inference import (
    InferenceResult,
    main,
    predict_pair,
    run_inference,
    write_geotiff,
)
from ml.change_detection.model import SiameseUNet, SiameseUNetConfig
from ml.change_detection.train import save_checkpoint


@pytest.fixture()
def small_model() -> SiameseUNet:
    return SiameseUNet(SiameseUNetConfig(in_channels=4, base_channels=4, depth=2))


@pytest.fixture()
def dummy_checkpoint(tmp_path: Path, small_model: SiameseUNet) -> Path:
    ckpt_path = tmp_path / "models" / "siamese_unet.pth"
    save_checkpoint(ckpt_path, small_model, {"test_fixture": True})
    return ckpt_path


@pytest.fixture()
def mock_processed_dir(tmp_path: Path) -> Path:
    """Create a mock processed directory matching the backend preprocessing contract."""
    proc_dir = tmp_path / "processed" / "demo_area"
    proc_dir.mkdir(parents=True, exist_ok=True)

    height, width = 64, 64
    transform = from_origin(500_000.0, 2_000_000.0, 10.0, 10.0)
    crs = "EPSG:32643"

    before_arr = np.random.uniform(0.05, 0.4, (4, height, width)).astype("float32")
    after_arr = np.random.uniform(0.05, 0.4, (4, height, width)).astype("float32")

    # Introduce a simulated changed region in 'after'
    after_arr[:, 20:40, 20:40] += 0.2

    # Write GeoTIFFs
    prof = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 4,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "nodata": 0.0,
    }
    with rasterio.open(proc_dir / "before.tif", "w", **prof) as dst:
        dst.write(before_arr)
    with rasterio.open(proc_dir / "after.tif", "w", **prof) as dst:
        dst.write(after_arr)

    # Write NPY arrays
    np.save(proc_dir / "before.npy", before_arr)
    np.save(proc_dir / "after.npy", after_arr)

    # Valid mask (bottom right 10x10 is invalid/cloud)
    valid_mask = np.ones((height, width), dtype=bool)
    valid_mask[50:60, 50:60] = False
    np.save(proc_dir / "valid_mask.npy", valid_mask)

    return proc_dir


# --------------------------------------------------------------------- core prediction tests


def test_predict_pair_dimensions_and_types(small_model: SiameseUNet) -> None:
    before = np.random.rand(4, 48, 48).astype("float32")
    after = np.random.rand(4, 48, 48).astype("float32")
    valid_mask = np.ones((48, 48), dtype=bool)

    prob, mask, valid = predict_pair(small_model, before, after, valid_mask=valid_mask, threshold=0.5)

    assert prob.shape == (48, 48)
    assert prob.dtype == np.float32
    assert mask.shape == (48, 48)
    assert mask.dtype == np.uint8
    assert valid.shape == (48, 48)
    assert valid.dtype == bool
    assert set(np.unique(mask)).issubset({0, 1, 255})


def test_predict_pair_threshold_sensitivity(small_model: SiameseUNet) -> None:
    before = np.random.rand(4, 32, 32).astype("float32")
    after = np.random.rand(4, 32, 32).astype("float32")

    _, mask_low, _ = predict_pair(small_model, before, after, threshold=0.1)
    _, mask_high, _ = predict_pair(small_model, before, after, threshold=0.9)

    changed_low = (mask_low == 1).sum()
    changed_high = (mask_high == 1).sum()
    assert changed_low >= changed_high


def test_predict_pair_masks_invalid_pixels(small_model: SiameseUNet) -> None:
    before = np.random.rand(4, 32, 32).astype("float32")
    after = np.random.rand(4, 32, 32).astype("float32")
    valid_mask = np.ones((32, 32), dtype=bool)
    valid_mask[10:20, 10:20] = False  # masked region

    prob, mask, valid = predict_pair(small_model, before, after, valid_mask=valid_mask)

    assert np.isnan(prob[10:20, 10:20]).all()
    assert (mask[10:20, 10:20] == 255).all()
    assert not valid[10:20, 10:20].any()


def test_predict_pair_handles_nan_inputs(small_model: SiameseUNet) -> None:
    before = np.random.rand(4, 32, 32).astype("float32")
    after = np.random.rand(4, 32, 32).astype("float32")
    before[:, 0:5, 0:5] = np.nan

    prob, mask, valid = predict_pair(small_model, before, after)

    assert np.isnan(prob[0:5, 0:5]).all()
    assert (mask[0:5, 0:5] == 255).all()
    assert not valid[0:5, 0:5].any()


# ------------------------------------------------------------------- GeoTIFF writer tests


def test_write_geotiff_preserves_crs_and_transform(tmp_path: Path) -> None:
    out_tif = tmp_path / "test.tif"
    data = np.random.rand(32, 32).astype("float32")
    crs = "EPSG:32643"
    transform = from_origin(500_000.0, 2_000_000.0, 10.0, 10.0)

    write_geotiff(out_tif, data, crs=crs, transform=transform, nodata=-9999.0, dtype="float32",
                  description="Test Raster")

    assert out_tif.is_file()
    with rasterio.open(out_tif) as src:
        assert src.crs.to_string() == "EPSG:32643"
        assert src.transform[0] == 10.0
        assert src.transform[4] == -10.0
        assert src.shape == (32, 32)
        assert src.nodata == -9999.0
        assert src.tags().get("description") == "Test Raster"


# --------------------------------------------------------- end-to-end inference pipeline


def test_run_inference_missing_checkpoint(tmp_path: Path, mock_processed_dir: Path) -> None:
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        run_inference(checkpoint_path=tmp_path / "missing.pth", processed_dir=mock_processed_dir)


def test_run_inference_missing_processed_dir(dummy_checkpoint: Path, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Processed data directory not found"):
        run_inference(checkpoint_path=dummy_checkpoint, processed_dir=tmp_path / "missing_proc")


def test_run_inference_end_to_end(dummy_checkpoint: Path, mock_processed_dir: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "predictions"
    res = run_inference(
        checkpoint_path=dummy_checkpoint,
        processed_dir=mock_processed_dir,
        output_dir=out_dir,
        threshold=0.5,
        device="cpu",
    )

    assert isinstance(res, InferenceResult)
    assert res.shape == (64, 64)
    assert res.crs == "EPSG:32643"
    assert (out_dir / "change_probability.tif").is_file()
    assert (out_dir / "change_mask.tif").is_file()
    assert (out_dir / "inference_summary.json").is_file()

    summary = json.loads((out_dir / "inference_summary.json").read_text(encoding="utf-8"))
    assert summary["artifact"] == "terraguard.ai/inference"
    assert summary["statistics"]["valid_pixels"] == 64 * 64 - 100
    assert summary["statistics"]["masked_pixels"] == 100


def test_inference_cli_main(dummy_checkpoint: Path, mock_processed_dir: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "cli_predictions"
    exit_code = main([
        "--checkpoint", str(dummy_checkpoint),
        "--processed-dir", str(mock_processed_dir),
        "--output-dir", str(out_dir),
        "--threshold", "0.5",
        "--device", "cpu",
    ])
    assert exit_code == 0
    assert (out_dir / "change_probability.tif").is_file()


def test_inference_on_real_pune_pair_if_present(dummy_checkpoint: Path, tmp_path: Path) -> None:
    """Exercise inference on the actual 1025x1025 Pune Sentinel-2 demo pair."""
    pune_processed = Path("data/processed/demo_area")
    if not (pune_processed / "before.tif").is_file():
        pytest.skip("Pune demo pair not fetched or processed on disk")

    out_dir = tmp_path / "pune_predictions"
    res = run_inference(
        checkpoint_path=dummy_checkpoint,
        processed_dir=pune_processed,
        output_dir=out_dir,
        threshold=0.5,
        device="cpu",
    )

    assert res.shape == (1025, 1025)
    assert res.crs == "EPSG:32643"
    assert (out_dir / "change_probability.tif").is_file()
    assert (out_dir / "change_mask.tif").is_file()
    assert (out_dir / "inference_summary.json").is_file()
    assert res.stats["total_pixels"] == 1025 * 1025
    assert res.stats["valid_pixels"] > 0
