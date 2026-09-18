"""Siamese U-Net inference pipeline (Step 3.6).

Runs change-detection inference on paired satellite observations (preprocessed
reflectance stacks or raw Sentinel-2 directories).

Capabilities:
- Loads trained weight-shared Siamese U-Net from checkpoint
- Accepts before/after imagery (Numpy arrays, PyTorch tensors, or GeoTIFFs)
- Applies SCL validity and cloud masks to prevent false-change artifacts
- Generates float32 change probability map and binary change mask
- Strictly preserves geospatial metadata (CRS, Affine transform, dimensions, nodata)
- Writes geospatial GeoTIFF outputs (change_probability.tif, change_mask.tif) and
  a provenance summary JSON (inference_summary.json)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.crs import CRS
from rasterio.transform import Affine

from app.utils.geo import RasterGrid, raster_profile
from ml.change_detection.model import SiameseUNet
from ml.change_detection.train import load_checkpoint

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELS_DIR = REPO_ROOT / "models"
DEFAULT_CHECKPOINT = DEFAULT_MODELS_DIR / "siamese_unet.pth"
DEFAULT_PROCESSED_DIR = REPO_ROOT / "data" / "processed" / "demo_area"
DEFAULT_RAW_DIR = REPO_ROOT / "data" / "raw" / "demo_area"
DEFAULT_PREDICTIONS_DIR = REPO_ROOT / "data" / "predictions" / "demo_area"

NODATA_PROBABILITY = float("nan")
NODATA_MASK = 255


@dataclass
class InferenceResult:
    """Outcome of change-detection inference on a satellite pair."""

    probability: np.ndarray
    """``[H, W]`` float32 in [0, 1]; NaN for invalid/cloud-masked pixels."""

    change_mask: np.ndarray
    """``[H, W]`` uint8 (1=change, 0=no change, 255=nodata/invalid)."""

    valid_mask: np.ndarray
    """``[H, W]`` boolean validity mask."""

    threshold: float
    shape: tuple[int, int]
    crs: str | None
    transform: tuple[float, ...] | None
    stats: dict
    output_files: dict[str, str]

    def to_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            "shape": list(self.shape),
            "crs": self.crs,
            "transform": list(self.transform) if self.transform else None,
            "stats": self.stats,
            "output_files": self.output_files,
        }


def _file_hash(path: Path) -> dict:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def predict_pair(model: SiameseUNet,
                 before: torch.Tensor | np.ndarray,
                 after: torch.Tensor | np.ndarray,
                 valid_mask: torch.Tensor | np.ndarray | None = None,
                 threshold: float = 0.5,
                 device: torch.device | str = "cpu") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run model inference on a before/after observation pair.

    Returns
    -------
    probability : np.ndarray
        ``[H, W]`` float32 change probability in [0, 1], NaN where invalid.
    change_mask : np.ndarray
        ``[H, W]`` uint8 binary change mask (1=change, 0=no change, 255=nodata).
    combined_valid : np.ndarray
        ``[H, W]`` bool mask of usable pixels.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")

    dev = torch.device(device)
    model.eval()
    model.to(dev)

    # Convert inputs to float32 numpy arrays
    before_np = before.detach().cpu().numpy() if isinstance(before, torch.Tensor) else np.asarray(before, dtype="float32")
    after_np = after.detach().cpu().numpy() if isinstance(after, torch.Tensor) else np.asarray(after, dtype="float32")

    if before_np.shape != after_np.shape:
        raise ValueError(f"before shape {before_np.shape} does not match after shape {after_np.shape}")

    # Determine spatial dimensions and ensure [C, H, W]
    if before_np.ndim == 4:
        # [B, C, H, W] -> assume batch size 1 for single inference
        if before_np.shape[0] != 1:
            raise ValueError(f"predict_pair expects batch size 1, got {before_np.shape[0]}")
        before_arr = before_np[0]
        after_arr = after_np[0]
    elif before_np.ndim == 3:
        before_arr = before_np
        after_arr = after_np
    else:
        raise ValueError(f"expected 3D or 4D tensors/arrays, got ndim={before_np.ndim}")

    channels, height, width = before_arr.shape
    if channels != model.config.in_channels:
        raise ValueError(f"expected {model.config.in_channels} channels, got {channels}")

    # Determine validity mask
    finite_mask = np.all(np.isfinite(before_arr), axis=0) & np.all(np.isfinite(after_arr), axis=0)
    if valid_mask is not None:
        v_np = valid_mask.detach().cpu().numpy() if isinstance(valid_mask, torch.Tensor) else np.asarray(valid_mask)
        v_bool = v_np.squeeze() > 0
        if v_bool.shape != (height, width):
            raise ValueError(f"valid_mask shape {v_bool.shape} does not match spatial shape {(height, width)}")
        combined_valid = finite_mask & v_bool
    else:
        combined_valid = finite_mask

    # Replace NaNs with 0.0 for the network forward pass
    clean_before = np.where(np.isfinite(before_arr), before_arr, np.float32(0.0))
    clean_after = np.where(np.isfinite(after_arr), after_arr, np.float32(0.0))

    t_before = torch.from_numpy(clean_before).unsqueeze(0).to(dev)  # [1, C, H, W]
    t_after = torch.from_numpy(clean_after).unsqueeze(0).to(dev)

    with torch.no_grad():
        logits = model(t_before, t_after)
        prob_tensor = model.probability(logits)

    raw_prob = prob_tensor.squeeze().cpu().numpy().astype("float32")

    # Apply validity mask to probability and binary mask
    probability = np.where(combined_valid, raw_prob, np.float32(NODATA_PROBABILITY))

    change_mask = np.full((height, width), NODATA_MASK, dtype="uint8")
    change_mask[combined_valid] = (raw_prob[combined_valid] > threshold).astype("uint8")

    return probability, change_mask, combined_valid


def write_geotiff(path: str | Path,
                  data: np.ndarray,
                  crs: str | CRS | None,
                  transform: Affine | tuple | list | None,
                  nodata: float | int | None = None,
                  dtype: str | None = None,
                  description: str = "") -> None:
    """Write an array to a GeoTIFF while preserving spatial CRS and transform."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    arr = np.asarray(data)
    height, width = arr.shape[-2:]
    count = 1 if arr.ndim == 2 else arr.shape[0]
    out_dtype = dtype or arr.dtype.name

    resolved_transform = transform if isinstance(transform, Affine) else (
        Affine(*transform[:6]) if transform is not None else Affine.identity()
    )
    resolved_crs = CRS.from_user_input(crs) if crs is not None else None

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": count,
        "dtype": out_dtype,
        "crs": resolved_crs,
        "transform": resolved_transform,
        "nodata": nodata,
    }

    with rasterio.open(path, "w", **profile) as dst:
        if count == 1 and arr.ndim == 2:
            dst.write(arr.astype(out_dtype), 1)
        else:
            dst.write(arr.astype(out_dtype))
        if description:
            dst.update_tags(description=description)


def run_inference(checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
                  processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
                  output_dir: str | Path = DEFAULT_PREDICTIONS_DIR,
                  threshold: float = 0.5,
                  device: str | None = None) -> InferenceResult:
    """Execute change detection on a preprocessed Sentinel-2 pair directory.

    Reads ``before.tif`` / ``after.tif`` (or ``*.npy``) and ``valid_mask.npy``,
    runs the Siamese U-Net, and writes GeoTIFF outputs with full georeference.
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"Trained model checkpoint not found at: {ckpt_path}\n"
            "Train a model first: backend/.venv/Scripts/python.exe -m ml.change_detection.train"
        )

    proc_dir = Path(processed_dir)
    if not proc_dir.is_dir():
        raise FileNotFoundError(
            f"Processed data directory not found at: {proc_dir}\n"
            "Run preprocessing first: backend/.venv/Scripts/python.exe -m app.services.preprocessing"
        )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load inputs and spatial metadata
    before_tif = proc_dir / "before.tif"
    after_tif = proc_dir / "after.tif"
    before_npy = proc_dir / "before.npy"
    after_npy = proc_dir / "after.npy"
    valid_mask_npy = proc_dir / "valid_mask.npy"

    crs = None
    transform = None
    if before_tif.is_file():
        prof = raster_profile(before_tif)
        crs = prof.get("crs")
        transform = prof.get("transform")
        with rasterio.open(before_tif) as src:
            before_arr = src.read().astype("float32")
        with rasterio.open(after_tif) as src:
            after_arr = src.read().astype("float32")
    elif before_npy.is_file() and after_npy.is_file():
        before_arr = np.load(before_npy).astype("float32")
        after_arr = np.load(after_npy).astype("float32")
        meta_json = proc_dir / "metadata.json"
        if meta_json.is_file():
            meta = json.loads(meta_json.read_text(encoding="utf-8"))
            grid_meta = meta.get("grid", {})
            crs = grid_meta.get("crs")
            transform = grid_meta.get("transform")
    else:
        raise FileNotFoundError(f"Missing before/after rasters or arrays in {proc_dir}")

    valid_mask = None
    if valid_mask_npy.is_file():
        valid_mask = np.load(valid_mask_npy).astype(bool)

    # 2. Load model
    resolved_device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_meta = load_checkpoint(ckpt_path, device=resolved_device)

    # 3. Predict
    started_at = time.time()
    probability, change_mask, combined_valid = predict_pair(
        model=model,
        before=before_arr,
        after=after_arr,
        valid_mask=valid_mask,
        threshold=threshold,
        device=resolved_device,
    )
    duration_s = round(time.time() - started_at, 3)

    # 4. Write GeoTIFF outputs
    prob_path = out_dir / "change_probability.tif"
    mask_path = out_dir / "change_mask.tif"
    summary_path = out_dir / "inference_summary.json"

    write_geotiff(
        path=prob_path,
        data=probability,
        crs=crs,
        transform=transform,
        nodata=NODATA_PROBABILITY,
        dtype="float32",
        description=f"TerraGuard AI - Change Probability (threshold={threshold})",
    )

    write_geotiff(
        path=mask_path,
        data=change_mask,
        crs=crs,
        transform=transform,
        nodata=NODATA_MASK,
        dtype="uint8",
        description=f"TerraGuard AI - Binary Change Mask (1=change, 0=no_change, {NODATA_MASK}=nodata)",
    )

    # 5. Compute statistics
    total_pixels = int(combined_valid.size)
    valid_pixels = int(combined_valid.sum())
    change_pixels = int((change_mask == 1).sum())
    no_change_pixels = int((change_mask == 0).sum())
    change_fraction = round(change_pixels / valid_pixels, 6) if valid_pixels else 0.0

    # Area calculation assuming 10 m resolution (100 m2 per pixel)
    pixel_area_m2 = 100.0
    if transform is not None:
        pixel_area_m2 = abs(float(transform[0]) * float(transform[4]))
    change_area_ha = round((change_pixels * pixel_area_m2) / 10_000.0, 2)
    change_area_km2 = round(change_area_ha / 100.0, 4)

    stats = {
        "total_pixels": total_pixels,
        "valid_pixels": valid_pixels,
        "masked_pixels": total_pixels - valid_pixels,
        "valid_fraction": round(valid_pixels / total_pixels, 6) if total_pixels else 0.0,
        "change_pixels": change_pixels,
        "no_change_pixels": no_change_pixels,
        "change_fraction_of_valid": change_fraction,
        "pixel_area_m2": pixel_area_m2,
        "change_area_hectares": change_area_ha,
        "change_area_km2": change_area_km2,
        "duration_s": duration_s,
    }

    output_files = {
        "change_probability_tif": str(prob_path),
        "change_mask_tif": str(mask_path),
        "inference_summary_json": str(summary_path),
    }

    summary = {
        "artifact": "terraguard.ai/inference",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint": str(ckpt_path),
        "processed_dir": str(proc_dir),
        "model": model_meta.get("model", {}),
        "threshold": threshold,
        "crs": str(crs),
        "transform": list(transform) if transform else None,
        "shape": list(probability.shape),
        "statistics": stats,
        "files": {
            "change_probability.tif": _file_hash(prob_path),
            "change_mask.tif": _file_hash(mask_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return InferenceResult(
        probability=probability,
        change_mask=change_mask,
        valid_mask=combined_valid,
        threshold=threshold,
        shape=probability.shape,
        crs=str(crs) if crs else None,
        transform=tuple(transform) if transform else None,
        stats=stats,
        output_files=output_files,
    )


def run_inference_raw(checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
                      raw_dir: str | Path = DEFAULT_RAW_DIR,
                      output_dir: str | Path = DEFAULT_PREDICTIONS_DIR,
                      threshold: float = 0.5,
                      device: str | None = None) -> InferenceResult:
    """Run inference starting from a raw Sentinel-2 directory by invoking preprocessing."""
    from app.services.preprocessing import DEFAULT_PROCESSED_DIR, prepare_pair

    processed_meta = prepare_pair(raw_dir=raw_dir, processed_dir=DEFAULT_PROCESSED_DIR)
    processed_dir = Path(processed_meta.get("directory", DEFAULT_PROCESSED_DIR))
    return run_inference(
        checkpoint_path=checkpoint_path,
        processed_dir=processed_dir,
        output_dir=output_dir,
        threshold=threshold,
        device=device,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TerraGuard AI - Siamese U-Net Inference")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
                        help="Path to trained siamese_unet.pth checkpoint")
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR,
                        help="Path to preprocessed satellite pair directory")
    parser.add_argument("--raw-dir", type=Path, default=None,
                        help="Optional raw Sentinel-2 pair directory to preprocess first")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PREDICTIONS_DIR,
                        help="Directory to write output GeoTIFFs and summary")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Change probability threshold in [0, 1]")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to run inference on (e.g. cpu, cuda)")

    args = parser.parse_args(argv)

    try:
        if args.raw_dir is not None:
            res = run_inference_raw(
                checkpoint_path=args.checkpoint,
                raw_dir=args.raw_dir,
                output_dir=args.output_dir,
                threshold=args.threshold,
                device=args.device,
            )
        else:
            res = run_inference(
                checkpoint_path=args.checkpoint,
                processed_dir=args.processed_dir,
                output_dir=args.output_dir,
                threshold=args.threshold,
                device=args.device,
            )

        print("\nTerraGuard AI - Inference Complete")
        print("=" * 60)
        print(f"Output Directory : {args.output_dir}")
        print(f"Grid Dimensions  : {res.shape[0]} x {res.shape[1]} (CRS: {res.crs})")
        print(f"Valid Pixels     : {res.stats['valid_pixels']:,} / {res.stats['total_pixels']:,} ({res.stats['valid_fraction'] * 100:.1f}%)")
        print(f"Changed Pixels   : {res.stats['change_pixels']:,} ({res.stats['change_fraction_of_valid'] * 100:.2f}% of valid area)")
        print(f"Changed Area     : {res.stats['change_area_hectares']} ha ({res.stats['change_area_km2']} km2)")
        print(f"Inference Time   : {res.stats['duration_s']} s")
        print(f"Probability Map  : {res.output_files['change_probability_tif']}")
        print(f"Binary Mask      : {res.output_files['change_mask_tif']}")
        return 0
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Inference failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
