"""Siamese U-Net training (Step 3.4).

Loss
----
``BCE + Dice``: ``BCEWithLogitsLoss`` for stable per-pixel gradients plus a soft Dice
term for the changed/unchanged imbalance (change pixels are typically a small fraction
of a scene). All denominators are epsilon-clamped so empty masks never divide by zero.

Metrics
-------
Precision / Recall / F1 / IoU from micro-averaged confusion counts accumulated across
each epoch (the correct pool for segmentation), zero-division protected - an empty
prediction or target yields 0.0, never NaN, and raw counts are kept alongside so a
degenerate epoch is visible instead of silently averaged away.

Split
-----
Location-level, as the data ships it: official OSCD ``train.txt`` / ``test.txt``
regions. Validation is a seeded, recorded *location-level* partition of the official
train regions (OSCD ships no val split); pixels of one scene can never straddle a
split. The Pune Sentinel-2 pair is never a training input.

Checkpoint
----------
Best = highest validation IoU; a worse epoch never overwrites it. Unmeasured metrics
are absent from the metadata, never invented.

Reproducibility
---------------
Python / NumPy / torch seeds set from ``TrainingConfig.seed`` and recorded. CPU is the
default device; CUDA only when actually available.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

from ml.change_detection.dataset import BAND_ORDER, LevirDataset, OscdDataset, oscd_split_regions
from ml.change_detection.model import SiameseUNet, SiameseUNetConfig

DEFAULT_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
DEFAULT_OSCD_ROOT = Path(__file__).resolve().parents[2] / "data" / "external" / "oscd"
DEFAULT_LEVIR_ROOT = Path(__file__).resolve().parents[2] / "data" / "external" / "levir_cd"

EPSILON = 1e-7
BEST_METRIC = "val_iou"
"""Checkpoint selection metric (documented, higher is better)."""

LOSS_DESCRIPTION = "BCEWithLogitsLoss + soft Dice (equal weights), eps=1e-7"


# --------------------------------------------------------------------- configuration


@dataclass(frozen=True)
class TrainingConfig:
    """Everything that shapes a training run; serialized into the model metadata."""

    oscd_root: Path = DEFAULT_OSCD_ROOT
    levir_root: Path | None = None
    dataset: str = "oscd"
    """'oscd' (primary), 'levir' (secondary, explicit) or 'oscd+levir' (concatenated)."""

    epochs: int = 2
    batch_size: int = 2
    learning_rate: float = 1e-3
    patch_size: int = 64
    """Training crop size; validation runs full regions (no cropping noise in metrics)."""

    threshold: float = 0.5
    num_workers: int = 0
    seed: int = 42
    val_fraction: float = 0.2
    device: str | None = None
    """None = auto (CUDA when available, else CPU)."""

    base_channels: int = 16
    depth: int = 4
    bce_weight: float = 1.0
    dice_weight: float = 1.0
    models_dir: Path = DEFAULT_MODELS_DIR

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.patch_size < 16:
            raise ValueError(f"patch_size must be >= 16, got {self.patch_size}")
        if not 0.0 < self.val_fraction < 1.0:
            raise ValueError(f"val_fraction must be in (0, 1), got {self.val_fraction}")
        if self.dataset not in ("oscd", "levir", "oscd+levir"):
            raise ValueError(f"dataset must be oscd | levir | oscd+levir, got {self.dataset!r}")

    @property
    def model_config(self) -> SiameseUNetConfig:
        return SiameseUNetConfig(in_channels=4, base_channels=self.base_channels,
                                 depth=self.depth)

    def resolve_device(self) -> torch.device:
        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def to_dict(self) -> dict:
        payload = dict(vars(self))
        for key in ("oscd_root", "levir_root", "models_dir"):
            payload[key] = None if payload[key] is None else str(payload[key])
        return payload


def set_seeds(seed: int) -> None:
    """Seed python / NumPy / torch (recorded in metadata for reproducibility)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def location_split(regions: list[str], val_fraction: float, seed: int) -> tuple[list[str],
                                                                                list[str]]:
    """Deterministic location-level train/validation partition of ``regions``.

    Sorts first (independent of file order), then assigns with a seeded PRNG - the same
    seed always yields the same split, and every region (hence every pixel) lands on
    exactly one side. Raises when fewer than two regions exist, which would make a
    held-out validation impossible without leakage.
    """
    ordered = sorted(set(regions))
    if len(ordered) < 2:
        raise ValueError(f"need at least 2 regions for a location-level train/validation "
                         f"split, got {len(ordered)}: {ordered}")
    generator = random.Random(f"terraguard-split/{seed}")
    shuffled = ordered[:]
    generator.shuffle(shuffled)
    validation_count = max(1, int(round(val_fraction * len(shuffled))))
    validation = sorted(shuffled[:validation_count])
    training = sorted(shuffled[validation_count:])
    return training, validation


class MetricAccumulator:
    """Micro-averaged binary segmentation counts (confusion pool across an epoch)."""

    def __init__(self) -> None:
        self.true_positive = 0
        self.false_positive = 0
        self.false_negative = 0
        self.true_negative = 0
        self.loss_sum = 0.0
        self.loss_batches = 0

    def add(self, probability: torch.Tensor, target: torch.Tensor, loss: float | None = None,
            threshold: float = 0.5) -> None:
        predicted = (probability > threshold).reshape(-1)
        actual = target.reshape(-1) > 0.5
        self.true_positive += int((predicted & actual).sum())
        self.false_positive += int((predicted & ~actual).sum())
        self.false_negative += int((~predicted & actual).sum())
        self.true_negative += int((~predicted & ~actual).sum())
        if loss is not None:
            self.loss_sum += float(loss)
            self.loss_batches += 1

    @staticmethod
    def _ratio(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 0.0

    def metrics(self) -> dict:
        precision = self._ratio(self.true_positive, self.true_positive + self.false_positive)
        recall = self._ratio(self.true_positive, self.true_positive + self.false_negative)
        f1 = self._ratio(2.0 * precision * recall, precision + recall)
        iou = self._ratio(self.true_positive,
                          self.true_positive + self.false_positive + self.false_negative)
        return {
            "precision": round(precision, 6), "recall": round(recall, 6),
            "f1": round(f1, 6), "iou": round(iou, 6),
            "true_positive": self.true_positive, "false_positive": self.false_positive,
            "false_negative": self.false_negative, "true_negative": self.true_negative,
            "loss": round(self.loss_sum / self.loss_batches, 6) if self.loss_batches else None,
        }


class BCEDiceLoss(nn.Module):
    """BCEWithLogits + soft Dice, stable for empty masks and class imbalance."""

    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0) -> None:
        super().__init__()
        if bce_weight < 0 or dice_weight < 0 or bce_weight + dice_weight == 0:
            raise ValueError("loss weights must be non-negative and not both zero")
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()

    def dice(self, probability: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        axes = tuple(range(1, probability.dim()))
        numerator = 2.0 * (probability * target).sum(dim=axes)
        denominator = (probability + target).sum(dim=axes)
        return ((numerator + EPSILON) / (denominator + EPSILON)).mean()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> dict:
        probability = torch.sigmoid(logits)
        bce = self.bce(logits, target)
        dice = self.dice(probability, target)
        total = self.bce_weight * bce + self.dice_weight * (1.0 - dice)
        return {"loss": total, "bce": bce.detach(), "dice": dice.detach()}


SPLIT_METHODOLOGY = ("official OSCD train.txt/test.txt location split; validation is a "
                     "seeded location-level partition of the official train regions; "
                     "regions are atomic - no pixel appears on both sides of any split")


def build_datasets(config: TrainingConfig) -> tuple:
    """Train/validation datasets plus split provenance, honoring dataset roles.

    OSCD stays the primary source. LEVIR-CD is only used when explicitly requested
    ('levir' or 'oscd+levir'). Validation always evaluates full locations without
    augmentation; when datasets are mixed, validation crops keep batch shapes uniform.
    """
    seed = config.seed
    if config.dataset == "oscd":
        official = oscd_split_regions(config.oscd_root)
        train_regions, val_regions = location_split(official["train"], config.val_fraction, seed)
        training = OscdDataset(config.oscd_root, regions=train_regions,
                               patch_size=config.patch_size, augment=True, seed=seed)
        validation = OscdDataset(config.oscd_root, regions=val_regions, patch_size=None,
                                 augment=False, seed=seed)
        info = {"name": "oscd", "root": str(config.oscd_root),
                "train_regions": train_regions, "validation_regions": val_regions,
                "test_regions": official["test"], "split_methodology": SPLIT_METHODOLOGY}
        return training, validation, info

    if config.dataset == "levir":
        training = LevirDataset(config.levir_root, split="train", patch_size=config.patch_size,
                                augment=True, seed=seed)
        validation = LevirDataset(config.levir_root, split="val", patch_size=None,
                                  augment=False, seed=seed)
        info = {"name": "levir", "root": str(config.levir_root),
                "train_regions": sorted(training.identifiers),
                "validation_regions": sorted(validation.identifiers),
                "test_regions": [], "split_methodology": "official LEVIR-CD train/val/test "
                                                          "folders (location-level)"}
        return training, validation, info

    # 'oscd+levir': concatenated, OSCD listed first and named as primary
    official = oscd_split_regions(config.oscd_root)
    train_regions, val_regions = location_split(official["train"], config.val_fraction, seed)
    oscd_train = OscdDataset(config.oscd_root, regions=train_regions,
                             patch_size=config.patch_size, augment=True, seed=seed)
    oscd_val = OscdDataset(config.oscd_root, regions=val_regions, patch_size=config.patch_size,
                           augment=False, seed=seed)
    levir_train = LevirDataset(config.levir_root, split="train", patch_size=config.patch_size,
                               augment=True, seed=seed)
    levir_val = LevirDataset(config.levir_root, split="val", patch_size=config.patch_size,
                             augment=False, seed=seed)
    training = ConcatDataset([oscd_train, levir_train])
    validation = ConcatDataset([oscd_val, levir_val])
    info = {"name": "oscd+levir", "oscd_root": str(config.oscd_root),
            "levir_root": str(config.levir_root),
            "oscd_train_regions": train_regions, "oscd_validation_regions": val_regions,
            "oscd_test_regions": official["test"], "levir_train_samples": len(levir_train),
            "levir_val_samples": len(levir_val),
            "split_methodology": SPLIT_METHODOLOGY + "; LEVIR-CD official train/val folders "
                                                     "appended as the secondary source"}
    return training, validation, info


def _move(sample: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (sample["before"].to(device), sample["after"].to(device),
            sample["mask"].to(device))


def train_one_epoch(model: SiameseUNet, loader: DataLoader, loss_fn: BCEDiceLoss,
                    optimizer: torch.optim.Optimizer, device: torch.device,
                    threshold: float) -> dict:
    """One optimization pass; returns micro-averaged metrics over the epoch."""
    model.train()
    accumulator = MetricAccumulator()
    for sample in loader:
        before, after, target = _move(sample, device)
        optimizer.zero_grad()
        logits = model(before, after)
        output = loss_fn(logits, target)
        output["loss"].backward()
        optimizer.step()
        # metrics from the same forward that produced the loss (pre-update snapshot)
        accumulator.add(torch.sigmoid(logits.detach()), target, float(output["loss"]),
                        threshold)
    return accumulator.metrics()


def run_validation(model: SiameseUNet, loader: DataLoader, loss_fn: BCEDiceLoss,
                   device: torch.device, threshold: float) -> dict:
    """No-grad metrics over a validation/test loader (no augmentation, full locations)."""
    was_training = model.training
    model.eval()
    accumulator = MetricAccumulator()
    with torch.no_grad():
        for sample in loader:
            before, after, target = _move(sample, device)
            logits = model(before, after)
            output = loss_fn(logits, target)
            accumulator.add(torch.sigmoid(logits), target, float(output["loss"]), threshold)
    if was_training:
        model.train()
    return accumulator.metrics()


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> tuple[
        SiameseUNet, dict]:
    """Rebuild a SiameseUNet from its checkpoint; config comes from the metadata."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"checkpoint not found: {path}\n"
            "Train first: backend/.venv/Scripts/python.exe -m ml.change_detection.train "
            "--oscd-root <unzipped OSCD>")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata", {})
    config_dict = {key: value for key, value in
                   (dict(payload.get("config") or {}) or
                    dict((metadata.get("config") or metadata.get("model") or {}))).items()
                   if key in {"in_channels", "base_channels", "depth", "out_channels"}}
    config = SiameseUNetConfig(**(config_dict or SiameseUNetConfig().to_dict()))
    model = SiameseUNet(config)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    return model, {**metadata, "config": config.to_dict()}


def save_checkpoint(path: Path, model: SiameseUNet, metadata: dict) -> None:
    """Atomic checkpoint write; the JSON metadata twin is written beside it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pth.tmp")
    torch.save({"model_state_dict": model.state_dict(),
                "config": model.config.to_dict(),
                "metadata": metadata}, temporary)
    temporary.replace(path)
    metadata_path = path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


class BestCheckpointTracker:
    """Keeps the best checkpoint by one metric; a worse epoch never overwrites."""

    def __init__(self, metric_name: str = BEST_METRIC) -> None:
        self.metric_name = metric_name
        self.best_value: float | None = None
        self.best_epoch: int | None = None
        self.saved_epochs: list[int] = []

    def is_better(self, value: float) -> bool:
        return self.best_value is None or value > self.best_value

    def update(self, epoch: int, metrics: dict, model: SiameseUNet, path: Path,
               metadata: dict) -> bool:
        value = metrics[self.metric_name]
        if not self.is_better(value):
            return False
        self.best_value = value
        self.best_epoch = epoch
        self.saved_epochs.append(epoch)
        save_checkpoint(path, model, metadata)
        return True


def assemble_metadata(config: TrainingConfig, model: SiameseUNet, split_info: dict,
                      device: torch.device, history: list[dict], best_epoch: int | None,
                      best_value: float | None, validation_metrics: dict,
                      started: float) -> dict:
    """Model metadata (Step 3.7). Unmeasured values stay null - nothing is invented."""
    return {
        "artifact": "terraguard.ai/siamese_unet",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
        "model": model.summary(),
        "input": {"channels": list(BAND_ORDER), "band_order": list(BAND_ORDER),
                  "normalization": "surface_reflectance = clip(DN / 10000, 0, 1) "
                                   "(8-bit datasets: value / 255)"},
        "dataset": split_info,
        "training": {"seed": config.seed, "epochs_requested": config.epochs,
                     "epochs_run": len(history), "batch_size": config.batch_size,
                     "learning_rate": config.learning_rate, "patch_size": config.patch_size,
                     "loss": LOSS_DESCRIPTION, "threshold": config.threshold,
                     "best_epoch": best_epoch,
                     "best_metric": {BEST_METRIC: best_value},
                     "selection_metric": BEST_METRIC,
                     "duration_s": round(time.time() - started, 2)},
        "metrics": {"validation": validation_metrics, "history": history,
                    "test": None},
        "device": str(device),
        "software": {"python": platform.python_version(), "torch": torch.__version__},
    }


def run_training(config: TrainingConfig) -> dict:
    """Full training run; returns the metadata that is also written to disk."""
    set_seeds(config.seed)
    device = config.resolve_device()
    training_dataset, validation_dataset, split_info = build_datasets(config)
    if len(training_dataset) == 0 or len(validation_dataset) == 0:
        raise ValueError("train and validation datasets must both be non-empty")

    generator = torch.Generator().manual_seed(config.seed)
    training_loader = DataLoader(training_dataset, batch_size=config.batch_size, shuffle=True,
                                 num_workers=config.num_workers, generator=generator)
    validation_loader = DataLoader(validation_dataset, batch_size=config.batch_size,
                                   shuffle=False, num_workers=config.num_workers)

    model = SiameseUNet(config.model_config).to(device)
    loss_fn = BCEDiceLoss(config.bce_weight, config.dice_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    checkpoint_path = config.models_dir / "siamese_unet.pth"
    tracker = BestCheckpointTracker(BEST_METRIC)
    history: list[dict] = []
    validation_metrics: dict = {}
    started = time.time()

    for epoch in range(1, config.epochs + 1):
        epoch_started = time.time()
        training_metrics = train_one_epoch(model, training_loader, loss_fn, optimizer,
                                           device, config.threshold)
        validation_metrics = run_validation(model, validation_loader, loss_fn, device,
                                            config.threshold)
        row = {"epoch": epoch, "train_loss": training_metrics["loss"],
               "train_iou": training_metrics["iou"], "val_loss": validation_metrics["loss"],
               "val_precision": validation_metrics["precision"],
               "val_recall": validation_metrics["recall"],
               "val_f1": validation_metrics["f1"], "val_iou": validation_metrics["iou"],
               "duration_s": round(time.time() - epoch_started, 2)}
        history.append(row)

        metadata = assemble_metadata(config, model, split_info, device, history,
                                     tracker.best_epoch, tracker.best_value,
                                     validation_metrics, started)
        if tracker.update(epoch, validation_metrics, model, checkpoint_path, metadata):
            print(f"  epoch {epoch}/{config.epochs} train_loss={row['train_loss']} "
                  f"val_iou={row['val_iou']} (best - checkpoint saved)")
        else:
            print(f"  epoch {epoch}/{config.epochs} train_loss={row['train_loss']} "
                  f"val_iou={row['val_iou']}")

    if tracker.best_epoch is None:
        raise RuntimeError("training finished without a single saveable validation IoU")
    model, _ = load_checkpoint(checkpoint_path, device=device)
    metadata = assemble_metadata(config, model, split_info, device, history,
                                 tracker.best_epoch, tracker.best_value,
                                 validation_metrics, started)
    save_checkpoint(checkpoint_path, model, metadata)
    print(f"  best epoch: {tracker.best_epoch} ({BEST_METRIC}={tracker.best_value})")
    print(f"  checkpoint: {checkpoint_path}")
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TerraGuard AI - Siamese U-Net training")
    parser.add_argument("--oscd-root", type=Path, default=DEFAULT_OSCD_ROOT)
    parser.add_argument("--levir-root", type=Path, default=DEFAULT_LEVIR_ROOT)
    parser.add_argument("--dataset", default="oscd", choices=["oscd", "levir", "oscd+levir"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patch-size", type=int, default=96)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--device", default=None, help="auto (default), cpu or cuda")
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    args = parser.parse_args(argv)

    config = TrainingConfig(oscd_root=args.oscd_root, levir_root=args.levir_root,
                            dataset=args.dataset, epochs=args.epochs,
                            batch_size=args.batch_size, learning_rate=args.learning_rate,
                            patch_size=args.patch_size, threshold=args.threshold,
                            num_workers=args.num_workers, seed=args.seed,
                            val_fraction=args.val_fraction, device=args.device,
                            base_channels=args.base_channels, depth=args.depth,
                            models_dir=args.models_dir)
    print(f"Training SiameseUNet on {config.dataset} ({config.resolve_device()}); "
          f"seed={config.seed}")
    metadata = run_training(config)
    print(f"  validation: {metadata['metrics']['validation']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
