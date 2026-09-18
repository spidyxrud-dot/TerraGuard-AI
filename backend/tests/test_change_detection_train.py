"""Tests for Siamese U-Net training (Step 3.4).

Covers the Phase 3 checklist: loss executes and stays finite, gradients flow,
checkpoint is written, best-checkpoint logic never regresses, metrics are finite and
zero-division safe, location-level split integrity, CPU execution end-to-end.
Synthetic OSCD trees keep this fast and CI-safe.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ml.change_detection.dataset import oscd_split_regions
from ml.change_detection.model import SiameseUNet, SiameseUNetConfig
from ml.change_detection.train import (
    BEST_METRIC,
    BCEDiceLoss,
    BestCheckpointTracker,
    MetricAccumulator,
    TrainingConfig,
    build_datasets,
    load_checkpoint,
    location_split,
    run_training,
    save_checkpoint,
)


@pytest.fixture()
def training_config(train_oscd_root: Path, tmp_path: Path) -> TrainingConfig:
    """Tiny but complete CPU run: 2 epochs, 32px patches, a few steps per epoch."""
    return TrainingConfig(oscd_root=train_oscd_root, dataset="oscd", epochs=2,
                          batch_size=2, patch_size=32, base_channels=4, depth=3,
                          num_workers=0, seed=7, models_dir=tmp_path / "models")


# --------------------------------------------------------------------------- loss


def test_bce_dice_executes_and_stays_finite() -> None:
    loss_fn = BCEDiceLoss()
    logits = torch.randn((2, 1, 32, 32))
    target = (torch.rand((2, 1, 32, 32)) > 0.8).float()
    output = loss_fn(logits, target)
    assert torch.isfinite(output["loss"])
    assert torch.isfinite(output["bce"]) and torch.isfinite(output["dice"])
    assert float(output["loss"]) > 0.0


def test_bce_dice_is_stable_for_empty_and_full_masks() -> None:
    loss_fn = BCEDiceLoss()
    logits = torch.randn((2, 1, 16, 16))
    for target in (torch.zeros((2, 1, 16, 16)), torch.ones((2, 1, 16, 16))):
        output = loss_fn(logits, target)
        assert torch.isfinite(output["loss"]), "empty/full masks must not divide by zero"


def test_bce_dice_gradients_flow_to_the_model() -> None:
    model = SiameseUNet(SiameseUNetConfig(base_channels=4, depth=2))
    loss_fn = BCEDiceLoss()
    before, after = torch.rand((1, 4, 32, 32)), torch.rand((1, 4, 32, 32))
    target = (torch.rand((1, 1, 32, 32)) > 0.7).float()
    output = loss_fn(model(before, after), target)
    output["loss"].backward()
    gradients = [parameter.grad for parameter in model.parameters()
                 if parameter.requires_grad]
    assert any(grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
               for grad in gradients), "loss must produce finite non-zero gradients"


def test_loss_weights_are_validated() -> None:
    with pytest.raises(ValueError, match="weights"):
        BCEDiceLoss(bce_weight=0.0, dice_weight=0.0)


# ------------------------------------------------------------------------ metrics


def test_metric_accumulator_matches_hand_computed_values() -> None:
    accumulator = MetricAccumulator()
    probability = torch.tensor([[[[0.9, 0.2], [0.8, 0.1]]]])
    target = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    accumulator.add(probability, target, loss=0.5, threshold=0.5)
    metrics = accumulator.metrics()
    assert metrics["true_positive"] == 2 and metrics["false_negative"] == 0
    assert metrics["false_positive"] == 0 and metrics["true_negative"] == 2
    assert metrics["precision"] == 1.0 and metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0 and metrics["iou"] == 1.0
    assert metrics["loss"] == 0.5


def test_metric_accumulator_zero_division_is_safe() -> None:
    accumulator = MetricAccumulator()
    accumulator.add(torch.zeros((1, 1, 4, 4)), torch.zeros((1, 1, 4, 4)))
    metrics = accumulator.metrics()
    assert metrics["precision"] == 0.0 and metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0 and metrics["iou"] == 0.0
    assert not any(np.isnan(value) for value in metrics.values()
                   if isinstance(value, float))


def test_metric_accumulator_micro_averages_across_batches() -> None:
    accumulator = MetricAccumulator()
    accumulator.add(torch.tensor([[[[0.9]]]]), torch.tensor([[[[1.0]]]]))    # TP
    accumulator.add(torch.tensor([[[[0.9]]]]), torch.tensor([[[[0.0]]]]))    # FP
    accumulator.add(torch.tensor([[[[0.1]]]]), torch.tensor([[[[1.0]]]]))    # FN
    metrics = accumulator.metrics()
    assert metrics["precision"] == pytest.approx(1 / 2)
    assert metrics["recall"] == pytest.approx(1 / 2)
    assert metrics["f1"] == pytest.approx(1 / 2)
    assert metrics["iou"] == pytest.approx(1 / 3)


def test_threshold_is_respected_by_the_accumulator() -> None:
    accumulator = MetricAccumulator()
    accumulator.add(torch.full((1, 1, 2, 2), 0.45), torch.ones((1, 1, 2, 2)), threshold=0.5)
    assert accumulator.true_positive == 0 and accumulator.false_negative == 4
    relaxed = MetricAccumulator()
    relaxed.add(torch.full((1, 1, 2, 2), 0.45), torch.ones((1, 1, 2, 2)), threshold=0.4)
    assert relaxed.true_positive == 4


# -------------------------------------------------------------------------- split


def test_location_split_is_deterministic_and_disjoint() -> None:
    regions = ["delta", "alpha", "charlie", "bravo", "echo"]
    first_train, first_val = location_split(regions, 0.4, seed=42)
    second_train, second_val = location_split(regions, 0.4, seed=42)
    assert first_train == second_train and first_val == second_val
    assert not set(first_train) & set(first_val)
    assert sorted(first_train + first_val) == sorted(regions)


def test_location_split_respects_the_fraction() -> None:
    training, validation = location_split([f"r{i:02d}" for i in range(10)], 0.3, seed=1)
    assert len(validation) == 3 and len(training) == 7


def test_location_split_needs_two_regions() -> None:
    with pytest.raises(ValueError, match="at least 2 regions"):
        location_split(["only"], 0.2, seed=0)


def test_build_datasets_keeps_locations_atomic(train_oscd_root: Path) -> None:
    config = TrainingConfig(oscd_root=train_oscd_root, dataset="oscd", val_fraction=0.5)
    training, validation, info = build_datasets(config)
    assert not set(info["train_regions"]) & set(info["validation_regions"])
    assert (set(info["train_regions"]) | set(info["validation_regions"])
            == set(oscd_split_regions(train_oscd_root)["train"]))
    assert set(training.identifiers) == set(info["train_regions"])
    assert set(validation.identifiers) == set(info["validation_regions"])
    assert info["test_regions"] == ["t01", "t02"], "official test regions stay untouched"


def test_official_test_regions_are_never_trained_on(train_oscd_root: Path) -> None:
    _, _, info = build_datasets(TrainingConfig(oscd_root=train_oscd_root, dataset="oscd"))
    assert not set(info["train_regions"]) & set(info["test_regions"])
    assert not set(info["validation_regions"]) & set(info["test_regions"])


# --------------------------------------------------------------------- checkpoint


def test_best_checkpoint_never_regresses(tmp_path: Path) -> None:
    path = tmp_path / "models" / "siamese_unet.pth"
    model = SiameseUNet(SiameseUNetConfig(base_channels=4, depth=2))
    tracker = BestCheckpointTracker(BEST_METRIC)
    saved = []
    for epoch, iou in ((1, 0.30), (2, 0.55), (3, 0.40), (4, 0.62)):
        if tracker.update(epoch, {"val_iou": iou}, model, path, {"epoch": epoch}):
            saved.append((epoch, iou))
    assert tracker.best_epoch == 4 and tracker.best_value == 0.62
    assert saved == [(1, 0.30), (2, 0.55), (4, 0.62)], "epoch 3 must not overwrite"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["metadata"]["epoch"] == 4


def test_checkpoint_round_trip_preserves_config_and_weights(tmp_path: Path) -> None:
    path = tmp_path / "model.pth"
    model = SiameseUNet(SiameseUNetConfig(base_channels=4, depth=2))
    save_checkpoint(path, model, {"note": "round trip"})
    restored, metadata = load_checkpoint(path)
    assert metadata["note"] == "round trip"
    assert restored.config == model.config
    for original, copy in zip(model.state_dict().values(), restored.state_dict().values()):
        assert torch.equal(original, copy)


def test_load_checkpoint_missing_file_is_actionable(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Train first"):
        load_checkpoint(tmp_path / "missing.pth")


# ------------------------------------------------------------- end-to-end training


def test_run_training_writes_checkpoint_and_metadata(training_config: TrainingConfig) -> None:
    metadata = run_training(training_config)
    checkpoint = training_config.models_dir / "siamese_unet.pth"
    metadata_path = checkpoint.with_suffix(".json")
    assert checkpoint.is_file() and metadata_path.is_file()

    assert metadata["training"]["epochs_run"] == 2
    assert metadata["training"]["seed"] == 7
    assert metadata["training"]["best_epoch"] in (1, 2)
    assert metadata["training"]["selection_metric"] == "val_iou"
    assert metadata["metrics"]["test"] is None, "unmeasured metrics stay null"

    validation = metadata["metrics"]["validation"]
    for key in ("precision", "recall", "f1", "iou"):
        assert isinstance(validation[key], float) and np.isfinite(validation[key])
    assert metadata["model"]["parameters"] > 0
    assert metadata["dataset"]["split_methodology"], "split provenance must be recorded"

    on_disk = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert on_disk["training"]["best_epoch"] == metadata["training"]["best_epoch"]


def test_training_is_reproducible_for_a_fixed_seed(train_oscd_root: Path,
                                                   tmp_path: Path) -> None:
    def one_run(models_dir: Path) -> dict:
        config = TrainingConfig(oscd_root=train_oscd_root, epochs=1, batch_size=2,
                                patch_size=32, base_channels=4, depth=3, seed=11,
                                models_dir=models_dir)
        return run_training(config)

    first = one_run(tmp_path / "m1")
    second = one_run(tmp_path / "m2")

    # Wall-clock duration is intentionally excluded: it is timing, not a metric,
    # and can never be bit-reproducible. Everything else must match exactly.
    def metrics_only(history: list[dict]) -> list[dict]:
        return [{k: v for k, v in row.items() if k != "duration_s"} for row in history]

    assert metrics_only(first["metrics"]["history"]) == metrics_only(second["metrics"]["history"]), \
        "same seed must give identical metrics"


def test_training_requires_an_actual_dataset(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        run_training(TrainingConfig(oscd_root=tmp_path / "missing", dataset="oscd"))


def test_cpu_device_resolution(tmp_path: Path) -> None:
    assert TrainingConfig(device="cpu", models_dir=tmp_path).resolve_device().type == "cpu"
    automatic = TrainingConfig(models_dir=tmp_path).resolve_device()
    assert automatic.type in ("cpu", "cuda")


def test_training_config_rejects_nonsense(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="epochs"):
        TrainingConfig(epochs=0, models_dir=tmp_path)
    with pytest.raises(ValueError, match="dataset"):
        TrainingConfig(dataset="kaggle", models_dir=tmp_path)
    with pytest.raises(ValueError, match="val_fraction"):
        TrainingConfig(val_fraction=1.5, models_dir=tmp_path)