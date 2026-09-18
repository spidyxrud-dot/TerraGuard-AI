"""Tests for Siamese U-Net evaluation pipeline (Step 3.5).

Covers:
- Metric calculation accuracy (precision, recall, F1, IoU, confusion matrix)
- Validity and cloud mask awareness (masked pixels excluded from metrics)
- Zero-division safety (no NaNs on empty targets/predictions)
- Model evaluation on synthetic OSCD and LEVIR-CD fixtures
- CLI and error handling for missing checkpoints/datasets (zero fabrication)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ml.change_detection.evaluate import (
    EvaluationAccumulator,
    EvaluationMetrics,
    evaluate_dataset,
    evaluate_single_pair,
    main,
    run_evaluation,
)
from ml.change_detection.model import SiameseUNet, SiameseUNetConfig
from ml.change_detection.train import save_checkpoint


@pytest.fixture()
def small_model() -> SiameseUNet:
    """Lightweight 2-level Siamese U-Net for fast test execution."""
    return SiameseUNet(SiameseUNetConfig(in_channels=4, base_channels=4, depth=2))


@pytest.fixture()
def dummy_checkpoint(tmp_path: Path, small_model: SiameseUNet) -> Path:
    """Save a valid test checkpoint for evaluation testing."""
    ckpt_path = tmp_path / "models" / "siamese_unet.pth"
    save_checkpoint(ckpt_path, small_model, {"test_fixture": True})
    return ckpt_path


# ------------------------------------------------------------------ metric math tests


def test_evaluation_accumulator_perfect_prediction() -> None:
    acc = EvaluationAccumulator(threshold=0.5)
    prob = torch.tensor([[[[0.9, 0.1], [0.8, 0.2]]]])
    target = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    acc.add(prob, target, loss=0.1)
    metrics = acc.compute()

    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0
    assert metrics.iou == 1.0
    assert metrics.accuracy == 1.0
    assert metrics.true_positive == 2
    assert metrics.true_negative == 2
    assert metrics.false_positive == 0
    assert metrics.false_negative == 0
    assert metrics.valid_pixels == 4
    assert metrics.total_pixels == 4
    assert metrics.masked_pixels == 0
    assert metrics.loss == 0.1


def test_evaluation_accumulator_zero_division_safety() -> None:
    acc = EvaluationAccumulator(threshold=0.5)
    prob = torch.zeros((1, 1, 4, 4))
    target = torch.zeros((1, 1, 4, 4))
    acc.add(prob, target)
    metrics = acc.compute()

    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.f1 == 0.0
    assert metrics.iou == 0.0
    assert metrics.accuracy == 1.0  # 16 true negatives out of 16 pixels
    assert metrics.true_positive == 0
    assert metrics.true_negative == 16


def test_evaluation_accumulator_respects_validity_mask() -> None:
    """Verify that pixels flagged as invalid (e.g. cloud/shadow) are ignored."""
    acc = EvaluationAccumulator(threshold=0.5)
    # Shape 2x2:
    # (0,0): TP (prob 0.9, target 1) - valid
    # (0,1): TN (prob 0.1, target 0) - valid
    # (1,0): would be FP (prob 0.9, target 0) - BUT MASKED OUT (valid=False)
    # (1,1): would be FN (prob 0.1, target 1) - BUT MASKED OUT (valid=False)
    prob = np.array([[[[0.9, 0.1], [0.9, 0.1]]]])
    target = np.array([[[[1.0, 0.0], [0.0, 1.0]]]])
    valid_mask = np.array([[[[True, True], [False, False]]]])

    acc.add(prob, target, valid_mask=valid_mask)
    metrics = acc.compute()

    assert metrics.total_pixels == 4
    assert metrics.valid_pixels == 2
    assert metrics.masked_pixels == 2
    assert metrics.true_positive == 1
    assert metrics.true_negative == 1
    assert metrics.false_positive == 0  # Was masked out!
    assert metrics.false_negative == 0  # Was masked out!
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0
    assert metrics.iou == 1.0


def test_evaluation_metrics_serialization() -> None:
    metrics = EvaluationMetrics(
        precision=0.85,
        recall=0.75,
        f1=0.796875,
        iou=0.662651,
        accuracy=0.92,
        true_positive=75,
        false_positive=13,
        false_negative=25,
        true_negative=887,
        total_pixels=1000,
        valid_pixels=1000,
        masked_pixels=0,
        threshold=0.5,
        loss=0.35,
    )
    d = metrics.to_dict()
    assert d["precision"] == 0.85
    assert d["f1"] == 0.796875
    assert d["confusion_matrix"]["true_positive"] == 75
    assert d["valid_fraction"] == 1.0


# ----------------------------------------------------------- model evaluation tests


def test_evaluate_single_pair(small_model: SiameseUNet) -> None:
    before = torch.rand((4, 32, 32))
    after = torch.rand((4, 32, 32))
    target = (torch.rand((1, 32, 32)) > 0.8).float()
    valid_mask = np.ones((32, 32), dtype=bool)

    metrics = evaluate_single_pair(small_model, before, after, target, valid_mask=valid_mask)
    assert isinstance(metrics, EvaluationMetrics)
    assert metrics.total_pixels == 1024
    assert metrics.valid_pixels == 1024
    assert 0.0 <= metrics.iou <= 1.0
    assert 0.0 <= metrics.precision <= 1.0


def test_evaluate_dataset_oscd_synthetic(small_model: SiameseUNet, train_oscd_root: Path) -> None:
    from ml.change_detection.dataset import OscdDataset

    ds = OscdDataset(train_oscd_root, regions=["t01", "t02"], patch_size=None, augment=False)
    metrics = evaluate_dataset(small_model, ds, device="cpu")
    assert isinstance(metrics, EvaluationMetrics)
    assert metrics.valid_pixels > 0
    assert metrics.total_pixels > 0


def test_evaluate_dataset_levir_synthetic(small_model: SiameseUNet, levir_root: Path) -> None:
    from ml.change_detection.dataset import LevirDataset

    ds = LevirDataset(levir_root, split="test", patch_size=None, augment=False)
    metrics = evaluate_dataset(small_model, ds, device="cpu")
    assert isinstance(metrics, EvaluationMetrics)
    assert metrics.valid_pixels > 0


# --------------------------------------------------------- end-to-end pipeline & CLI


def test_run_evaluation_missing_checkpoint_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        run_evaluation(checkpoint_path=tmp_path / "nonexistent.pth")


def test_run_evaluation_missing_dataset_raises(dummy_checkpoint: Path, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="OSCD dataset not found"):
        run_evaluation(checkpoint_path=dummy_checkpoint, oscd_root=tmp_path / "nonexistent_oscd")


def test_run_evaluation_success_with_output_json(dummy_checkpoint: Path, train_oscd_root: Path, tmp_path: Path) -> None:
    out_json = tmp_path / "eval_report.json"
    report = run_evaluation(
        checkpoint_path=dummy_checkpoint,
        dataset="oscd",
        oscd_root=train_oscd_root,
        split="test",
        threshold=0.5,
        device="cpu",
        output_json=out_json,
    )

    assert out_json.is_file()
    saved = json.loads(out_json.read_text(encoding="utf-8"))
    assert saved["artifact"] == "terraguard.ai/evaluation"
    assert "metrics" in saved
    assert saved["dataset"]["name"] == "oscd"
    assert saved["dataset"]["split"] == "test"
    assert report["metrics"]["iou"] == saved["metrics"]["iou"]


def test_evaluate_cli_main(dummy_checkpoint: Path, train_oscd_root: Path) -> None:
    exit_code = main([
        "--checkpoint", str(dummy_checkpoint),
        "--oscd-root", str(train_oscd_root),
        "--dataset", "oscd",
        "--split", "test",
        "--threshold", "0.5",
        "--device", "cpu",
    ])
    assert exit_code == 0
