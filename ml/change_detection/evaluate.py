"""Siamese U-Net model evaluation (Step 3.5).

Evaluates change-detection models against ground-truth annotations on benchmark
datasets (OSCD, LEVIR-CD) or test fixtures.

Metrics
-------
- Precision : TP / (TP + FP)
- Recall    : TP / (TP + FN)
- F1-Score  : 2 * Precision * Recall / (Precision + Recall)
- IoU       : TP / (TP + FP + FN) (Intersection over Union / Jaccard Index)
- Accuracy  : (TP + TN) / (TP + TN + FP + FN)
- Confusion matrix: True Positive, False Positive, False Negative, True Negative

Validity / Cloud Masking
------------------------
When evaluation samples provide a validity mask (e.g. SCL cloud/shadow mask, nodata,
or valid observation footprint), all confusion counts and metrics are evaluated
strictly over valid pixels. Masked/invalid pixels never contaminate evaluation.

Zero Fabrication
----------------
If model weights or evaluation datasets are not present on disk, evaluation fails
loudly with an actionable FileNotFoundError. Unmeasured metrics stay null and are
never fabricated.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ml.change_detection.dataset import LevirDataset, OscdDataset, oscd_split_regions
from ml.change_detection.model import SiameseUNet
from ml.change_detection.train import BCEDiceLoss, load_checkpoint

DEFAULT_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
DEFAULT_CHECKPOINT = DEFAULT_MODELS_DIR / "siamese_unet.pth"
DEFAULT_OSCD_ROOT = Path(__file__).resolve().parents[2] / "data" / "external" / "oscd"
DEFAULT_LEVIR_ROOT = Path(__file__).resolve().parents[2] / "data" / "external" / "levir_cd"

EPSILON = 1e-7


@dataclass
class EvaluationMetrics:
    """Standard change-detection evaluation metrics and confusion counts."""

    precision: float
    recall: float
    f1: float
    iou: float
    accuracy: float
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int
    total_pixels: int
    valid_pixels: int
    masked_pixels: int
    threshold: float = 0.5
    loss: float | None = None

    @property
    def confusion_matrix(self) -> dict[str, int]:
        return {
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
            "true_negative": self.true_negative,
        }

    def to_dict(self) -> dict:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "iou": self.iou,
            "accuracy": self.accuracy,
            "confusion_matrix": self.confusion_matrix,
            "total_pixels": self.total_pixels,
            "valid_pixels": self.valid_pixels,
            "masked_pixels": self.masked_pixels,
            "valid_fraction": round(self.valid_pixels / self.total_pixels, 6) if self.total_pixels else 0.0,
            "threshold": self.threshold,
            "loss": self.loss,
        }


class EvaluationAccumulator:
    """Micro-averaged confusion counter respecting validity and cloud masks."""

    def __init__(self, threshold: float = 0.5) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        self.threshold = threshold
        self.true_positive = 0
        self.false_positive = 0
        self.false_negative = 0
        self.true_negative = 0
        self.total_pixels = 0
        self.valid_pixels = 0
        self.loss_sum = 0.0
        self.loss_batches = 0

    @staticmethod
    def _to_numpy(tensor_or_array: torch.Tensor | np.ndarray) -> np.ndarray:
        if isinstance(tensor_or_array, torch.Tensor):
            return tensor_or_array.detach().cpu().numpy()
        return np.asarray(tensor_or_array)

    def add(self, probability: torch.Tensor | np.ndarray,
            target: torch.Tensor | np.ndarray,
            valid_mask: torch.Tensor | np.ndarray | None = None,
            loss: float | None = None) -> None:
        """Add prediction and target pair to accumulator, masked by validity."""
        prob_np = self._to_numpy(probability).astype("float32")
        target_np = self._to_numpy(target).astype("float32")

        if prob_np.shape != target_np.shape:
            prob_flat = prob_np.squeeze()
            target_flat = target_np.squeeze()
            if prob_flat.shape != target_flat.shape:
                raise ValueError(f"probability shape {prob_np.shape} does not match target {target_np.shape}")
        else:
            prob_flat = prob_np.squeeze()
            target_flat = target_np.squeeze()

        total = prob_flat.size
        self.total_pixels += total

        # Compute validity mask
        finite_mask = np.isfinite(prob_flat) & np.isfinite(target_flat)
        if valid_mask is not None:
            valid_np = self._to_numpy(valid_mask).squeeze() > 0
            if valid_np.shape != prob_flat.shape:
                raise ValueError(f"valid_mask shape {valid_np.shape} does not match probability {prob_flat.shape}")
            valid = finite_mask & valid_np
        else:
            valid = finite_mask

        valid_count = int(valid.sum())
        self.valid_pixels += valid_count

        if valid_count > 0:
            pred_valid = prob_flat[valid] > self.threshold
            target_valid = target_flat[valid] > 0.5

            tp = int((pred_valid & target_valid).sum())
            fp = int((pred_valid & ~target_valid).sum())
            fn = int((~pred_valid & target_valid).sum())
            tn = int((~pred_valid & ~target_valid).sum())

            self.true_positive += tp
            self.false_positive += fp
            self.false_negative += fn
            self.true_negative += tn

        if loss is not None:
            self.loss_sum += float(loss)
            self.loss_batches += 1

    @staticmethod
    def _ratio(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 0.0

    def compute(self) -> EvaluationMetrics:
        """Calculate final metrics from accumulated confusion counts."""
        tp = self.true_positive
        fp = self.false_positive
        fn = self.false_negative
        tn = self.true_negative

        precision = self._ratio(tp, tp + fp)
        recall = self._ratio(tp, tp + fn)
        f1 = self._ratio(2.0 * precision * recall, precision + recall)
        iou = self._ratio(tp, tp + fp + fn)
        accuracy = self._ratio(tp + tn, tp + tn + fp + fn)

        avg_loss = round(self.loss_sum / self.loss_batches, 6) if self.loss_batches else None
        masked_pixels = self.total_pixels - self.valid_pixels

        return EvaluationMetrics(
            precision=round(precision, 6),
            recall=round(recall, 6),
            f1=round(f1, 6),
            iou=round(iou, 6),
            accuracy=round(accuracy, 6),
            true_positive=tp,
            false_positive=fp,
            false_negative=fn,
            true_negative=tn,
            total_pixels=self.total_pixels,
            valid_pixels=self.valid_pixels,
            masked_pixels=masked_pixels,
            threshold=self.threshold,
            loss=avg_loss,
        )


def evaluate_single_pair(model: SiameseUNet,
                         before: torch.Tensor | np.ndarray,
                         after: torch.Tensor | np.ndarray,
                         target: torch.Tensor | np.ndarray,
                         valid_mask: torch.Tensor | np.ndarray | None = None,
                         threshold: float = 0.5,
                         device: torch.device | str = "cpu") -> EvaluationMetrics:
    """Evaluate Siamese U-Net on a single bi-temporal observation pair."""
    dev = torch.device(device)
    model.eval()
    model.to(dev)

    def _prepare_tensor(x: torch.Tensor | np.ndarray) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            t = torch.from_numpy(x.astype("float32"))
        else:
            t = x.float()
        if t.dim() == 3:
            t = t.unsqueeze(0)  # [C, H, W] -> [1, C, H, W]
        return t.to(dev)

    before_t = _prepare_tensor(before)
    after_t = _prepare_tensor(after)

    with torch.no_grad():
        logits = model(before_t, after_t)
        probability = model.probability(logits)

    accumulator = EvaluationAccumulator(threshold=threshold)
    accumulator.add(probability=probability.squeeze().cpu().numpy(),
                    target=target,
                    valid_mask=valid_mask)
    return accumulator.compute()


def evaluate_dataset(model: SiameseUNet,
                     dataset_or_loader: Dataset | DataLoader,
                     device: torch.device | str = "cpu",
                     threshold: float = 0.5,
                     loss_fn: nn.Module | None = None,
                     batch_size: int = 2,
                     num_workers: int = 0) -> EvaluationMetrics:
    """Evaluate Siamese U-Net over an entire dataset or DataLoader."""
    dev = torch.device(device)
    model.eval()
    model.to(dev)

    if isinstance(dataset_or_loader, DataLoader):
        loader = dataset_or_loader
    else:
        loader = DataLoader(dataset_or_loader, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)

    accumulator = EvaluationAccumulator(threshold=threshold)

    with torch.no_grad():
        for sample in loader:
            before = sample["before"].to(dev)
            after = sample["after"].to(dev)
            target = sample["mask"].to(dev)
            valid_mask = sample.get("valid_mask")

            logits = model(before, after)
            probability = model.probability(logits)

            loss_val = None
            if loss_fn is not None:
                out = loss_fn(logits, target)
                loss_val = float(out["loss"].detach()) if isinstance(out, dict) else float(out.detach())

            accumulator.add(probability=probability, target=target,
                            valid_mask=valid_mask, loss=loss_val)

    return accumulator.compute()


def run_evaluation(checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
                   dataset: str = "oscd",
                   oscd_root: str | Path = DEFAULT_OSCD_ROOT,
                   levir_root: str | Path = DEFAULT_LEVIR_ROOT,
                   split: str = "test",
                   threshold: float = 0.5,
                   device: str | None = None,
                   output_json: Path | None = None) -> dict:
    """End-to-end evaluation pipeline from checkpoint and dataset path.

    Fails with an actionable error if checkpoint or dataset is not found.
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found at: {ckpt_path}\n"
            "Train a model first: backend/.venv/Scripts/python.exe -m ml.change_detection.train "
            "--oscd-root <data/external/oscd>"
        )

    resolved_device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, metadata = load_checkpoint(ckpt_path, device=resolved_device)

    started_at = time.time()
    if dataset == "oscd":
        oscd_path = Path(oscd_root)
        if not oscd_path.is_dir():
            raise FileNotFoundError(
                f"OSCD dataset not found at: {oscd_path}\n"
                "Provide an unzipped OSCD dataset root or specify --oscd-root <path>."
            )
        regions_split = oscd_split_regions(oscd_path)
        eval_regions = regions_split.get(split)
        if not eval_regions:
            raise ValueError(f"No regions found for OSCD split {split!r} in {oscd_path}")
        eval_ds = OscdDataset(oscd_path, regions=eval_regions, patch_size=None, augment=False)
        dataset_info = {
            "name": "oscd",
            "root": str(oscd_path),
            "split": split,
            "regions": eval_regions,
            "sample_count": len(eval_ds),
        }
    elif dataset == "levir":
        levir_path = Path(levir_root)
        if not levir_path.is_dir():
            raise FileNotFoundError(
                f"LEVIR-CD dataset not found at: {levir_path}\n"
                "Provide an unzipped LEVIR-CD dataset root or specify --levir-root <path>."
            )
        eval_ds = LevirDataset(levir_path, split=split, patch_size=None, augment=False)
        dataset_info = {
            "name": "levir",
            "root": str(levir_path),
            "split": split,
            "sample_count": len(eval_ds),
        }
    else:
        raise ValueError(f"Unknown dataset {dataset!r}, must be 'oscd' or 'levir'")

    if len(eval_ds) == 0:
        raise ValueError(f"Evaluation dataset {dataset!r} ({split}) is empty.")

    loss_fn = BCEDiceLoss()
    metrics = evaluate_dataset(model, eval_ds, device=resolved_device,
                               threshold=threshold, loss_fn=loss_fn)
    duration_s = round(time.time() - started_at, 3)

    report = {
        "artifact": "terraguard.ai/evaluation",
        "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint": str(ckpt_path),
        "model_config": model.config.to_dict(),
        "dataset": dataset_info,
        "threshold": threshold,
        "metrics": metrics.to_dict(),
        "duration_s": duration_s,
        "device": str(resolved_device),
    }

    if output_json is not None:
        out_p = Path(output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TerraGuard AI - Siamese U-Net evaluation")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
                        help="Path to trained siamese_unet.pth checkpoint")
    parser.add_argument("--dataset", default="oscd", choices=["oscd", "levir"],
                        help="Evaluation dataset to run against")
    parser.add_argument("--oscd-root", type=Path, default=DEFAULT_OSCD_ROOT,
                        help="Path to unzipped OSCD dataset root")
    parser.add_argument("--levir-root", type=Path, default=DEFAULT_LEVIR_ROOT,
                        help="Path to unzipped LEVIR-CD dataset root")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"],
                        help="Dataset split to evaluate")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Change probability threshold in [0, 1]")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to evaluate on (e.g. cpu, cuda)")
    parser.add_argument("--output-json", type=Path, default=None,
                        help="Path to write evaluation report JSON")

    args = parser.parse_args(argv)

    try:
        report = run_evaluation(
            checkpoint_path=args.checkpoint,
            dataset=args.dataset,
            oscd_root=args.oscd_root,
            levir_root=args.levir_root,
            split=args.split,
            threshold=args.threshold,
            device=args.device,
            output_json=args.output_json,
        )
        print(f"\nTerraGuard AI - Evaluation Complete ({report['dataset']['name']} - {report['dataset']['split']})")
        print("=" * 60)
        m = report["metrics"]
        print(f"Precision : {m['precision']:.4f}")
        print(f"Recall    : {m['recall']:.4f}")
        print(f"F1-Score  : {m['f1']:.4f}")
        print(f"IoU       : {m['iou']:.4f}")
        print(f"Accuracy  : {m['accuracy']:.4f}")
        if m.get("loss") is not None:
            print(f"Loss      : {m['loss']:.4f}")
        cm = m["confusion_matrix"]
        print(f"Confusion : TP={cm['true_positive']} FP={cm['false_positive']} FN={cm['false_negative']} TN={cm['true_negative']}")
        print(f"Valid Pix : {m['valid_pixels']:,} / {m['total_pixels']:,} ({m['valid_fraction'] * 100:.1f}%)")
        print(f"Duration  : {report['duration_s']} s on {report['device']}")
        return 0
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
