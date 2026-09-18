"""Training pipeline for the XGBoost Environmental Priority Model (Phase 4).

Trains a 3-class priority model over the 25 environmental features, evaluates
multi-class classification metrics (Macro F1, Precision, Recall, Confusion Matrix),
and outputs both the model booster (`models/priority_xgboost.json`) and provenance
metadata (`models/priority_metadata.json`).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "backend"
for entry in (BACKEND_DIR, REPO_ROOT):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
)

from app.services.features import FEATURE_NAMES
from ml.priority.dataset import (
    CLASS_NAMES,
    INT_TO_CLASS,
    LabelProvenance,
    PriorityDataset,
    generate_prototype_dataset,
    stratified_split,
)
from ml.priority.model import PriorityClassifier, PriorityModelConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELS_DIR = REPO_ROOT / "models"
DEFAULT_MODEL_PATH = DEFAULT_MODELS_DIR / "priority_xgboost.json"
DEFAULT_METADATA_PATH = DEFAULT_MODELS_DIR / "priority_metadata.json"


@dataclass
class EvaluationMetrics:
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    cohen_kappa: float
    log_loss_value: float
    per_class_f1: dict[str, float]
    confusion_matrix: list[list[int]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_classifier(model: PriorityClassifier, dataset: PriorityDataset) -> EvaluationMetrics:
    """Compute comprehensive multi-class classification metrics on a dataset."""
    y_true = dataset.y
    y_pred = model.predict(dataset.X)
    y_prob = model.predict_proba(dataset.X)

    acc = float(accuracy_score(y_true, y_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    _, _, f1_per_class, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()
    loss = float(log_loss(y_true, y_prob, labels=[0, 1, 2]))
    kappa = float(cohen_kappa_score(y_true, y_pred))

    per_class = {INT_TO_CLASS[i]: round(float(f1_per_class[i]), 4) for i in range(len(CLASS_NAMES))}

    return EvaluationMetrics(
        accuracy=round(acc, 4),
        macro_precision=round(float(prec), 4),
        macro_recall=round(float(rec), 4),
        macro_f1=round(float(f1), 4),
        cohen_kappa=round(kappa, 4),
        log_loss_value=round(loss, 4),
        per_class_f1=per_class,
        confusion_matrix=cm,
    )


def train_priority_model(
    dataset: PriorityDataset | None = None,
    config: PriorityModelConfig | None = None,
    models_dir: str | Path = DEFAULT_MODELS_DIR,
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
    seed: int = 42,
    save_artifacts: bool = True,
) -> tuple[PriorityClassifier, dict[str, Any]]:
    """Train, validate, test, and save the XGBoost Environmental Priority Model."""
    cfg = config or PriorityModelConfig(random_state=seed)
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)

    # 1. Dataset loading or prototype generation
    if dataset is None:
        raw_dataset = generate_prototype_dataset(n_samples=600, seed=seed)
    else:
        raw_dataset = dataset

    train_ds, val_ds, test_ds = stratified_split(
        raw_dataset, val_fraction=val_fraction, test_fraction=test_fraction, seed=seed
    )

    # 2. Train model
    started_at = time.time()
    classifier = PriorityClassifier(config=cfg)
    classifier.fit(train_ds.X, train_ds.y, eval_set=[(val_ds.X, val_ds.y)], verbose=False)
    training_duration_s = round(time.time() - started_at, 3)

    # 3. Evaluate splits
    train_metrics = evaluate_classifier(classifier, train_ds)
    val_metrics = evaluate_classifier(classifier, val_ds)
    test_metrics = evaluate_classifier(classifier, test_ds)

    importances = classifier.feature_importances()

    # 4. Compile metadata
    is_proto = raw_dataset.label_source == LabelProvenance.RULE_DERIVED_PROTOTYPE
    metadata = {
        "artifact": "terraguard.ai/priority_xgboost",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": {
            "type": "XGBoostClassifier",
            "framework": "xgboost",
            "config": cfg.to_dict(),
            "feature_names": list(FEATURE_NAMES),
            "num_features": len(FEATURE_NAMES),
            "classes": list(CLASS_NAMES),
            "class_mapping": {idx: name for idx, name in enumerate(CLASS_NAMES)},
        },
        "dataset": {
            "label_source": raw_dataset.label_source.value,
            "is_prototype_rule_derived": is_proto,
            "provenance_note": (
                "Trained on prototype rule-derived multi-criteria labels representing baseline domain heuristics. "
                "Distinguished from field-validated ground truth." if is_proto else "Trained on expert-validated field ground truth."
            ),
            "total_samples": len(raw_dataset),
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "test_samples": len(test_ds),
            "class_distribution": raw_dataset.class_counts,
        },
        "metrics": {
            "train": train_metrics.to_dict(),
            "validation": val_metrics.to_dict(),
            "test": test_metrics.to_dict(),
        },
        "feature_importances": importances,
        "training_duration_s": training_duration_s,
    }

    # 5. Save artifacts if requested
    if save_artifacts:
        model_file = models_path / "priority_xgboost.json"
        meta_file = models_path / "priority_metadata.json"
        classifier.save(model_file)
        meta_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return classifier, metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TerraGuard AI - XGBoost Priority Model Training")
    parser.add_argument("--samples", type=int, default=600, help="Number of synthetic samples to generate if no CSV provided")
    parser.add_argument("--dataset-csv", type=Path, default=None, help="Optional path to verified dataset CSV")
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR, help="Directory to save model artifacts")
    parser.add_argument("--learning-rate", type=float, default=0.08, help="XGBoost learning rate")
    parser.add_argument("--max-depth", type=int, default=4, help="XGBoost max tree depth")
    parser.add_argument("--n-estimators", type=int, default=120, help="Number of trees")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args(argv)

    if args.dataset_csv is not None and args.dataset_csv.is_file():
        print(f"Loading dataset from: {args.dataset_csv}")
        dataset = PriorityDataset.load_csv(args.dataset_csv)
    else:
        print(f"Generating synthetic prototype dataset ({args.samples} samples, seed={args.seed})...")
        dataset = generate_prototype_dataset(n_samples=args.samples, seed=args.seed)

    config = PriorityModelConfig(
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        n_estimators=args.n_estimators,
        random_state=args.seed,
    )

    print("Training XGBoost priority classifier...")
    classifier, metadata = train_priority_model(
        dataset=dataset,
        config=config,
        models_dir=args.models_dir,
        seed=args.seed,
        save_artifacts=True,
    )

    test_m = metadata["metrics"]["test"]
    print("\nTraining Complete!")
    print("=" * 60)
    print(f"Test Accuracy  : {test_m['accuracy'] * 100:.2f}%")
    print(f"Test Macro F1  : {test_m['macro_f1']:.4f}")
    print(f"Per-Class F1   : {test_m['per_class_f1']}")
    print(f"Cohen's Kappa  : {test_m['cohen_kappa']:.4f}")
    print(f"Label Source   : {metadata['dataset']['label_source']} (is_prototype={metadata['dataset']['is_prototype_rule_derived']})")
    print(f"Saved Model    : {args.models_dir / 'priority_xgboost.json'}")
    print(f"Saved Metadata : {args.models_dir / 'priority_metadata.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
