"""Priority inference pipeline (Phase 4 / Priority Model).

Takes extracted 25-feature environmental vectors (from satellite change detection)
and produces structured priority assessments with SHAP evidence attribution.
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

from app.services.features import (
    EnvironmentalFeatures,
    extract_environmental_features,
    extract_features_from_processed,
)
from ml.priority.explain import ExplanationResult, PriorityExplainer
from ml.priority.model import PriorityClassifier

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELS_DIR = REPO_ROOT / "models"
DEFAULT_MODEL_PATH = DEFAULT_MODELS_DIR / "priority_xgboost.json"
DEFAULT_METADATA_PATH = DEFAULT_MODELS_DIR / "priority_metadata.json"
DEFAULT_PROCESSED_DIR = REPO_ROOT / "data" / "processed" / "demo_area"
DEFAULT_PREDICTIONS_DIR = REPO_ROOT / "data" / "predictions" / "demo_area"


@dataclass
class PriorityAssessment:
    """Full environmental priority decision with SHAP-backed evidence."""

    priority: str
    confidence: float
    probabilities: dict[str, float]
    action_insight: str
    top_drivers: list[dict[str, Any]]
    environmental_metrics: dict[str, Any]
    label_provenance_notice: str
    model_artifact: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def assess_priority(
    features: EnvironmentalFeatures | dict[str, float] | np.ndarray,
    model: PriorityClassifier | None = None,
    model_path: str | Path | None = None,
    explainer: PriorityExplainer | None = None,
) -> PriorityAssessment:
    """Evaluate environmental priority and compute SHAP explanation from features."""
    # 1. Load model if not provided
    if model is None:
        target_path = Path(model_path or DEFAULT_MODEL_PATH)
        if not target_path.is_file():
            raise FileNotFoundError(
                f"Trained Priority Model not found at: {target_path}\n"
                "Train the model first: backend/.venv/Scripts/python.exe -m ml.priority.train"
            )
        classifier = PriorityClassifier.load(target_path)
    else:
        classifier = model
        target_path = Path(model_path or DEFAULT_MODEL_PATH)

    # 2. Initialize explainer if not provided
    if explainer is None:
        exp = PriorityExplainer(classifier)
    else:
        exp = explainer

    # 3. Format features
    if isinstance(features, EnvironmentalFeatures):
        feat_dict = features.to_feature_dict()
        summary_dict = {
            "changed_area_ha": features.changed_area_ha,
            "change_fraction": features.change_fraction_of_valid,
            "mean_ndvi_drop": features.change_ndvi_diff_mean,
            "severe_veg_loss_ha": features.veg_loss_ha_02,
            "patch_count": features.patch_count,
            "large_patch_count": features.large_patch_count,
            "scl_veg_loss_ha": features.scl_veg_loss_ha,
        }
    elif isinstance(features, dict):
        feat_dict = features
        summary_dict = {k: features.get(k, 0.0) for k in [
            "changed_area_ha", "change_fraction_of_valid", "change_ndvi_diff_mean",
            "veg_loss_ha_02", "patch_count", "large_patch_count", "scl_veg_loss_ha"
        ]}
    else:
        feat_vec = np.asarray(features, dtype=np.float32).flatten()
        feat_dict = {name: float(val) for name, val in zip(classifier.feature_names, feat_vec)}
        summary_dict = feat_dict

    # 4. Generate SHAP explanation & decision
    explanation: ExplanationResult = exp.explain_sample(feat_dict, top_k=4)

    # 5. Read metadata if present
    meta_path = target_path.with_name("priority_metadata.json")
    provenance_note = "Trained on prototype rule-derived multi-criteria labels representing baseline domain heuristics."
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            provenance_note = meta.get("dataset", {}).get("provenance_note", provenance_note)
        except Exception:
            pass

    return PriorityAssessment(
        priority=explanation.predicted_class,
        confidence=explanation.confidence,
        probabilities=explanation.probabilities,
        action_insight=explanation.action_insight,
        top_drivers=[d.to_dict() for d in explanation.top_positive_drivers],
        environmental_metrics=summary_dict,
        label_provenance_notice=provenance_note,
        model_artifact=str(target_path),
    )


def assess_processed_scene(
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
    predictions_dir: str | Path | None = DEFAULT_PREDICTIONS_DIR,
    change_mask_path: str | Path | None = None,
    model_path: str | Path | None = None,
) -> PriorityAssessment:
    """Run full feature extraction on preprocessed rasters and evaluate priority."""
    features = extract_features_from_processed(
        processed_dir=processed_dir,
        predictions_dir=predictions_dir,
        change_mask_path=change_mask_path,
    )
    return assess_priority(features=features, model_path=model_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TerraGuard AI - Priority Assessment & SHAP Explanation")
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR,
                        help="Path to preprocessed satellite pair directory")
    parser.add_argument("--predictions-dir", type=Path, default=DEFAULT_PREDICTIONS_DIR,
                        help="Path to predictions directory containing change_mask.tif")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH,
                        help="Path to trained priority_xgboost.json model")
    parser.add_argument("--output-json", type=Path, default=None,
                        help="Optional path to write assessment JSON output")

    args = parser.parse_args(argv)

    try:
        assessment = assess_processed_scene(
            processed_dir=args.processed_dir,
            predictions_dir=args.predictions_dir,
            model_path=args.model_path,
        )

        print("\nTerraGuard AI - Environmental Priority Assessment")
        print("=" * 60)
        print(f"Priority Level   : {assessment.priority} (Confidence: {assessment.confidence * 100:.1f}%)")
        print(f"Probabilities    : {assessment.probabilities}")
        print(f"Action Insight   : {assessment.action_insight}")
        print("\nTop Evidence Drivers (SHAP Attribution):")
        for idx, d in enumerate(assessment.top_drivers, start=1):
            print(f"  {idx}. {d['description']}: value={d['value']} (SHAP impact: +{d['shap_value']:.4f})")
        print(f"\nLabel Provenance : {assessment.label_provenance_notice}")

        if args.output_json is not None:
            out_p = Path(args.output_json)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_text(json.dumps(assessment.to_dict(), indent=2), encoding="utf-8")
            print(f"Report saved to  : {out_p}")
        return 0
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Priority assessment failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
