"""SHAP Explainability and Action Insight Synthesis (Phase 4 / Priority Model).

Translates machine learning predictions into evidence-based environmental explanations
using TreeSHAP (Lundberg et al., Nature MI 2020).

Capabilities
------------
- Computes exact TreeSHAP attribution values for multi-class XGBoost priority predictions.
- Extracts top positive and negative environmental drivers for assigned priority ratings.
- Maps technical remote sensing features to human-readable domain descriptors.
- Generates natural language action insights for the GIS dashboard and decision-makers.
- Computes global feature importance rankings across datasets.
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "backend"
for entry in (BACKEND_DIR, REPO_ROOT):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import numpy as np
import shap

from app.services.features import FEATURE_NAMES, EnvironmentalFeatures
from ml.priority.dataset import CLASS_NAMES, CLASS_TO_INT, INT_TO_CLASS
from ml.priority.model import PriorityClassifier

FEATURE_DESCRIPTIONS: dict[str, str] = {
    "changed_area_ha": "Total changed area (hectares)",
    "change_fraction_of_valid": "Fraction of observation area with detected change",
    "ndvi_before_mean": "Baseline pre-change vegetation index (NDVI)",
    "ndvi_after_mean": "Post-change vegetation index (NDVI)",
    "ndvi_diff_mean": "Mean landscape NDVI difference",
    "change_ndvi_before_mean": "Pre-change NDVI within change zones",
    "change_ndvi_after_mean": "Post-change NDVI within change zones",
    "change_ndvi_diff_mean": "Mean NDVI drop in change zones",
    "change_ndvi_diff_std": "NDVI variance in change zones",
    "veg_loss_ha_01": "Vegetation loss area (ΔNDVI ≤ -0.1)",
    "veg_loss_ha_02": "Severe vegetation loss area (ΔNDVI ≤ -0.2)",
    "veg_loss_ha_03": "Extreme vegetation destruction area (ΔNDVI ≤ -0.3)",
    "veg_gain_ha_02": "Vegetation regrowth / greening area (ΔNDVI ≥ +0.2)",
    "veg_loss_fraction": "Fraction of AOI undergoing severe vegetation loss",
    "patch_count": "Number of distinct change clusters / patches",
    "mean_patch_area_ha": "Mean change cluster area (hectares)",
    "max_patch_area_ha": "Largest change cluster area (hectares)",
    "patch_density_per_km2": "Change cluster density per km²",
    "large_patch_count": "Count of critical change clusters ≥ 1.0 ha",
    "delta_b02_mean": "Blue band reflectance shift in change zones",
    "delta_b03_mean": "Green band reflectance shift in change zones",
    "delta_b04_mean": "Red band reflectance shift (soil exposure indicator)",
    "delta_b08_mean": "NIR band reflectance shift (biomass loss indicator)",
    "brightness_diff_mean": "Visible spectrum brightness difference",
    "scl_veg_loss_ha": "SCL categorical vegetation loss (hectares)",
}


@dataclass
class DriverFeature:
    """An individual feature contributing to the model's priority prediction."""

    feature: str
    description: str
    value: float
    shap_value: float
    effect: str  # "increases_priority" or "decreases_priority"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExplanationResult:
    """SHAP-derived evidence explanation for a specific priority assessment."""

    predicted_class: str
    predicted_class_int: int
    confidence: float
    probabilities: dict[str, float]
    top_positive_drivers: list[DriverFeature]
    top_negative_drivers: list[DriverFeature]
    action_insight: str
    shap_values_dict: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicted_class": self.predicted_class,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "top_positive_drivers": [d.to_dict() for d in self.top_positive_drivers],
            "top_negative_drivers": [d.to_dict() for d in self.top_negative_drivers],
            "action_insight": self.action_insight,
            "shap_values": self.shap_values_dict,
        }


def synthesize_action_insight(
    priority: str,
    confidence: float,
    positive_drivers: list[DriverFeature],
    feature_dict: dict[str, float],
) -> str:
    """Generate concise, evidence-backed natural language action insight."""
    top_names = [d.description for d in positive_drivers[:2]]
    driver_text = " and ".join(top_names) if top_names else "overall environmental indicators"

    veg_loss_02 = feature_dict.get("veg_loss_ha_02", 0.0)
    changed_ha = feature_dict.get("changed_area_ha", 0.0)
    large_patches = int(feature_dict.get("large_patch_count", 0))

    if priority == "HIGH":
        insight = (
            f"HIGH PRIORITY: Urgent review recommended. Detected {changed_ha:.2f} ha of change with "
            f"{veg_loss_02:.2f} ha of severe vegetation loss (ΔNDVI ≤ -0.2)"
        )
        if large_patches > 0:
            insight += f" across {large_patches} large contiguous cluster(s) (≥1.0 ha)."
        else:
            insight += "."
        insight += f" Assessment driven by {driver_text} (confidence {confidence * 100:.1f}%)."
        return insight

    if priority == "MEDIUM":
        insight = (
            f"MEDIUM PRIORITY: Monitoring advised. Detected {changed_ha:.2f} ha of localized change "
            f"({veg_loss_02:.2f} ha severe vegetation loss). "
            f"Driven primarily by {driver_text} (confidence {confidence * 100:.1f}%)."
        )
        return insight

    # LOW
    insight = (
        f"LOW PRIORITY: Normal environmental variance. Minimal severe vegetation disturbance "
        f"({veg_loss_02:.2f} ha) and no large contiguous change clusters. "
        f"Baseline landscape stability verified (confidence {confidence * 100:.1f}%)."
    )
    return insight


class PriorityExplainer:
    """TreeSHAP explainer for the PriorityClassifier."""

    def __init__(self, classifier: PriorityClassifier) -> None:
        if not classifier.is_fitted:
            raise ValueError("Classifier must be fitted before initializing PriorityExplainer.")
        self.classifier = classifier
        self.explainer = shap.TreeExplainer(classifier.model)
        self.feature_names = list(classifier.feature_names)

    def explain_sample(
        self,
        features: dict[str, float] | np.ndarray | EnvironmentalFeatures,
        predicted_class: str | None = None,
        top_k: int = 4,
    ) -> ExplanationResult:
        """Compute local SHAP explanation and action insight for a single sample."""
        if isinstance(features, EnvironmentalFeatures):
            feat_dict = features.to_feature_dict()
            X_vec = np.array([feat_dict[name] for name in self.feature_names], dtype=np.float32)
        elif isinstance(features, dict):
            feat_dict = {name: float(features.get(name, 0.0)) for name in self.feature_names}
            X_vec = np.array([feat_dict[name] for name in self.feature_names], dtype=np.float32)
        else:
            X_vec = np.asarray(features, dtype=np.float32).flatten()
            feat_dict = {name: float(val) for name, val in zip(self.feature_names, X_vec)}

        X_2d = X_vec.reshape(1, -1)
        probs_raw = self.classifier.predict_proba(X_2d)[0]
        probs = {CLASS_NAMES[i]: round(float(probs_raw[i]), 4) for i in range(len(CLASS_NAMES))}

        pred_idx = int(np.argmax(probs_raw)) if predicted_class is None else CLASS_TO_INT[predicted_class]
        pred_label = INT_TO_CLASS[pred_idx]
        confidence = float(probs_raw[pred_idx])

        # Compute SHAP values [num_samples, num_features, num_classes] or [num_classes, num_samples, num_features]
        raw_shap = self.explainer.shap_values(X_2d)

        # Normalize shape across shap version differences
        if isinstance(raw_shap, list):
            # List of [1, 25] arrays per class
            class_shap = raw_shap[pred_idx][0]
        elif raw_shap.ndim == 3:
            if raw_shap.shape[0] == 1 and raw_shap.shape[2] == len(CLASS_NAMES):
                # [1, 25, 3]
                class_shap = raw_shap[0, :, pred_idx]
            elif raw_shap.shape[0] == len(CLASS_NAMES):
                # [3, 1, 25]
                class_shap = raw_shap[pred_idx, 0, :]
            else:
                class_shap = raw_shap[0, :, pred_idx]
        else:
            class_shap = raw_shap.flatten()

        shap_dict = {name: round(float(val), 6) for name, val in zip(self.feature_names, class_shap)}

        # Split into positive drivers (increasing class probability) and negative drivers
        sorted_indices = np.argsort(class_shap)
        top_pos_indices = sorted_indices[::-1]  # descending
        top_neg_indices = sorted_indices        # ascending

        pos_drivers: list[DriverFeature] = []
        for idx in top_pos_indices:
            val = class_shap[idx]
            if val > 0.0:
                fname = self.feature_names[idx]
                pos_drivers.append(
                    DriverFeature(
                        feature=fname,
                        description=FEATURE_DESCRIPTIONS.get(fname, fname),
                        value=round(float(X_vec[idx]), 4),
                        shap_value=round(float(val), 6),
                        effect="increases_priority",
                    )
                )
            if len(pos_drivers) >= top_k:
                break

        neg_drivers: list[DriverFeature] = []
        for idx in top_neg_indices:
            val = class_shap[idx]
            if val < 0.0:
                fname = self.feature_names[idx]
                neg_drivers.append(
                    DriverFeature(
                        feature=fname,
                        description=FEATURE_DESCRIPTIONS.get(fname, fname),
                        value=round(float(X_vec[idx]), 4),
                        shap_value=round(float(val), 6),
                        effect="decreases_priority",
                    )
                )
            if len(neg_drivers) >= top_k:
                break

        action_insight = synthesize_action_insight(
            priority=pred_label,
            confidence=confidence,
            positive_drivers=pos_drivers,
            feature_dict=feat_dict,
        )

        return ExplanationResult(
            predicted_class=pred_label,
            predicted_class_int=pred_idx,
            confidence=round(confidence, 4),
            probabilities=probs,
            top_positive_drivers=pos_drivers,
            top_negative_drivers=neg_drivers,
            action_insight=action_insight,
            shap_values_dict=shap_dict,
        )

    def global_feature_importance(self, X: np.ndarray) -> dict[str, float]:
        """Compute mean absolute SHAP value per feature across samples."""
        X_arr = np.asarray(X, dtype=np.float32)
        raw_shap = self.explainer.shap_values(X_arr)

        if isinstance(raw_shap, list):
            # List of [N, 25] arrays
            stacked = np.stack(raw_shap, axis=-1)  # [N, 25, 3]
        elif raw_shap.ndim == 3:
            stacked = raw_shap
        else:
            stacked = raw_shap[..., None]

        mean_abs_shap = np.mean(np.abs(stacked), axis=(0, -1))
        res = {name: round(float(val), 6) for name, val in zip(self.feature_names, mean_abs_shap)}
        return dict(sorted(res.items(), key=lambda item: item[1], reverse=True))
