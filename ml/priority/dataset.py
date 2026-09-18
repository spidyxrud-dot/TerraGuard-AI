"""Priority dataset and label provenance specification (Phase 4 / Priority Model).

Defines the 3-tier environmental priority taxonomy (LOW, MEDIUM, HIGH) and provides
rigorous label provenance tracking to distinguish between:
1. Prototype / rule-derived labels (heuristic baseline expert policy)
2. Expert / field-validated ground truth (human environmental assessment)

Taxonomy
--------
- LOW (0)    : Minor, seasonal, or localized natural variance; low intervention urgency.
- MEDIUM (1) : Noticeable vegetation disturbance or moderate land-cover change; monitoring recommended.
- HIGH (2)   : Severe deforestation, large contiguous clearing, or critical land degradation; immediate action required.
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "backend"
for entry in (BACKEND_DIR, REPO_ROOT):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import numpy as np

from app.services.features import FEATURE_NAMES, EnvironmentalFeatures

CLASS_NAMES: tuple[str, ...] = ("LOW", "MEDIUM", "HIGH")
CLASS_TO_INT: dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}
INT_TO_CLASS: dict[int, str] = {idx: name for idx, name in enumerate(CLASS_NAMES)}


class PriorityClass(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class LabelProvenance(str, Enum):
    RULE_DERIVED_PROTOTYPE = "rule_derived_prototype"
    EXPERT_VALIDATED = "expert_validated"


def derive_prototype_priority_label(features: dict[str, float] | EnvironmentalFeatures) -> tuple[PriorityClass, dict[str, Any]]:
    """Deterministic multi-criteria environmental priority rule for prototype datasets.

    Scientifically documented baseline heuristic combining:
    1. Severe vegetation loss extent (veg_loss_ha_02, veg_loss_ha_03)
    2. Large contiguous patch fragmentation (large_patch_count, max_patch_area_ha)
    3. Direct SCL deforestation / land conversion (scl_veg_loss_ha)
    4. Total changed area (changed_area_ha)

    Returns
    -------
    priority : PriorityClass
        The derived priority class (LOW, MEDIUM, or HIGH).
    details : dict[str, Any]
        Breakdown of the composite score and rule triggers.
    """
    if isinstance(features, EnvironmentalFeatures):
        f = features.to_feature_dict()
    else:
        f = dict(features)

    veg_loss_02 = f.get("veg_loss_ha_02", 0.0)
    veg_loss_03 = f.get("veg_loss_ha_03", 0.0)
    scl_loss_ha = f.get("scl_veg_loss_ha", 0.0)
    large_patches = f.get("large_patch_count", 0.0)
    max_patch_ha = f.get("max_patch_area_ha", 0.0)
    changed_ha = f.get("changed_area_ha", 0.0)
    change_frac = f.get("change_fraction_of_valid", 0.0)
    veg_loss_frac = f.get("veg_loss_fraction", 0.0)

    # Composite environmental severity score (0 to 100)
    score = 0.0

    # 1. Vegetation loss score (up to 40 pts)
    # Severe loss (NDVI <= -0.2): 1 ha = 4 pts (up to 25 pts); Extreme loss (NDVI <= -0.3): 1 ha = 6 pts (up to 15 pts)
    score += min(25.0, veg_loss_02 * 2.5)
    score += min(15.0, veg_loss_03 * 5.0)

    # 2. SCL direct vegetation-to-bare/water conversion (up to 25 pts)
    score += min(25.0, scl_loss_ha * 5.0)

    # 3. Patch agglomeration & scale (up to 20 pts)
    score += min(12.0, large_patches * 4.0)
    score += min(8.0, max_patch_ha * 1.5)

    # 4. Landscape fraction (up to 15 pts)
    score += min(15.0, (change_frac + veg_loss_frac) * 50.0)

    score = round(min(100.0, score), 2)

    # Hard overrides for critical large-scale events
    hard_override = None
    if veg_loss_02 >= 10.0 or scl_loss_ha >= 5.0 or (large_patches >= 3 and changed_ha >= 8.0):
        priority = PriorityClass.HIGH
        hard_override = "critical_deforestation_scale"
    elif score >= 55.0:
        priority = PriorityClass.HIGH
    elif score >= 22.0 or (veg_loss_02 >= 2.0) or (changed_ha >= 3.0 and large_patches >= 1):
        priority = PriorityClass.MEDIUM
    else:
        priority = PriorityClass.LOW

    details = {
        "composite_score": score,
        "priority": priority.value,
        "hard_override": hard_override,
        "criteria": {
            "veg_loss_ha_02": veg_loss_02,
            "scl_veg_loss_ha": scl_loss_ha,
            "large_patch_count": large_patches,
            "max_patch_area_ha": max_patch_ha,
            "changed_area_ha": changed_ha,
        },
    }
    return priority, details


@dataclass
class PriorityDataset:
    """Tabular dataset of environmental features and priority labels."""

    X: np.ndarray
    """``[N, 25]`` float32 feature matrix aligned with FEATURE_NAMES."""

    y: np.ndarray
    """``[N]`` int64 label array in {0: LOW, 1: MEDIUM, 2: HIGH}."""

    feature_names: list[str] = field(default_factory=lambda: list(FEATURE_NAMES))
    sample_ids: list[str] = field(default_factory=list)
    label_source: LabelProvenance = LabelProvenance.RULE_DERIVED_PROTOTYPE
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:
        return self.X[idx], int(self.y[idx])

    @property
    def class_counts(self) -> dict[str, int]:
        unique, counts = np.unique(self.y, return_counts=True)
        out = {name: 0 for name in CLASS_NAMES}
        for u, c in zip(unique, counts):
            if int(u) in INT_TO_CLASS:
                out[INT_TO_CLASS[int(u)]] = int(c)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "samples": len(self),
            "features": self.feature_names,
            "class_counts": self.class_counts,
            "label_source": self.label_source.value,
            "is_prototype_rule_derived": self.label_source == LabelProvenance.RULE_DERIVED_PROTOTYPE,
            "metadata": self.metadata,
        }

    def save_csv(self, path: str | Path) -> None:
        """Save feature matrix and labels to CSV."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["sample_id", *self.feature_names, "priority_int", "priority_label"])
            for idx in range(len(self)):
                sid = self.sample_ids[idx] if idx < len(self.sample_ids) else f"sample_{idx:05d}"
                row = [sid, *[f"{v:.6f}" for v in self.X[idx]], int(self.y[idx]), INT_TO_CLASS[int(self.y[idx])]]
                writer.writerow(row)

    @classmethod
    def load_csv(cls, path: str | Path, label_source: LabelProvenance = LabelProvenance.EXPERT_VALIDATED) -> PriorityDataset:
        """Load feature matrix and labels from CSV."""
        path = Path(path)
        sample_ids: list[str] = []
        rows: list[list[float]] = []
        labels: list[int] = []

        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sample_ids.append(row.get("sample_id", f"sample_{len(sample_ids):05d}"))
                feat_vals = [float(row[col]) for col in FEATURE_NAMES]
                rows.append(feat_vals)
                if "priority_int" in row:
                    labels.append(int(row["priority_int"]))
                elif "priority_label" in row:
                    labels.append(CLASS_TO_INT[row["priority_label"].upper()])
                else:
                    raise KeyError("CSV must contain 'priority_int' or 'priority_label' column")

        return cls(
            X=np.array(rows, dtype=np.float32),
            y=np.array(labels, dtype=np.int64),
            feature_names=list(FEATURE_NAMES),
            sample_ids=sample_ids,
            label_source=label_source,
            metadata={"source_file": str(path)},
        )


def generate_prototype_dataset(n_samples: int = 600, seed: int = 42) -> PriorityDataset:
    """Generate synthetic environmental change samples spanning realistic scenarios.

    Scenarios Covered:
    1. Low Priority (~40%): Minimal seasonal drift, small noise patches, negligible vegetation loss.
    2. Medium Priority (~35%): Moderate agricultural harvesting, scattered small clearings, moderate NDVI drops.
    3. High Priority (~25%): Severe deforestation, large contiguous clearing patches (>1 ha), soil exposure.

    All samples are labeled via ``derive_prototype_priority_label`` and explicitly
    marked with ``LabelProvenance.RULE_DERIVED_PROTOTYPE``.
    """
    rng = np.random.RandomState(seed)

    features_list: list[np.ndarray] = []
    labels_list: list[int] = []
    sample_ids: list[str] = []

    for idx in range(n_samples):
        sid = f"proto_{idx:05d}"
        scenario = rng.choice(["low_seasonal", "med_agricultural", "high_deforestation"], p=[0.40, 0.35, 0.25])

        if scenario == "low_seasonal":
            changed_area_ha = float(rng.exponential(0.3))
            change_fraction = float(rng.uniform(0.001, 0.03))
            ndvi_before = float(rng.uniform(0.35, 0.65))
            ndvi_diff = float(rng.uniform(-0.08, 0.05))
            veg_loss_ha_01 = changed_area_ha * rng.uniform(0.1, 0.4)
            veg_loss_ha_02 = changed_area_ha * rng.uniform(0.0, 0.05)
            veg_loss_ha_03 = 0.0
            patch_count = int(rng.poisson(3)) + 1
            max_patch_ha = changed_area_ha * rng.uniform(0.2, 0.5)
            large_patch_count = 0
            scl_veg_loss_ha = 0.0
            delta_b04 = float(rng.uniform(-0.02, 0.03))
            delta_b08 = float(rng.uniform(-0.04, 0.02))

        elif scenario == "med_agricultural":
            changed_area_ha = float(rng.uniform(1.0, 6.0))
            change_fraction = float(rng.uniform(0.03, 0.12))
            ndvi_before = float(rng.uniform(0.45, 0.75))
            ndvi_diff = float(rng.uniform(-0.25, -0.10))
            veg_loss_ha_01 = changed_area_ha * rng.uniform(0.5, 0.8)
            veg_loss_ha_02 = changed_area_ha * rng.uniform(0.2, 0.5)
            veg_loss_ha_03 = changed_area_ha * rng.uniform(0.0, 0.1)
            patch_count = int(rng.poisson(8)) + 3
            max_patch_ha = float(rng.uniform(0.4, 1.2))
            large_patch_count = int(max_patch_ha >= 1.0)
            scl_veg_loss_ha = changed_area_ha * rng.uniform(0.1, 0.3)
            delta_b04 = float(rng.uniform(0.04, 0.12))
            delta_b08 = float(rng.uniform(-0.15, -0.05))

        else:  # high_deforestation
            changed_area_ha = float(rng.uniform(5.0, 25.0))
            change_fraction = float(rng.uniform(0.10, 0.35))
            ndvi_before = float(rng.uniform(0.55, 0.85))
            ndvi_diff = float(rng.uniform(-0.55, -0.25))
            veg_loss_ha_01 = changed_area_ha * rng.uniform(0.7, 0.95)
            veg_loss_ha_02 = changed_area_ha * rng.uniform(0.5, 0.85)
            veg_loss_ha_03 = changed_area_ha * rng.uniform(0.2, 0.6)
            patch_count = int(rng.poisson(15)) + 5
            max_patch_ha = float(rng.uniform(2.5, 12.0))
            large_patch_count = int(rng.randint(2, 8))
            scl_veg_loss_ha = changed_area_ha * rng.uniform(0.4, 0.9)
            delta_b04 = float(rng.uniform(0.10, 0.25))
            delta_b08 = float(rng.uniform(-0.35, -0.15))

        ndvi_after = ndvi_before + ndvi_diff
        change_ndvi_before = ndvi_before
        change_ndvi_after = ndvi_before + ndvi_diff * rng.uniform(1.1, 1.4)
        change_ndvi_diff = change_ndvi_after - change_ndvi_before
        change_ndvi_std = float(rng.uniform(0.05, 0.18))
        veg_gain_ha = float(rng.uniform(0.0, 0.3)) if scenario == "low_seasonal" else 0.0
        veg_loss_frac = veg_loss_ha_02 / (changed_area_ha + 1e-5) * change_fraction
        mean_patch_ha = changed_area_ha / max(1, patch_count)
        patch_density = patch_count / 10.5  # 105 km2 AOI
        delta_b02 = delta_b04 * 0.4
        delta_b03 = delta_b04 * 0.6
        brightness_diff = (delta_b02 + delta_b03 + delta_b04) / 3.0

        feat_dict = {
            "changed_area_ha": round(changed_area_ha, 4),
            "change_fraction_of_valid": round(change_fraction, 6),
            "ndvi_before_mean": round(ndvi_before, 4),
            "ndvi_after_mean": round(ndvi_after, 4),
            "ndvi_diff_mean": round(ndvi_diff, 4),
            "change_ndvi_before_mean": round(change_ndvi_before, 4),
            "change_ndvi_after_mean": round(change_ndvi_after, 4),
            "change_ndvi_diff_mean": round(change_ndvi_diff, 4),
            "change_ndvi_diff_std": round(change_ndvi_std, 4),
            "veg_loss_ha_01": round(veg_loss_ha_01, 4),
            "veg_loss_ha_02": round(veg_loss_ha_02, 4),
            "veg_loss_ha_03": round(veg_loss_ha_03, 4),
            "veg_gain_ha_02": round(veg_gain_ha, 4),
            "veg_loss_fraction": round(veg_loss_frac, 6),
            "patch_count": patch_count,
            "mean_patch_area_ha": round(mean_patch_ha, 4),
            "max_patch_area_ha": round(max_patch_ha, 4),
            "patch_density_per_km2": round(patch_density, 4),
            "large_patch_count": large_patch_count,
            "delta_b02_mean": round(delta_b02, 4),
            "delta_b03_mean": round(delta_b03, 4),
            "delta_b04_mean": round(delta_b04, 4),
            "delta_b08_mean": round(delta_b08, 4),
            "brightness_diff_mean": round(brightness_diff, 4),
            "scl_veg_loss_ha": round(scl_veg_loss_ha, 4),
        }

        priority, _ = derive_prototype_priority_label(feat_dict)
        vec = np.array([feat_dict[name] for name in FEATURE_NAMES], dtype=np.float32)

        features_list.append(vec)
        labels_list.append(CLASS_TO_INT[priority.value])
        sample_ids.append(sid)

    return PriorityDataset(
        X=np.array(features_list, dtype=np.float32),
        y=np.array(labels_list, dtype=np.int64),
        feature_names=list(FEATURE_NAMES),
        sample_ids=sample_ids,
        label_source=LabelProvenance.RULE_DERIVED_PROTOTYPE,
        metadata={
            "description": "Synthetic environmental change dataset labeled by documented prototype heuristic policy",
            "seed": seed,
            "total_samples": n_samples,
        },
    )


def stratified_split(
    dataset: PriorityDataset,
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[PriorityDataset, PriorityDataset, PriorityDataset]:
    """Deterministically partition dataset into train, validation, and test splits."""
    if not (0.0 < val_fraction < 1.0 and 0.0 < test_fraction < 1.0 and val_fraction + test_fraction < 1.0):
        raise ValueError("val_fraction and test_fraction must sum to < 1.0")

    rng = np.random.RandomState(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []

    unique_classes = np.unique(dataset.y)
    for c in unique_classes:
        c_indices = np.where(dataset.y == c)[0]
        rng.shuffle(c_indices)

        n_total = len(c_indices)
        n_val = max(1, int(round(val_fraction * n_total)))
        n_test = max(1, int(round(test_fraction * n_total)))

        val_idx = c_indices[:n_val]
        test_idx = c_indices[n_val:n_val + n_test]
        train_idx = c_indices[n_val + n_test:]

        val_indices.extend(val_idx)
        test_indices.extend(test_idx)
        train_indices.extend(train_idx)

    def _subset(indices: list[int], split_name: str) -> PriorityDataset:
        idx_arr = np.array(sorted(indices))
        return PriorityDataset(
            X=dataset.X[idx_arr],
            y=dataset.y[idx_arr],
            feature_names=dataset.feature_names,
            sample_ids=[dataset.sample_ids[i] for i in idx_arr] if dataset.sample_ids else [],
            label_source=dataset.label_source,
            metadata={**dataset.metadata, "split": split_name, "split_samples": len(idx_arr)},
        )

    return _subset(train_indices, "train"), _subset(val_indices, "val"), _subset(test_indices, "test")
