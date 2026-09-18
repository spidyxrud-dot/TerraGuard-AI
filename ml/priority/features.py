"""Feature extraction utilities for the priority assessment stage (Phase 4).

Wraps backend feature extraction services into ML-ready tabular formats and
feature vectors for XGBoost training and SHAP explainability.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from app.services.features import (
    FEATURE_NAMES,
    EnvironmentalFeatures,
    extract_environmental_features,
    extract_features_from_processed,
)

__all__ = [
    "FEATURE_NAMES",
    "EnvironmentalFeatures",
    "extract_environmental_features",
    "extract_features_from_processed",
]
