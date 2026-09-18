"""XGBoost Environmental Priority Classifier (Phase 4 / Priority Model).

Classifies 25-feature environmental vectors into action-oriented priorities:
- LOW (0)
- MEDIUM (1)
- HIGH (2)
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "backend"
for entry in (BACKEND_DIR, REPO_ROOT):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import numpy as np
import xgboost as xgb

from app.services.features import FEATURE_NAMES
from ml.priority.dataset import CLASS_NAMES, CLASS_TO_INT, INT_TO_CLASS


@dataclass
class PriorityModelConfig:
    """Hyperparameters and configuration for the Priority XGBoost model."""

    max_depth: int = 4
    n_estimators: int = 120
    learning_rate: float = 0.08
    subsample: float = 0.85
    colsample_bytree: float = 0.85
    min_child_weight: float = 1.0
    gamma: float = 0.1
    objective: str = "multi:softprob"
    num_class: int = 3
    eval_metric: str = "mlogloss"
    random_state: int = 42

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PriorityClassifier:
    """Wrapper around XGBoost Classifier with 25-feature alignment and JSON persistence."""

    def __init__(self, config: PriorityModelConfig | None = None) -> None:
        self.config = config or PriorityModelConfig()
        self.feature_names = list(FEATURE_NAMES)
        self.model = xgb.XGBClassifier(
            max_depth=self.config.max_depth,
            n_estimators=self.config.n_estimators,
            learning_rate=self.config.learning_rate,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            min_child_weight=self.config.min_child_weight,
            gamma=self.config.gamma,
            objective=self.config.objective,
            num_class=self.config.num_class,
            eval_metric=self.config.eval_metric,
            random_state=self.config.random_state,
        )
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(self, X: np.ndarray, y: np.ndarray, eval_set: list[tuple[np.ndarray, np.ndarray]] | None = None,
            verbose: bool = False) -> PriorityClassifier:
        """Fit the XGBoost classifier."""
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.int64)

        if X_arr.shape[1] != len(self.feature_names):
            raise ValueError(f"Expected {len(self.feature_names)} features, got {X_arr.shape[1]}")

        self.model.fit(X_arr, y_arr, eval_set=eval_set, verbose=verbose)
        self._is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities ``[N, 3]`` float32."""
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        return self.model.predict_proba(X_arr).astype(np.float32)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict integer class labels ``[N]`` in {0, 1, 2}."""
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        return self.model.predict(X_arr).astype(np.int64)

    def predict_labels(self, X: np.ndarray) -> list[str]:
        """Predict string class labels in {'LOW', 'MEDIUM', 'HIGH'}."""
        ints = self.predict(X)
        return [INT_TO_CLASS[int(idx)] for idx in ints]

    def feature_importances(self) -> dict[str, float]:
        """Feature importance dictionary ordered by importance."""
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        importances = self.model.feature_importances_
        res = {name: round(float(imp), 6) for name, imp in zip(self.feature_names, importances)}
        return dict(sorted(res.items(), key=lambda item: item[1], reverse=True))

    def save(self, path: str | Path) -> None:
        """Save model to native XGBoost JSON format."""
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted model.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))

    @classmethod
    def load(cls, path: str | Path, config: PriorityModelConfig | None = None) -> PriorityClassifier:
        """Load model from native XGBoost JSON file."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Model file not found at: {path}")

        instance = cls(config=config)
        instance.model.load_model(str(path))
        instance._is_fitted = True
        return instance
