"""Tests for PriorityClassifier model and training pipeline (Phase 4 / Priority Model)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ml.priority.dataset import generate_prototype_dataset
from ml.priority.model import PriorityClassifier, PriorityModelConfig
from ml.priority.train import evaluate_classifier, train_priority_model


@pytest.fixture()
def small_dataset():
    return generate_prototype_dataset(n_samples=120, seed=42)


def test_priority_classifier_fit_and_predict(small_dataset) -> None:
    config = PriorityModelConfig(n_estimators=10, max_depth=3, random_state=42)
    classifier = PriorityClassifier(config=config)
    assert not classifier.is_fitted

    classifier.fit(small_dataset.X, small_dataset.y)
    assert classifier.is_fitted

    probs = classifier.predict_proba(small_dataset.X[:5])
    assert probs.shape == (5, 3)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    preds = classifier.predict(small_dataset.X[:5])
    assert preds.shape == (5,)
    assert set(preds).issubset({0, 1, 2})

    labels = classifier.predict_labels(small_dataset.X[:5])
    assert len(labels) == 5
    assert all(l in {"LOW", "MEDIUM", "HIGH"} for l in labels)


def test_priority_classifier_json_round_trip(small_dataset, tmp_path: Path) -> None:
    model_file = tmp_path / "model.json"
    config = PriorityModelConfig(n_estimators=10, max_depth=3, random_state=42)
    classifier = PriorityClassifier(config=config)
    classifier.fit(small_dataset.X, small_dataset.y)
    classifier.save(model_file)

    assert model_file.is_file()

    loaded = PriorityClassifier.load(model_file, config=config)
    assert loaded.is_fitted

    orig_probs = classifier.predict_proba(small_dataset.X[:10])
    loaded_probs = loaded.predict_proba(small_dataset.X[:10])
    assert np.allclose(orig_probs, loaded_probs, atol=1e-5)


def test_train_priority_model_pipeline(small_dataset, tmp_path: Path) -> None:
    config = PriorityModelConfig(n_estimators=15, max_depth=3, random_state=7)
    model, metadata = train_priority_model(
        dataset=small_dataset,
        config=config,
        models_dir=tmp_path,
        seed=7,
        save_artifacts=True,
    )

    assert (tmp_path / "priority_xgboost.json").is_file()
    assert (tmp_path / "priority_metadata.json").is_file()

    assert metadata["artifact"] == "terraguard.ai/priority_xgboost"
    assert metadata["dataset"]["is_prototype_rule_derived"] is True
    assert "accuracy" in metadata["metrics"]["test"]
    assert "macro_f1" in metadata["metrics"]["test"]
    assert len(metadata["feature_importances"]) == 25
