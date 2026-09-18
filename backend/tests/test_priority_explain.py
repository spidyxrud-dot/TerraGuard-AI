"""Tests for SHAP Explainability and Action Insight Synthesis (Phase 4 / Priority Model)."""

from __future__ import annotations

import numpy as np
import pytest

from ml.priority.dataset import generate_prototype_dataset
from ml.priority.explain import ExplanationResult, PriorityExplainer
from ml.priority.model import PriorityClassifier, PriorityModelConfig


@pytest.fixture()
def fitted_classifier():
    dataset = generate_prototype_dataset(n_samples=150, seed=42)
    config = PriorityModelConfig(n_estimators=15, max_depth=3, random_state=42)
    classifier = PriorityClassifier(config=config)
    classifier.fit(dataset.X, dataset.y)
    return classifier, dataset


def test_priority_explainer_initialization(fitted_classifier) -> None:
    classifier, _ = fitted_classifier
    explainer = PriorityExplainer(classifier)
    assert len(explainer.feature_names) == 25


def test_priority_explainer_explain_sample(fitted_classifier) -> None:
    classifier, dataset = fitted_classifier
    explainer = PriorityExplainer(classifier)

    sample = dataset.X[0]
    res = explainer.explain_sample(sample, top_k=3)

    assert isinstance(res, ExplanationResult)
    assert res.predicted_class in {"LOW", "MEDIUM", "HIGH"}
    assert 0.0 <= res.confidence <= 1.0
    assert len(res.top_positive_drivers) <= 3
    assert len(res.action_insight) > 10
    assert len(res.shap_values_dict) == 25

    d = res.to_dict()
    assert "top_positive_drivers" in d
    assert "action_insight" in d


def test_global_feature_importance(fitted_classifier) -> None:
    classifier, dataset = fitted_classifier
    explainer = PriorityExplainer(classifier)

    global_imp = explainer.global_feature_importance(dataset.X[:30])
    assert len(global_imp) == 25
    # Verify descending order
    vals = list(global_imp.values())
    assert vals == sorted(vals, reverse=True)
