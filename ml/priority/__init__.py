"""Priority assessment and explainability package (Phase 4 / Priority Model).

Combines 25 environmental change features, XGBoost classification, and TreeSHAP
attribution into action-oriented priority intelligence.
"""

from typing import Any

_DATASET_SYMBOLS = (
    "CLASS_NAMES",
    "CLASS_TO_INT",
    "INT_TO_CLASS",
    "LabelProvenance",
    "PriorityClass",
    "PriorityDataset",
    "derive_prototype_priority_label",
    "generate_prototype_dataset",
    "stratified_split",
)

_MODEL_SYMBOLS = (
    "PriorityClassifier",
    "PriorityModelConfig",
)

_TRAIN_SYMBOLS = (
    "EvaluationMetrics",
    "evaluate_classifier",
    "train_priority_model",
)

_EXPLAIN_SYMBOLS = (
    "FEATURE_DESCRIPTIONS",
    "DriverFeature",
    "ExplanationResult",
    "PriorityExplainer",
)

_INFERENCE_SYMBOLS = (
    "PriorityAssessment",
    "assess_priority",
    "assess_processed_scene",
)

__all__ = list(_DATASET_SYMBOLS + _MODEL_SYMBOLS + _TRAIN_SYMBOLS + _EXPLAIN_SYMBOLS + _INFERENCE_SYMBOLS)


def __getattr__(name: str) -> Any:
    if name in _DATASET_SYMBOLS:
        from ml.priority import dataset

        return getattr(dataset, name)
    if name in _MODEL_SYMBOLS:
        from ml.priority import model

        return getattr(model, name)
    if name in _TRAIN_SYMBOLS:
        from ml.priority import train

        return getattr(train, name)
    if name in _EXPLAIN_SYMBOLS:
        from ml.priority import explain

        return getattr(explain, name)
    if name in _INFERENCE_SYMBOLS:
        from ml.priority import inference

        return getattr(inference, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
