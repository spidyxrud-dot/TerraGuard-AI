"""Change-detection stage: dataset loaders, Siamese U-Net, training, evaluation.

Added incrementally per checkpoint (3.2 datasets, 3.3 model, 3.4 training, 3.5
evaluation, 3.6 inference on the held-out Pune Sentinel-2 pair).

Names are resolved lazily from ``ml.change_detection.dataset`` so that
``python -m ml.change_detection.dataset`` does not double-import the module.
"""

from typing import Any

_DEFERRED = (
    "BAND_ORDER",
    "EIGHT_BIT_SCALE",
    "LEVIR_LAYOUT_HINT",
    "NORMALIZATION_DESCRIPTION",
    "OSCD_LAYOUT_HINT",
    "REFLECTANCE_RANGE",
    "REFLECTANCE_SCALE",
    "ChangePairDataset",
    "LevirDataset",
    "OscdDataset",
    "normalize_eight_bit",
    "normalize_reflectance",
    "oscd_split_regions",
    "read_change_mask",
    "SiameseUNet",
    "SiameseUNetConfig",
    "EvaluationMetrics",
    "EvaluationAccumulator",
    "evaluate_single_pair",
    "evaluate_dataset",
    "run_evaluation",
)

__all__ = list(_DEFERRED)


def __getattr__(name: str) -> Any:
    if name in ("SiameseUNet", "SiameseUNetConfig"):
        from ml.change_detection import model

        return getattr(model, name)
    if name in ("EvaluationMetrics", "EvaluationAccumulator", "evaluate_single_pair", "evaluate_dataset", "run_evaluation"):
        from ml.change_detection import evaluate

        return getattr(evaluate, name)
    if name in ("InferenceResult", "predict_pair", "run_inference", "run_inference_raw"):
        from ml.change_detection import inference

        return getattr(inference, name)
    if name in _DEFERRED:
        from ml.change_detection import dataset

        return getattr(dataset, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

