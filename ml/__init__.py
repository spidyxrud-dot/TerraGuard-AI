"""TerraGuard machine-learning pipelines.

Trained on standardized change-detection datasets (OSCD primary, LEVIR-CD secondary);
inference targets the Sentinel-2 contract produced by
``backend/app/services/preprocessing.py::prepare_pair``. The models never see anything
but the frozen ``before [C,H,W] / after [C,H,W]`` tensor interface.
"""
