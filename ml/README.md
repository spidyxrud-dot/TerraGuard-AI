# Machine learning pipelines

## Frozen input contract

Every model in this directory consumes exactly what
`backend/app/services/preprocessing.py::prepare_pair()` produces - nothing else:

```python
before : torch.Tensor  # [4, H, W] float32 surface reflectance in [0, 1]
after  : torch.Tensor  # [4, H, W] float32 surface reflectance in [0, 1]
# channels: B02 (blue), B03 (green), B04 (red), B08 (nir)
# one shared grid (EPSG:32643, 10 m) so before[:, y, x] and after[:, y, x] are the same place
```

Change label / feature targets come from `data/processed/<aoi>/ndvi_difference.tif`
and the `*_valid_mask.npy` arrays. Details: `docs/DATA.md`.

## Next up

- `change_detection/` - Siamese U-Net trained on the aligned tensor pairs
- `priority/` - XGBoost priority model over NDVI/feature stack, explained with SHAP

Both are added after the preprocessing contract was verified (it is: see
`scripts/validate_satellite_pair.py` -> `RESULT: PASS`).
