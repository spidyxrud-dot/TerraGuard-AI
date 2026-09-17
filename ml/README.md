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

## Training datasets (Step 3.2)

`ml/change_detection/dataset.py` turns supervised change-detection datasets into the
frozen sample interface `{before [C,H,W], after [C,H,W], mask [1,H,W]}`:

- **OSCD** (primary, real Sentinel-2, 4-band contract natively) - `OscdDataset`.
  Region-level official `train.txt`/`test.txt` split, so no patch can leak between
  training and validation locations. Unannotated regions get an honest zero mask and
  `region_report[...]["annotated"] is False`.
- **LEVIR-CD** (secondary, 8-bit RGB aerial imagery) - `LevirDataset`. It has no NIR:
  R->B04, G->B03, B->B02 and the B08 channel is **zero-filled as an explicit absence
  marker** (`channel_availability == (True, True, True, False)`). NIR is never
  synthesized; models trained on LEVIR-CD are RGB structural-change specialists.

Cropping/flip augmentation is deterministic per sample index (seeded PRNG), so any
index always yields the same window regardless of worker count. Inspect a download:

```powershell
backend/.venv/Scripts/python.exe -m ml.change_detection.dataset --oscd-root <unzipped> --split train
```

## Next up

- `change_detection/train.py` (Step 3.4) - BCE+Dice training with Precision/Recall/F1/IoU,
  location-level validation, best checkpoint to `models/siamese_unet.pth`
- `priority/` - XGBoost priority model over NDVI/feature stack, explained with SHAP

## Siamese U-Net (Step 3.3)

`ml/change_detection/model.py` - `SiameseUNet(SiameseUNetConfig)`:

```text
before [B,C,H,W] ─┐                    C = 4 (B02/B03/B04/B08) or 3 (LEVIR RGB)
                   ├─► SAME encoder ─►  per-date feature pyramids (weights shared)
after  [B,C,H,W] ─┘
      bottleneck: concat(before, after, |before - after|) -> fusion
      U-Net decoder, skips = concat(before_feats, after_feats)
      1x1 head -> change logits [B,1,H,W]
```

- **Siamese by construction**: one encoder module applied to both dates (verified by a
  forward-hook test that counts encoder invocations: exactly `2 x depth` per forward).
- **Any spatial size >= 2^depth works**: the model reflect-pads internally to a multiple
  of `2^depth` and crops the output back - required because the Pune inference grid is
  1025 x 1025 (odd). No manual tiling/padding at inference time.
- `forward` returns raw logits; `probability()` and `change_mask(threshold)` keep the
  decision threshold explicit and out of the graph.

```python
from ml.change_detection import SiameseUNet, SiameseUNetConfig

model = SiameseUNet(SiameseUNetConfig(in_channels=4, base_channels=16, depth=4))
logits = model(before, after)              # [B, 1, H, W]
probability = model.probability(logits)    # [B, 1, H, W] in [0, 1]
mask = model.change_mask(logits, 0.5)      # [B, 1, H, W] bool
```

The Pune Sentinel-2 pair stays held out: demo and inference only, never training data.
