# Sentinel-2 demo dataset + preprocessing contract

This document records exactly which satellite data TerraGuard AI runs on, where it
came from, and what the preprocessing stage guarantees to the AI stages.

## 1. Source

| Item | Value |
| --- | --- |
| Data | Sentinel-2 Level-2A (surface reflectance) |
| Provider | Copernicus Programme (ESA), distributed as open AWS Open Data COGs |
| Catalogue | Element 84 Earth Search STAC API - `https://earth-search.aws.element84.com/v1` |
| Collection | `sentinel-2-l2a` |
| Credentials | none (open access) |
| Licence | Copernicus Sentinel Data Terms: free, open, attribution requested |

Only the AOI window of each band is downloaded, using GDAL HTTP range requests against
the Cloud-Optimized GeoTIFFs. One acquisition of the demo pair costs ~7 MB instead of
~1 GB per full scene.

## 2. Area of interest

| Item | Value |
| --- | --- |
| AOI | Pune region, Maharashtra, India (`demo_area`) |
| Centre | 73.790 E, 18.530 N |
| Size | 10 240 m x 10 240 m (105.06 km2) |
| Analysis grid | 1025 x 1025 px @ 10 m, EPSG:32643 (UTM zone 43N) |
| MGRS tile | 43QCA |

The AOI bounds are defined in the analysis CRS and snapped to the 10 m grid, so the
window is aligned to the Sentinel-2 tile pixel grid.

## 3. Acquired pair

| | before | after |
| --- | --- | --- |
| Date | 2023-12-15 | 2024-05-03 |
| STAC item | `S2A_43QCA_20231215_0_L2A` | `S2A_43QCA_20240503_0_L2A` |
| Scene cloud cover | 0.0081 % | 0.0002 % |
| Platform | Sentinel-2A | Sentinel-2A |
| Bands fetched | B02, B03, B04, B08 (10 m) + SCL (20 m, auxiliary) | same |

The pair is deliberately controlled: same MGRS tile, same platform, both essentially
cloud-free, 140 days apart across the post-monsoon -> pre-monsoon transition, which is
a real and strong vegetation signal in this region (mean NDVI change -0.097, with
20.3 % of pixels losing more than 0.2 NDVI).

Season-pair alternatives that were measured but rejected: 2024-02-03 -> 2024-05-03
(mean NDVI change only -0.003, too weak to demonstrate change detection).

## 4. Re-fetching the data

Imagery is not committed to git (see `.gitignore`); `acquisition.json` is. Re-create the
exact pair with:

```powershell
backend/.venv/Scripts/python.exe scripts/fetch_sentinel_pair.py `
  --before-start 2023-12-01 --before-end 2023-12-31 `
  --after-start 2024-05-01  --after-end 2024-06-20
```

Scene selection is deterministic (lowest cloud cover, same-tile preference), so this
command re-selects `S2A_43QCA_20231215_0_L2A` and `S2A_43QCA_20240503_0_L2A`.
Running the script with no date arguments uses these same windows, so plain
`python scripts/fetch_sentinel_pair.py` reproduces the committed pair.
Use `--dry-run` to list candidates without downloading, and `--max-cloud` to relax the
cloud filter.

## 5. Layout

```text
data/
├── raw/demo_area/
│   ├── acquisition.json          provenance: scene ids, dates, cloud, band URLs, windows
│   ├── before/{B02,B03,B04,B08,SCL}.tif    raw DN, unmodified, 1025x1025 @ 10 m
│   └── after/{B02,B03,B04,B08,SCL}.tif
└── processed/demo_area/          written by app.services.preprocessing.prepare_pair
    ├── before.tif  after.tif     4-band float32 reflectance, nodata=0
    ├── before.npy  after.npy     [4, H, W] float32 arrays (same values as the tifs)
    ├── before_valid_mask.npy     [H, W] bool
    ├── after_valid_mask.npy      [H, W] bool
    ├── valid_mask.npy            [H, W] bool = before AND after
    ├── ndvi_before.tif  ndvi_after.tif  ndvi_difference.tif   float32, nodata=NaN
    └── metadata.json             grid, normalization, statistics, hashes
```

`SCL.tif` (scene classification: cloud, shadow, water...) is fetched for future
per-pixel cloud masking. It is **not** consumed by the current pipeline.

## 6. Frozen interface for the AI stage

`prepare_pair()` ends by handing the models two tensors and nothing else:

```python
before: torch.Tensor  # [4, H, W] float32, surface reflectance in [0, 1]
after:  torch.Tensor  # [4, H, W] float32
# channels: B02 (blue), B03 (green), B04 (red), B08 (nir)
```

Rules that the rest of the project can rely on:

1. **One grid.** Both observations are resampled onto the intersection of their
   footprints, snapped to 10 m in EPSG:32643. `before[:, y, x]` and `after[:, y, x]`
   always describe the same point on the ground - whatever tile, UTM zone or CRS the
   imagery arrived in.
2. **Fixed normalization.** `reflectance = clip(DN / 10000, 0, 1)`. No per-image
   statistics, so the two dates stay directly comparable and results are reproducible.
3. **Explicit validity.** Invalid pixels (nodata, non-finite, sub-zero) are NaN inside
   the pipeline; on disk they are written as `0` and the authoritative information is in
   the `*_valid_mask.npy` arrays. Tensors are NaN-free (`convert_to_tensor` fills with 0)
   because a network cannot consume NaN.
4. **Provenance.** `metadata.json` carries the scene ids, dates, cloud cover, grid,
   array/sha256 hashes, NDVI statistics and the validation results of the run.

## 7. Verification

```powershell
# full check: raw inputs -> processed artifacts -> tensor contract
backend/.venv/Scripts/python.exe scripts/validate_satellite_pair.py

# unit + integration tests (synthetic shifted / reprojected pairs)
backend/.venv/Scripts/python.exe -m pytest backend/tests -q
```

Latest verified result: `RESULT: PASS` (raw before/after, spatial alignment, 11 processed
artifacts, tensor contract, NDVI), 13/13 tests green.

## 8. Known limitations

- Cloud masking is currently scene-level (scene selection filters cloud cover) rather
  than per-pixel; `SCL.tif` is staged for the next step.
- The pair is same-tile and same-platform, so the cross-CRS / cross-resolution alignment
  logic is exercised by synthetic tests rather than by this pair.
- No atmospheric normalization between dates beyond L2A surface reflectance: residual
  radiometric drift is not corrected (suitable for relative change, not absolute trends).

