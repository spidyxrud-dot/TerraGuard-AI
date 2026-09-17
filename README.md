# TerraGuard AI

Environmental intelligence from paired satellite observations.

Two Sentinel-2 observations -> preprocessing -> Siamese U-Net change detection ->
NDVI / area / land-cover feature extraction -> XGBoost priority classification ->
SHAP explanation -> action insight on a Leaflet GIS dashboard.

## Stack (locked)

- **Frontend:** React 19 + Vite, Tailwind CSS, Leaflet / React-Leaflet, Axios, Recharts
- **Backend:** Python 3.12, FastAPI, Uvicorn, Pydantic
- **ML:** PyTorch (Siamese U-Net change detection), XGBoost (priority), SHAP (explanations), OpenCV, scikit-learn
- **GIS:** Rasterio, GeoPandas, Shapely, PyProj
- **Database:** PostgreSQL + PostGIS via SQLAlchemy (activated in the DB milestone)

## Repository layout

```text
backend/   FastAPI app + analysis pipeline services
frontend/  React + Vite dashboard
ml/        change_detection/ (Siamese U-Net) and priority/ (XGBoost) pipelines
models/    trained artifacts (siamese_unet.pth, priority_xgboost.json)
data/      raw/, processed/, predictions/, demo/ GeoTIFF workspace
scripts/   environment setup, verification, data acquisition + pair validation
docs/      environment notes and methodology
notebooks/ exploration
```

## Phase 1 status - environment verified

- Python 3.12 venv at `backend/.venv` with the full locked stack installed
- `scripts/verify_environment.py` passes: torch math, GeoTIFF I/O + NDVI,
  CRS reprojection, XGBoost training + SHAP, API imports
- `GET /api/health` returns `{"status": "ok"}` from a live uvicorn process
- `npm run build` compiles the dashboard (Tailwind v4 processed)
- See `docs/ENVIRONMENT.md` for exact versions and the Windows VC++
  runtime fix required for torch on this machine.

## Phase 2 status - real Sentinel-2 data + geospatial preprocessing

- Real Sentinel-2 L2A pair acquired from open Copernicus data through the
  Element 84 Earth Search STAC API (no credentials): 2023-12-15 vs 2024-05-03,
  MGRS tile 43QCA, both ~0 % cloud, bands B02/B03/B04/B08 @ 10 m
- `backend/app/services/preprocessing.py` turns the raw per-band GeoTIFFs into
  two aligned, normalized tensors `[4, H, W]` float32 on one shared grid
  (EPSG:32643, 1025x1025 @ 10 m), plus NDVI and a change map
- `backend/app/services/ndvi.py` computes NDVI/NDVI-difference with NaN
  propagation; `backend/app/utils/geo.py` owns grids and resampling
- `scripts/validate_satellite_pair.py` reports `RESULT: PASS`
- 13 unit/integration tests green, including co-registration of deliberately
  shifted and reprojected synthetic pairs
- See `docs/DATA.md` for the dataset, the tensor contract and how to re-fetch.

Measured change signal of the demo pair: mean NDVI -0.097, 20.3 % of the AOI
lost more than 0.2 NDVI between the two dates.

## Run locally

### Backend

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --reload --port 8000
```

Health check: `http://127.0.0.1:8000/api/health`

### Frontend

```powershell
cd frontend
npm run dev
```

Open `http://localhost:5173` (Vite proxies `/api/*` to port 8000).

### Data + preprocessing pipeline

```powershell
# 1. acquire the demo pair (only needed once, not committed to git)
backend/.venv/Scripts/python.exe scripts/fetch_sentinel_pair.py `
  --before-start 2023-12-01 --before-end 2023-12-31 `
  --after-start 2024-05-01  --after-end 2024-06-20

# 2. raw -> processed -> tensor contract, with the PASS/FAIL report
backend/.venv/Scripts/python.exe scripts/validate_satellite_pair.py

# 3. preprocessing tests
backend/.venv/Scripts/python.exe -m pytest backend/tests -q
```

### Fresh machine setup

```powershell
.\scripts\setup_env.ps1
```

## Next milestone (Step 3.4)

The weight-shared Siamese U-Net is in place (`ml/change_detection/model.py`): encoder
applied to both dates, |before−after| fusion, U-Net decoder with bi-temporal skips,
logits `[B,1,H,W]`, internal padding so the odd 1025x1025 Pune grid flows through
unmodified. Next: `ml/change_detection/train.py` - BCE+Dice loss, Precision/Recall/
F1/IoU tracking, location-level validation, best checkpoint to
`models/siamese_unet.pth` (+ `.json` metadata). The Pune pair stays held-out
demo/inference data.


