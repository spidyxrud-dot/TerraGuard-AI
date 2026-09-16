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
scripts/   environment setup + verification
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

### Fresh machine setup

```powershell
.\scripts\setup_env.ps1
```

## Next milestone (Step 2)

Obtain one before/after Sentinel-2 pair (GeoTIFF, bands B2/B3/B4/B8) for the
demo area, then build `backend/app/services/preprocessing.py` against it
(Phase 2).

