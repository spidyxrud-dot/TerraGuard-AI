# TerraGuard AI - Environment Notes

Recorded from the Phase 1 setup (Windows 11, build 26200). Re-run
`scripts/verify_environment.py` after any environment change:

```powershell
backend\.venv\Scripts\python.exe scripts/verify_environment.py
```

## Locked versions (installed 2026-09-17)

| Layer | Packages |
| --- | --- |
| Runtime | Python 3.12.10 (venv at `backend/.venv`), Node 24.18.1, npm 11.16.0 |
| API | fastapi 0.141.1, uvicorn 0.53.0, pydantic 2.13.5, python-multipart |
| ML | torch 2.14.0+cpu, torchvision 0.29.0+cpu, scikit-learn 1.9.1, xgboost 3.4.1, shap 0.52.0 |
| Data | numpy 2.5.3, pandas 2.3.3, opencv-python 4.14.0 |
| GIS | rasterio 1.5.1, geopandas 1.1.4, shapely 2.1.2, pyproj 3.8.0 |
| DB | SQLAlchemy 2.0.54, psycopg 3.3.5 (binary), alembic 1.20.0 |
| Frontend | react 19.3.0, vite 8.3.0, tailwindcss 4.3.3, leaflet 1.9.4, react-leaflet 5.0.0, axios 1.20.0, recharts 3.10.1 |

## Windows VC++ runtime issue (torch)

**Symptom.** `import torch` fails with
`OSError: [WinError 126] ... Error loading "...torch\lib\shm.dll" or one of its dependencies.`
`c10.dll` loads fine, but `torch_cpu.dll` cannot resolve its imports.

**Root cause.** `torch_cpu.dll` (2.14) imports `VCRUNTIME140_THREADS.dll`, part of the
MSVC 17.10+ "threads" runtime family (`*threads*.dll`). This machine had
`vcruntime140.dll` 14.44 in System32 but no `*_threads.dll` family members.

**Fixes (either one):**

1. Install the current Microsoft Visual C++ 2015-2022 Redistributable (x64):
   ```powershell
   winget install --id Microsoft.VCRedist.2015+.x64 -e
   ```
   (needs elevation/UAC approval)

2. App-local deployment without elevation - copy the `*_threads*.dll` runtime
   family from any application that bundles it (e.g. DaVinci Resolve) into
   `backend\.venv\Lib\site-packages\torch\lib\`:
   ```powershell
   Copy-Item 'C:\Program Files\Blackmagic Design\DaVinci Resolve\*threads*.dll' `
     'backend\.venv\Lib\site-packages\torch\lib\'
   ```
   Option 2 was used on this machine; `scripts/setup_env.ps1` performs it
   automatically when the source files are present.

## Verified working (Phase 1 exit criteria)

- `scripts/verify_environment.py`: torch math, GeoTIFF write/read + NDVI math,
  CRS reprojection, XGBoost training, SHAP TreeExplainer, API stack imports.
- `uvicorn app.main:app` serves `GET /api/health` -> `{"status": "ok"}`.
- `npm run build` produces the Vite production bundle (Tailwind processed).

## PostgreSQL/PostGIS

Deferred to the database milestone (Step 11). `docker-compose.yml` already
defines the `postgis/postgis:16-3.4` service; verify with `docker compose up -d`
once Docker Desktop is available on this machine.
