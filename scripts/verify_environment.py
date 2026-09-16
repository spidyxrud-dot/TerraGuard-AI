"""TerraGuard AI - Phase 1 environment verification.

Run with:
    backend\\.venv\\Scripts\\python.exe scripts/verify_environment.py

Proves the locked stack is importable and functional:
torch, torchvision, rasterio (GeoTIFF I/O + NDVI math), geopandas/pyproj
(reprojection), xgboost (training), shap (TreeExplainer), fastapi, sqlalchemy.
"""

from __future__ import annotations

import tempfile
from pathlib import Path


def check_torch() -> None:
    import torch
    import torchvision

    x = torch.rand(8, 4)
    assert (x @ x.T).shape == (8, 8)
    print(f"[ok] torch {torch.__version__} | torchvision {torchvision.__version__} | cuda={torch.cuda.is_available()}")


def check_rasterio_ndvi() -> None:
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    rng = np.random.default_rng(42)
    red = rng.integers(500, 3000, size=(64, 64), dtype=np.uint16)
    nir = np.clip(red.astype(np.int32) + rng.integers(-1000, 2000, size=(64, 64)), 0, 10000).astype(np.uint16)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "pair.tif"
        profile = {
            "driver": "GTiff", "height": 64, "width": 64, "count": 2,
            "dtype": "uint16", "crs": "EPSG:32643", "transform": from_origin(73_000, 2_050_000, 10, 10),
        }
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(red, 1)
            dst.write(nir, 2)
        with rasterio.open(path) as src:
            red_r = src.read(1).astype(np.float32)
            nir_r = src.read(2).astype(np.float32)
            assert src.crs.to_epsg() == 32643
            assert src.res == (10.0, 10.0)

    ndvi = (nir_r - red_r) / (nir_r + red_r + 1e-8)
    assert -1.0 <= float(ndvi.min()) and float(ndvi.max()) <= 1.0
    print(f"[ok] rasterio {rasterio.__version__} | GeoTIFF round-trip + NDVI mean={ndvi.mean():.3f}")


def check_geopandas() -> None:
    import geopandas as gpd
    from shapely.geometry import Point

    gdf = gpd.GeoDataFrame(
        {"name": ["a", "b"]},
        geometry=[Point(73.78, 18.52), Point(73.82, 18.58)],
        crs="EPSG:4326",
    ).to_crs(epsg=32643)
    assert gdf.crs.to_epsg() == 32643
    print(f"[ok] geopandas {gpd.__version__} | reprojection bounds={tuple(round(v, 1) for v in gdf.total_bounds)}")


def check_xgboost_shap() -> None:
    import numpy as np
    import shap
    import xgboost as xgb
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 5))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

    model = xgb.XGBClassifier(n_estimators=40, max_depth=3, eval_metric="logloss")
    model.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, model.predict(X_te))

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_te[:5])
    top = np.abs(np.asarray(shap_values)).mean(axis=0).argmax()
    print(f"[ok] xgboost {xgb.__version__} | shap {shap.__version__} | accuracy={acc:.2f} | top_feature=f{top}")


def check_api_stack() -> None:
    import fastapi
    import sqlalchemy
    import uvicorn

    import pydantic
    import pydantic_settings

    assert sqlalchemy.text("SELECT 1") is not None
    print(
        "[ok] fastapi", fastapi.__version__, "| uvicorn", uvicorn.__version__,
        "| pydantic", pydantic.__version__, "| sqlalchemy", sqlalchemy.__version__,
        "| pydantic-settings", pydantic_settings.__version__,
    )


def main() -> None:
    print("TerraGuard AI - environment verification")
    check_torch()
    check_rasterio_ndvi()
    check_geopandas()
    check_xgboost_shap()
    check_api_stack()
    print("ALL CHECKS PASSED - Phase 1 environment is ready.")


if __name__ == "__main__":
    main()
