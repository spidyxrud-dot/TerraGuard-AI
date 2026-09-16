"""Acquire one real Sentinel-2 L2A before/after pair for the TerraGuard demo AOI.

Source
------
Element 84 Earth Search STAC API (https://earth-search.aws.element84.com/v1) over the
AWS Open Data ``sentinel-cogs`` bucket. Open access - no credentials required.

What this does
--------------
1. Searches two date windows over the demo AOI (default: Pune, MGRS 43QCA, EPSG:32643).
2. Keeps scenes whose footprint fully contains the AOI (so before/after are comparable).
3. Prefers the lowest-cloud scene per window, preferring a shared MGRS tile for the pair.
4. Reads ONLY the AOI window of each required band from the remote Cloud-Optimized
   GeoTIFF (GDAL HTTP range requests), so a run downloads a few MB instead of ~1 GB/scene.
5. Writes the raw bands unmodified to ``data/raw/demo_area/<before|after>/`` plus an
   ``acquisition.json`` provenance record.

Bands: B02 (blue), B03 (green), B04 (red), B08 (nir) at native 10 m, plus SCL (20 m)
as an auxiliary asset for future per-pixel cloud masking.

Usage
-----
    backend/.venv/Scripts/python.exe scripts/fetch_sentinel_pair.py --dry-run
    backend/.venv/Scripts/python.exe scripts/fetch_sentinel_pair.py

Defaults reproduce the committed demo pair (before 2023-12-15, after 2024-05-03);
override --before-start/--before-end/--after-start/--after-end for other seasons.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.warp import transform_bounds
from rasterio.windows import Window, from_bounds

REPO_ROOT = Path(__file__).resolve().parents[1]
STAC_SEARCH_URL = "https://earth-search.aws.element84.com/v1/search"
COLLECTION = "sentinel-2-l2a"

# Sentinel-2 band -> Earth Search asset key
BAND_ASSETS: dict[str, str] = {"B02": "blue", "B03": "green", "B04": "red", "B08": "nir"}
AUX_ASSETS: dict[str, str] = {"SCL": "scl"}

GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "GDAL_HTTP_MULTIRANGE": "YES",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "GDAL_HTTP_MAX_RETRY": "5",
    "GDAL_HTTP_RETRY_DELAY": "2",
    "VSI_CACHE": "TRUE",
    "VSI_CACHE_SIZE": "134217728",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
}


@dataclass(frozen=True)
class AoiSpec:
    """Demo area of interest, defined in the analysis CRS for exact grid alignment."""

    center_lon: float = 73.79
    center_lat: float = 18.53
    edge_m: int = 10_240  # 1024 px x 10 m
    target_epsg: int = 32643  # UTM zone 43N
    resolution_m: int = 10

    @property
    def size_px(self) -> int:
        return int(self.edge_m // self.resolution_m)

    def bounds_utm(self) -> tuple[float, float, float, float]:
        """Snap the AOI to the analysis resolution grid, centred on the AOI centre."""
        transformer = Transformer.from_crs(4326, self.target_epsg, always_xy=True)
        centre_x, centre_y = transformer.transform(self.center_lon, self.center_lat)
        half = self.edge_m / 2
        res = self.resolution_m
        min_x = math.floor((centre_x - half) / res) * res
        min_y = math.floor((centre_y - half) / res) * res
        return (min_x, min_y, min_x + self.edge_m, min_y + self.edge_m)

    def bounds_wgs84(self) -> tuple[float, float, float, float]:
        min_x, min_y, max_x, max_y = self.bounds_utm()
        return transform_bounds(self.target_epsg, 4326, min_x, min_y, max_x, max_y)


@dataclass
class Scene:
    item_id: str
    datetime: str
    cloud_cover: float
    grid_code: str
    assets: dict[str, str]

    @property
    def date(self) -> str:
        return self.datetime[:10]


def search_scenes(client: httpx.Client, bbox_wgs84, start: str, end: str, max_cloud: float) -> list[Scene]:
    body = {
        "collections": [COLLECTION],
        "bbox": list(bbox_wgs84),
        "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z",
        "query": {"eo:cloud_cover": {"lt": max_cloud}},
        "limit": 50,
    }
    response = client.post(STAC_SEARCH_URL, json=body, timeout=120)
    response.raise_for_status()
    scenes: list[Scene] = []
    for feature in response.json().get("features", []):
        props = feature.get("properties", {})
        assets = {key: asset.get("href") for key, asset in feature.get("assets", {}).items()}
        if not all(assets.get(key) for key in BAND_ASSETS.values()):
            continue
        scenes.append(
            Scene(
                item_id=feature["id"],
                datetime=props.get("datetime") or "",
                cloud_cover=float(props.get("eo:cloud_cover", 100.0)),
                grid_code=str(props.get("grid:code", "")),
                assets={k: v for k, v in assets.items() if v},
            )
        )
    scenes.sort(key=lambda s: s.cloud_cover)
    return scenes


def aoi_window_in_scene(href: str, spec: AoiSpec, resolution_m: int | None = None,
                        size_px: int | None = None) -> tuple[Window, object] | None:
    """Return the integer pixel window covering the AOI, or None if the AOI is not fully inside.

    ``resolution_m`` / ``size_px`` default to the analysis grid (10 m), but can be overridden
    for auxiliary assets stored at a coarser native resolution (e.g. SCL at 20 m).
    """
    resolution = resolution_m or spec.resolution_m
    expected = size_px or spec.size_px
    min_x, min_y, max_x, max_y = spec.bounds_utm()
    with rasterio.Env(**GDAL_ENV), rasterio.open(href) as src:
        if src.crs is None:
            return None
        if src.crs.to_epsg() != spec.target_epsg:
            min_x, min_y, max_x, max_y = transform_bounds(spec.target_epsg, src.crs, min_x, min_y, max_x, max_y)
        if abs(abs(src.transform.a) - resolution) > 1e-6:
            return None

        float_window = from_bounds(min_x, min_y, max_x, max_y, transform=src.transform)
        col_off = max(0, math.floor(float_window.col_off))
        row_off = max(0, math.floor(float_window.row_off))
        width = math.ceil(float_window.width) + 1
        height = math.ceil(float_window.height) + 1
        if col_off + width > src.width or row_off + height > src.height:
            return None
        if not (expected <= width <= expected + 2 and expected <= height <= expected + 2):
            return None
        inside = (
            src.bounds.left <= min_x and src.bounds.bottom <= min_y
            and src.bounds.right >= max_x and src.bounds.top >= max_y
        )
        if not inside:
            return None
        return Window(col_off=int(col_off), row_off=int(row_off), width=int(width), height=int(height)), src.crs



def read_band_window(href: str, window: Window) -> tuple[np.ndarray, dict]:
    with rasterio.Env(**GDAL_ENV), rasterio.open(href) as src:
        data = src.read(1, window=window)
        profile = {
            "crs": src.crs,
            "transform": src.window_transform(window),
            "nodata": src.nodata,
            "dtype": data.dtype.name,
            "source_bounds": tuple(src.bounds),
            "source_crs": src.crs.to_string() if src.crs else None,
        }
    return data, profile


def write_band(path: Path, data: np.ndarray, profile: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out_profile = {
        "driver": "GTiff",
        "height": data.shape[0],
        "width": data.shape[1],
        "count": 1,
        "dtype": data.dtype.name,
        "crs": profile["crs"],
        "transform": profile["transform"],
        "nodata": profile["nodata"],
        "compress": "deflate",
        "predictor": 2,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(data, 1)


def band_stats(data: np.ndarray, nodata) -> dict:
    valid = (data != nodata) if nodata is not None else np.ones(data.shape, dtype=bool)
    values = data[valid].astype("float64")
    if values.size == 0:
        return {"valid_pixels": 0}
    return {
        "valid_pixels": int(values.size),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": round(float(values.mean()), 3),
    }


def fetch_observation(
    client: httpx.Client,
    spec: AoiSpec,
    label: str,
    start: str,
    end: str,
    max_cloud: float,
    preferred_grid: str | None,
    out_root: Path,
) -> dict:
    scenes = search_scenes(client, spec.bounds_wgs84(), start, end, max_cloud)
    if not scenes:
        raise SystemExit(f"No {label} scenes found between {start} and {end} with cloud < {max_cloud}%")

    print(f"\n{label.upper()} candidates ({start} -> {end}, cloud < {max_cloud}%):")
    for scene in scenes[:8]:
        print(f"  {scene.date}  cloud={scene.cloud_cover:6.2f}%  tile={scene.grid_code:12s} {scene.item_id}")

    ordered = sorted(scenes, key=lambda s: (s.grid_code != preferred_grid if preferred_grid else False, s.cloud_cover))
    for scene in ordered:
        window_info = aoi_window_in_scene(scene.assets["red"], spec)
        if window_info is None:
            continue
        window, _ = window_info
        print(f"  selected: {scene.item_id} ({scene.date}, cloud {scene.cloud_cover:.2f}%, tile {scene.grid_code})")
        print(f"  AOI window: {window.width}x{window.height} px at offset ({window.col_off}, {window.row_off})")

        record: dict = {
            "item_id": scene.item_id,
            "acquisition_date": scene.date,
            "datetime": scene.datetime,
            "cloud_cover_percent": round(scene.cloud_cover, 4),
            "mgrs_grid": scene.grid_code,
            "bands": {},
            "aux": {},
            "window": {
                "col_off": int(window.col_off),
                "row_off": int(window.row_off),
                "width": int(window.width),
                "height": int(window.height),
            },
        }
        for band, asset_key in BAND_ASSETS.items():
            data, profile = read_band_window(scene.assets[asset_key], window)
            write_band(out_root / label / f"{band}.tif", data, profile)
            record["bands"][band] = {
                "asset": asset_key,
                "href": scene.assets[asset_key],
                "dtype": profile["dtype"],
                "nodata": profile["nodata"],
                "stats": band_stats(data, profile["nodata"]),
            }
            print(f"    {band}: {data.shape[1]}x{data.shape[0]} {profile['dtype']} -> {label}/{band}.tif")

        for aux, asset_key in AUX_ASSETS.items():
            if asset_key not in scene.assets:
                continue
            # SCL is a 20 m native asset: use its own resolution/size, never the 10 m window.
            aux_res = 20
            aux_size = spec.size_px * spec.resolution_m // aux_res
            aux_window_info = aoi_window_in_scene(scene.assets[asset_key], spec,
                                                 resolution_m=aux_res, size_px=aux_size)
            if aux_window_info is None:
                print(f"    {aux}: skipped (AOI not resolvable at {aux_res} m)")
                continue
            aux_window = aux_window_info[0]
            data, profile = read_band_window(scene.assets[asset_key], aux_window)
            write_band(out_root / label / f"{aux}.tif", data, profile)
            record["aux"][aux] = {
                "asset": asset_key,
                "href": scene.assets[asset_key],
                "dtype": profile["dtype"],
                "nodata": profile["nodata"],
                "resolution_m": aux_res,
                "window": {
                    "col_off": int(aux_window.col_off),
                    "row_off": int(aux_window.row_off),
                    "width": int(aux_window.width),
                    "height": int(aux_window.height),
                },
                "note": "auxiliary asset for future per-pixel cloud masking; not consumed yet",
            }
            print(f"    {aux}: {data.shape[1]}x{data.shape[0]} @ {aux_res} m -> {label}/{aux}.tif")
        return record


    raise SystemExit(f"No {label} scene fully contains the AOI between {start} and {end}.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--before-start", default="2023-12-01")
    parser.add_argument("--before-end", default="2023-12-31")
    parser.add_argument("--after-start", default="2024-05-01")
    parser.add_argument("--after-end", default="2024-06-20")
    parser.add_argument("--max-cloud", type=float, default=15.0, help="maximum scene cloud cover percent")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "raw" / "demo_area")
    parser.add_argument("--dry-run", action="store_true", help="search and print candidates only")
    args = parser.parse_args(argv)

    spec = AoiSpec()
    print("TerraGuard AI - Sentinel-2 pair acquisition")
    print(f"AOI  : centre {spec.center_lon}, {spec.center_lat} | {spec.edge_m} m square "
          f"| {spec.size_px}x{spec.size_px} px @ {spec.resolution_m} m | EPSG:{spec.target_epsg}")
    print(f"UTM bounds   : {spec.bounds_utm()}")
    print(f"WGS84 bounds : {tuple(round(v, 6) for v in spec.bounds_wgs84())}")

    with httpx.Client() as client:
        if args.dry_run:
            for label, start, end in (
                ("before", args.before_start, args.before_end),
                ("after", args.after_start, args.after_end),
            ):
                scenes = search_scenes(client, spec.bounds_wgs84(), start, end, args.max_cloud)
                print(f"\n{label.upper()} window {start}..{end}: {len(scenes)} candidate scenes")
                for scene in scenes[:10]:
                    ok = aoi_window_in_scene(scene.assets["red"], spec) is not None
                    print(f"  contains AOI: {str(ok):5s}  {scene.date}  cloud={scene.cloud_cover:6.2f}%  {scene.item_id}")
            return 0

        before = fetch_observation(client, spec, "before", args.before_start, args.before_end,
                                   args.max_cloud, None, args.out)
        after = fetch_observation(client, spec, "after", args.after_start, args.after_end,
                                  args.max_cloud, before["mgrs_grid"], args.out)

    provenance = {
        "source": "Element 84 Earth Search STAC (sentinel-2-l2a) / AWS Open Data sentinel-cogs",
        "stac_endpoint": STAC_SEARCH_URL,
        "collection": COLLECTION,
        "license_note": "Copernicus Sentinel data - free and open (Copernicus Programme).",
        "aoi": {
            "centre_lon": spec.center_lon,
            "centre_lat": spec.center_lat,
            "edge_m": spec.edge_m,
            "resolution_m": spec.resolution_m,
            "size_px": spec.size_px,
            "bounds_utm": spec.bounds_utm(),
            "bounds_wgs84": spec.bounds_wgs84(),
            "analysis_epsg": spec.target_epsg,
        },
        "bands": BAND_ASSETS,
        "aux_bands": AUX_ASSETS,
        "search_windows": {
            "before": [args.before_start, args.before_end],
            "after": [args.after_start, args.after_end],
        },
        "max_cloud_percent": args.max_cloud,
        "before": before,
        "after": after,
    }
    provenance_path = args.out / "acquisition.json"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    print(f"\nRaw pair written to {args.out}")
    print(f"Provenance written to {provenance_path}")
    print("Next: backend/.venv/Scripts/python.exe scripts/validate_satellite_pair.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())



