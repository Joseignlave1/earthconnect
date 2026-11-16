# /// script
# requires-python = ">=3.11"
# dependencies = ["rasterio==1.5.2", "numpy==2.5.4", "exactextract==0.3.0", "shapely==2.2.0", "pyproj==3.8.0", "h3==4.5.0"]
# ///
"""Read remote DEM windows and retain only pilot cell summaries and provenance.

Run: uv run scripts/process_pilot_terrain.py
This is a bounded research ingestion, not a published scientific release.
"""

import argparse
import csv
import hashlib
import json
import math
import resource
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import h3
import numpy as np
import rasterio
from exactextract import exact_extract
from exactextract.raster import NumPyRasterSource
from pyproj import Geod
from rasterio.windows import Window, from_bounds
from rasterio.windows import bounds as window_bounds
from shapely.geometry import Polygon, box, mapping, shape
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research" / "pilots" / "terrain"
GEOD = Geod(ellps="WGS84")
METHOD = "native-post-support-fractional-spherical-area-horn3x3-v1"
WKT = rasterio.crs.CRS.from_epsg(4326).to_wkt()
STATS = [
    "mean(coverage_weight=area_spherical_m2)",
    "stdev(coverage_weight=area_spherical_m2)",
    "min",
    "max",
    "area=count(coverage_weight=area_spherical_m2)",
    "total=count(default_value=0,coverage_weight=area_spherical_m2)",
]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def area_m2(geometry):
    # Our pilot geometries are small, valid polygons without holes.
    if geometry.is_empty:
        return 0.0
    if geometry.geom_type in ("MultiPolygon", "GeometryCollection"):
        return sum(area_m2(g) for g in geometry.geoms)
    return abs(GEOD.geometry_area_perimeter(geometry)[0])


def https_url(uri):
    parsed = urlparse(uri)
    if parsed.scheme == "s3" and parsed.netloc == "copernicus-dem-30m":
        return f"https://{parsed.netloc}.s3.amazonaws.com{parsed.path}"
    if (
        parsed.scheme == "https"
        and parsed.netloc == "copernicus-dem-30m.s3.amazonaws.com"
    ):
        return uri
    raise ValueError("Unexpected terrain source host")


def target_cells(site, resolution):
    region = shape(site["geometry"])
    grid = next(g for g in site["grids"] if g["resolution"] == resolution)
    features = []
    for cell in grid["cell_ids"]:
        whole = Polygon([(lon, lat) for lat, lon in h3.cell_to_boundary(cell)])
        clipped = whole.intersection(region)
        features.append(
            {
                "type": "Feature",
                "geometry": mapping(clipped),
                "properties": {
                    "site": site["key"],
                    "cell_id": cell,
                    "h3_resolution": resolution,
                    "target_area_m2": area_m2(clipped),
                    "fraction_of_full_cell": area_m2(clipped) / area_m2(whole),
                    "spatial_support": "h3_cell_intersection_with_pilot_region",
                },
            }
        )
    assert len(features) == len({f["properties"]["cell_id"] for f in features})
    assert (
        region.difference(unary_union([shape(f["geometry"]) for f in features])).area
        < 1e-10
    )
    return features


def summarize(values, bounds, features):
    raster = NumPyRasterSource(values, *bounds, nodata=np.nan, srs_wkt=WKT)
    return exact_extract(raster, features, STATS, include_cols=["cell_id"])


def derivatives(z, transform):
    """Horn slope/aspect and 3x3 height variability at native posting intervals.

    Gradients use geodesic row-specific metre spacing. Asset edge pixels and any
    neighborhood containing nodata are excluded, never padded with false values.
    """
    center = z[1:-1, 1:-1]
    parts = [
        z[r : r + center.shape[0], c : c + center.shape[1]]
        for r in range(3)
        for c in range(3)
    ]
    valid = np.logical_and.reduce([np.isfinite(p) for p in parts])
    lat = transform.f + (np.arange(1, z.shape[0] - 1) + 0.5) * transform.e
    lon = np.full(lat.shape, transform.c + transform.a * z.shape[1] / 2)
    dx = GEOD.inv(lon - transform.a / 2, lat, lon + transform.a / 2, lat)[2][:, None]
    dy = GEOD.inv(lon, lat - abs(transform.e) / 2, lon, lat + abs(transform.e) / 2)[2][
        :, None
    ]
    a, b, c, d, _center, f, g, h, i = parts
    east = ((c + 2 * f + i) - (a + 2 * d + g)) / (8 * dx)
    south = ((g + 2 * h + i) - (a + 2 * b + c)) / (8 * dy)
    magnitude = np.hypot(east, south)
    slope = np.degrees(np.arctan(magnitude))
    delta_sum = np.zeros(center.shape, dtype="float64")
    delta_sq = np.zeros_like(delta_sum)
    lo, hi = center.copy(), center.copy()
    for part in parts:
        delta = part.astype("float64") - center
        delta_sum += delta
        delta_sq += delta * delta
        np.minimum(lo, part, out=lo)
        np.maximum(hi, part, out=hi)
    tpi = -delta_sum / 9
    roughness = np.sqrt(np.maximum(0, delta_sq / 9 - (delta_sum / 9) ** 2))
    nonflat = valid & (slope >= 0.1)

    def padded(values, mask=valid):
        result = np.full(z.shape, np.nan, dtype="float32")
        result[1:-1, 1:-1] = np.where(mask, values, np.nan)
        return result

    with np.errstate(divide="ignore", invalid="ignore"):
        return {
            "slope": padded(slope),
            "roughness_3x3": padded(roughness),
            "topographic_position_3x3": padded(tpi),
            "local_relief_3x3": padded(hi - lo),
            "aspect_eastness": padded(-east / magnitude, nonflat),
            "aspect_northness": padded(south / magnitude, nonflat),
        }


def merge_stats(acc, row):
    """Merge disjoint asset summaries using stable weighted central moments."""
    p = row["properties"]
    acc["total"] += p["total"]
    weight = p["area"]
    if weight <= 0:
        return
    old = acc["area"]
    new = old + weight
    delta = p["mean"] - acc["mean"]
    acc["m2"] += p["stdev"] ** 2 * weight + delta**2 * old * weight / new
    acc["mean"] += delta * weight / new
    acc["area"] = new
    acc["min"] = min(acc["min"], p["min"])
    acc["max"] = max(acc["max"], p["max"])


def accumulator():
    return {
        "total": 0.0,
        "area": 0.0,
        "mean": 0.0,
        "m2": 0.0,
        "min": math.inf,
        "max": -math.inf,
    }


def process_site(site, resolution):
    started = time.monotonic()
    features = target_cells(site, resolution)
    aggregates, assets = {}, []
    owned = Polygon()
    region = shape(site["geometry"])
    for item in sorted(
        site["products"]["cop-dem-glo-30"]["items"], key=lambda x: x["id"]
    ):
        uri = https_url(item["assets"]["data"]["href"])
        print(f"{site['key']}: reading source window {item['id']}", flush=True)
        request = Request(
            uri, method="HEAD", headers={"User-Agent": "earth-index-pilot/1"}
        )
        with urlopen(request, timeout=45) as response:
            headers = {k.lower(): v for k, v in response.headers.items()}
        with rasterio.open(uri) as src:
            if src.crs.to_epsg() != 4326 or src.transform.b or src.transform.d:
                raise ValueError("Only north-up WGS84 source rasters are supported")
            available = box(*src.bounds).difference(owned)
            overlap = region.intersection(available)
            if overlap.is_empty or overlap.area <= 0:
                continue
            fractional = from_bounds(*overlap.bounds, transform=src.transform)
            left = max(0, math.floor(fractional.col_off) - 1)
            top = max(0, math.floor(fractional.row_off) - 1)
            right = min(src.width, math.ceil(fractional.col_off + fractional.width) + 1)
            bottom = min(
                src.height, math.ceil(fractional.row_off + fractional.height) + 1
            )
            window = Window(left, top, right - left, bottom - top)
            values = (
                src.read(1, window=window, masked=True).astype("float32").filled(np.nan)
            )
            values[~np.isfinite(values)] = np.nan
            transform = src.window_transform(window)
            bounds = window_bounds(window, src.transform)
            owned = unary_union([owned, box(*src.bounds)])
            clipped = []
            for feature in features:
                geometry = shape(feature["geometry"]).intersection(available)
                if not geometry.is_empty and geometry.area > 0:
                    clipped.append({**feature, "geometry": mapping(geometry)})
            entry = {
                "asset_id": item["id"],
                "uri": uri,
                "etag": headers.get("etag"),
                "etag_is_content_checksum": False,
                "last_modified": headers.get("last-modified"),
                "s3_version_id": headers.get("x-amz-version-id"),
                "upstream_file_bytes": int(headers["content-length"]),
                "source_shape": list(src.shape),
                "block_shapes": src.block_shapes,
                "source_transform": list(src.transform),
                "source_tags": src.tags(),
                "source_nodata": src.nodata,
                "read_window": [left, top, right - left, bottom - top],
                "read_shape": list(values.shape),
                "decoded_window_bytes": values.nbytes,
                "decoded_float32_window_sha256": hashlib.sha256(
                    values.tobytes()
                ).hexdigest(),
                "window_transform": list(transform),
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
            }
        layers = {"elevation": values, **derivatives(values, transform)}
        for name, array in layers.items():
            rows = summarize(array, bounds, clipped)
            for row in rows:
                key = (row["properties"]["cell_id"], name)
                merge_stats(aggregates.setdefault(key, accumulator()), row)
        assets.append(entry)
        # No raster file or persistent decoded-array cache is written.
        del layers, array, values
    uncovered = region.difference(owned)
    if uncovered.area / region.area > 1e-9:
        raise ValueError(f"Incomplete raster footprints for {site['key']}")
    rows = []
    selected = {
        "elevation": ("m", ("min", "mean", "max", "stddev")),
        "slope": ("degree", ("mean", "stddev", "max")),
        "roughness_3x3": ("m", ("mean",)),
        "topographic_position_3x3": ("m", ("mean",)),
        "local_relief_3x3": ("m", ("mean",)),
        "aspect_eastness": ("1", ("mean",)),
        "aspect_northness": ("1", ("mean",)),
    }
    for feature in features:
        p = feature["properties"]
        for layer, (unit, statistics) in selected.items():
            acc = aggregates.get((p["cell_id"], layer), accumulator())
            for stat in statistics:
                value = (
                    None
                    if not acc["area"]
                    else (
                        math.sqrt(max(0, acc["m2"] / acc["area"]))
                        if stat == "stddev"
                        else acc[stat]
                    )
                )
                rows.append(
                    {
                        "site": site["key"],
                        "cell_id": p["cell_id"],
                        "h3_resolution": resolution,
                        "feature_key": f"terrain.{layer}.{stat}",
                        "value": value,
                        "unit": unit,
                        "time_kind": "static",
                        "valid_area_fraction": acc["area"] / acc["total"]
                        if acc["total"]
                        else 0.0,
                        "target_area_m2": p["target_area_m2"],
                        "fraction_of_full_cell": p["fraction_of_full_cell"],
                        "method_version": METHOD,
                        "source": "cop-dem-glo-30",
                        "quality_status": "nodata_only_screening_auxiliary_quality_masks_unavailable",
                    }
                )
    assert len(rows) == len(features) * 12
    path = OUT / f"{site['key']}.csv"
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)
    dump(
        OUT / f"{site['key']}-cells.geojson",
        {"type": "FeatureCollection", "features": features},
    )
    report = {
        "site": site["key"],
        "cells": len(features),
        "observation_rows": len(rows),
        "wall_seconds": time.monotonic() - started,
        "assets": assets,
        "csv_bytes": path.stat().st_size,
        "csv_sha256": sha256(path),
        "missing_observation_rows": sum(r["value"] is None for r in rows),
        "minimum_valid_area_fraction": min(r["valid_area_fraction"] for r in rows),
    }
    dump(OUT / f"{site['key']}-run.json", report)
    print(
        f"{site['key']}: {len(rows)} summaries in {report['wall_seconds']:.1f}s",
        flush=True,
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--site", choices=["atacama", "iceland", "sahara"], action="append"
    )
    parser.add_argument("--resolution", type=int, choices=[5, 6, 7], default=6)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    inventory = ROOT / "research" / "pilots" / "inventory.json"
    data = json.loads(inventory.read_text())
    start = time.monotonic()
    with rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
        GDAL_CACHEMAX=128 * 1024 * 1024,
        CPL_VSIL_CURL_CACHE_SIZE=64 * 1024 * 1024,
        GDAL_HTTP_TIMEOUT="45",
        GDAL_HTTP_MAX_RETRY="2",
        GDAL_HTTP_RETRY_DELAY="1",
    ):
        reports = [
            process_site(s, args.resolution)
            for s in data["sites"]
            if not args.site or s["key"] in args.site
        ]
    report = {
        "status": "research_pilot_not_published_release",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method_version": METHOD,
        "h3_resolution": args.resolution,
        "script_sha256": sha256(Path(__file__)),
        "inventory_sha256": sha256(inventory),
        "dependencies": {
            "rasterio": rasterio.__version__,
            "gdal": rasterio.__gdal_version__,
            "numpy": np.__version__,
            "h3": h3.__version__,
            "exactextract": "0.3.0",
        },
        "total_wall_seconds": time.monotonic() - start,
        "peak_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / 1024,
        "sites": reports,
        "source_raster_files_retained": 0,
        "raster_scratch_bytes": 0,
        "network_bytes": None,
        "notes": [
            "HTTP range reads; decoded window bytes are not measured network transfer.",
            "Native elevation postings are treated as representative pixel-area supports; no elevation resampling.",
            "Exact planar polygon-pixel fractions with spherical pixel-area weights, not exact ellipsoidal integration.",
            "H3 boundary cells are summarized only over their pilot-region intersection, not the full H3 cell.",
            "Overlapping asset supports are assigned in lexicographic asset-ID order to avoid double counting.",
            "Horn 3x3 slope uses geodesic horizontal posting distances; asset seams/nodata neighborhoods are excluded.",
            "Aspect components are downhill eastness/northness; slopes below 0.1 degree are excluded.",
            "Roughness is 3x3 population height standard deviation; TPI is center minus 3x3 mean; relief is 3x3 range.",
            "Elevation is a mixed-epoch digital surface model, metres relative to EGM2008, not a 2024 observation.",
            "Only nodata/finite checks applied; editing/filling/error/water quality layers were not supplied by these assets.",
            "Decoded window hashes do not authenticate the whole upstream file; full-asset checksums remain unknown.",
            "Recomputation depends on continued availability of the provider assets.",
        ],
    }
    dump(OUT / ("run.json" if not args.site else "selected-run.json"), report)


if __name__ == "__main__":
    main()
