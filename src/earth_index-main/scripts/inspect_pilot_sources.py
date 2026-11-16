# /// script
# requires-python = ">=3.11"
# dependencies = ["pyproj==3.8.0", "shapely==2.2.0", "h3==4.5.0"]
# ///
"""Inventory public metadata; never download scientific raster payloads.

Run: uv run scripts/inspect_pilot_sources.py
Use --cached to replay the captured catalog responses without network requests.
"""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import h3
import pyproj
from pyproj import CRS, Geod, Transformer
import shapely
from shapely.geometry import Polygon, mapping, shape
from shapely.geometry.polygon import orient
from shapely.ops import unary_union


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "research" / "pilots"
CACHE = OUTPUT / "catalog-cache"
CMR = "https://cmr.earthdata.nasa.gov/search/"
STAC = "https://earth-search.aws.element84.com/v1/"
TEMPORAL = "2024-01-01T00:00:00Z,2024-12-31T23:59:59Z"
PRODUCTS = {
    "MOD11A2": ("061", False, 46),
    "MCD43A3": ("061", False, 366),
    "MCD12Q1": ("061", False, 1),
    "MOD13Q1": ("061", False, 23),
    "NASADEM_HGT": ("001", True, None),
}
SITES = [
    ("atacama", "Atacama / northern Chile", -68.5, -23.5),
    ("iceland", "Central Iceland", -18.0, 64.75),
    ("sahara", "Tassili / central Sahara", 10.5, 24.5),
]


def fetch_json(url, cached):
    path = CACHE / (hashlib.sha256(url.encode()).hexdigest() + ".json")
    if path.exists():
        return json.loads(path.read_text())
    if cached:
        raise RuntimeError(f"Missing cached request: {url}")
    for attempt in range(3):
        try:
            request = Request(url, headers={"User-Agent": "earth-index-pilot-inventory/1"})
            with urlopen(request, timeout=90) as response:
                record = {
                    "request_url": url,
                    "retrieved_at": datetime.now(timezone.utc).isoformat(),
                    "headers": {k.lower(): v for k, v in response.headers.items()},
                    "body": json.load(response),
                }
            path.write_text(json.dumps(record))
            return record
        except (HTTPError, OSError):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def footprint(site):
    key, name, lon, lat = site
    local = CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m"
    )
    to_wgs = Transformer.from_crs(local, 4326, always_xy=True)
    # Densify projected straight edges every 5 km before geographic conversion.
    corners = [(-50000, -50000), (50000, -50000), (50000, 50000), (-50000, 50000)]
    ring = []
    for i, a in enumerate(corners):
        b = corners[(i + 1) % 4]
        ring.extend((a[0] + (b[0] - a[0]) * j / 20,
                     a[1] + (b[1] - a[1]) * j / 20) for j in range(20))
    polygon = orient(shapely.transform(Polygon(ring), to_wgs.transform, interleaved=False), sign=1)
    assert polygon.is_valid
    area, _ = Geod(ellps="WGS84").geometry_area_perimeter(polygon)
    return {
        "key": key, "name": name, "center": [lon, lat],
        "geometry": mapping(polygon), "bbox": list(polygon.bounds),
        "projected_width_m": 100000, "projected_height_m": 100000,
        "construction_crs": local.to_string(), "geodesic_area_km2": abs(area) / 1e6,
    }


def grid_inventory(site):
    polygon = shape(site["geometry"])
    hshape = h3.geo_to_h3shape(site["geometry"])
    rows = []
    for resolution in (5, 6, 7):
        candidates = h3.h3shape_to_cells_experimental(hshape, resolution, "bbox_overlap")
        cells = []
        clipped = []
        for cell in candidates:
            cell_polygon = Polygon([(lon, lat) for lat, lon in h3.cell_to_boundary(cell)])
            intersection = cell_polygon.intersection(polygon)
            if not intersection.is_empty and intersection.area > 0:
                cells.append(cell)
                clipped.append(intersection)
        uncovered = polygon.difference(unary_union(clipped))
        # Geometry checks in degrees here verify topology, not metric area.
        assert uncovered.area / polygon.area < 1e-9, (site["key"], resolution)
        assert len(cells) == len(set(cells))
        rows.append({
            "resolution": resolution, "intersecting_cells": len(cells),
            "mean_full_cell_area_km2": sum(h3.cell_area(c, "km^2") for c in cells) / len(cells),
            "cell_ids": sorted(cells),
        })
    return rows


def collection(product, version, cached):
    url = CMR + "collections.umm_json?" + urlencode({
        "short_name": product, "version": version, "page_size": 10,
    })
    response = fetch_json(url, cached)
    items = [x for x in response["body"]["items"]
             if x["meta"].get("provider-id") == "LPCLOUD"]
    if len(items) != 1:
        raise RuntimeError(f"Expected one LPCLOUD collection for {product}: {len(items)}")
    item = items[0]
    return {
        "concept_id": item["meta"]["concept-id"],
        "revision_id": item["meta"]["revision-id"],
        "metadata": item["umm"], "request_url": url,
        "retrieved_at": response["retrieved_at"],
    }


def granule_inventory(site, product, coll, static, expected, cached):
    params = {
        "collection_concept_id": coll["concept_id"], "page_size": 2000,
        "polygon": ",".join(f"{n:.6f}" for p in site["geometry"]["coordinates"][0] for n in p),
        "sort_key[]": ["start_date", "producer_granule_id"],
    }
    if not static:
        params["temporal"] = TEMPORAL
    entries, requests, page, retrieved = [], [], 1, []
    while True:
        params["page_num"] = page
        url = CMR + "granules.json?" + urlencode(params, doseq=True)
        response = fetch_json(url, cached)
        requests.append(url)
        retrieved.append(response["retrieved_at"])
        hits = int(response["headers"]["cmr-hits"])
        chunk = response["body"]["feed"]["entry"]
        entries.extend(chunk)
        if len(entries) >= hits:
            break
        if not chunk:
            raise RuntimeError("Catalog pagination stopped before advertised hit count")
        page += 1
    assert len(entries) == hits
    assert len({e["id"] for e in entries}) == len(entries)
    periods = Counter(e.get("time_start") for e in entries)
    tiles = sorted({part for e in entries for part in e["title"].split(".")
                    if len(part) == 6 and part.startswith("h") and part[3] == "v"})
    downloadable = [link["href"] for e in entries for link in e.get("links", [])
                    if link.get("rel", "").endswith("/data#")
                    and link.get("href", "").startswith("https://")
                    and not link.get("inherited", False)]
    manifest = []
    for entry in entries:
        match = re.search(r"\.A(\d{7})\.", entry["title"])
        nominal_date = datetime.strptime(match[1], "%Y%j").date().isoformat() if match else None
        manifest.append({
            "id": entry["id"], "title": entry["title"],
            "nominal_product_date": nominal_date,
            "time_start": entry.get("time_start"), "time_end": entry.get("time_end"),
            "granule_size_mb": entry.get("granule_size"),
        })
    nominal_dates = {g["nominal_product_date"] for g in manifest
                     if g["nominal_product_date"] and g["nominal_product_date"].startswith("2024-")}
    return {
        "product": product, "collection_concept_id": coll["concept_id"],
        "granule_count": hits, "modis_tiles": tiles,
        "distinct_start_times": len(periods), "expected_periods_in_2024": expected,
        "distinct_starts_in_2024": sum(t.startswith("2024-") for t in periods),
        "distinct_product_dates_2024": len(nominal_dates) if not static else None,
        "granules_with_product_date_2024": sum(
            g["nominal_product_date"] in nominal_dates for g in manifest
        ) if not static else None,
        "date_semantics": "Nominal product date is parsed from the provider ID; CMR start/end describe temporal support and can cross year boundaries.",
        "granules_per_start_min": min(periods.values(), default=0),
        "granules_per_start_max": max(periods.values(), default=0),
        "whole_granule_archive_size_mb": sum(float(e.get("granule_size", 0)) for e in entries),
        "first_start": min(periods, default=None), "last_start": max(periods, default=None),
        "example_download_urls": list(dict.fromkeys(downloadable))[:2],
        "query_urls": requests, "retrieved_at": retrieved,
        "availability_scope": "Catalog footprint intersection; valid pixel coverage not measured",
        "granules": manifest,
    }


def dem_inventory(site, cached):
    url = STAC + "search?" + urlencode({
        "collections": "cop-dem-glo-30", "bbox": ",".join(map(str, site["bbox"])), "limit": 100,
    })
    response = fetch_json(url, cached)
    features = list(response["body"]["features"])
    body = response["body"]
    visited = {url}
    # This server may emit a next link even when fewer than limit items remain.
    while body["features"]:
        next_url = next((x["href"] for x in body.get("links", []) if x.get("rel") == "next"), None)
        if not next_url:
            break
        if next_url in visited:
            raise RuntimeError("STAC pagination loop")
        visited.add(next_url)
        body = fetch_json(next_url, cached)["body"]
        features.extend(body["features"])
    assert len({f["id"] for f in features}) == len(features)
    polygon = shape(site["geometry"])
    selected = [f for f in features if shape(f["geometry"]).intersects(polygon)]
    coverage = unary_union([shape(f["geometry"]) for f in selected])
    ratio = polygon.intersection(coverage).area / polygon.area
    return {
        "collection": "cop-dem-glo-30", "item_count": len(selected),
        "catalog_footprint_coverage_fraction": ratio,
        "coverage_method": "Union of STAC footprints in lon/lat; topology indicator, not valid DEM pixels",
        "query_url": url, "retrieved_at": response["retrieved_at"],
        "items": [{"id": f["id"], "bbox": f["bbox"], "assets": f["assets"]} for f in selected],
    }


def probe_access(sites):
    """Read at most four file-header bytes, never raster measurement arrays."""
    site = sites[0]
    urls = [(product, site["products"][product]["example_download_urls"][0])
            for product in ("MOD11A2", "NASADEM_HGT")]
    asset = urlparse(site["products"]["cop-dem-glo-30"]["items"][0]["assets"]["data"]["href"])
    urls.append(("cop-dem-glo-30", f"https://{asset.netloc}.s3.amazonaws.com{asset.path}"))
    results = []
    for product, url in urls:
        result = {"product": product, "url": url,
                  "checked_at": datetime.now(timezone.utc).isoformat()}
        try:
            with urlopen(Request(url, headers={"Range": "bytes=0-3"}), timeout=30) as response:
                result.update({
                    "status": response.status, "final_host": urlparse(response.url).netloc,
                    "content_type": response.headers.get("Content-Type"),
                    "content_range": response.headers.get("Content-Range"),
                    "header_bytes_hex": response.read(4).hex(),
                })
        except HTTPError as error:
            result.update({"status": error.code, "final_host": urlparse(error.url).netloc})
        results.append(result)
    return results


def index_plan(sites):
    cells = sum(next(g["intersecting_cells"] for g in s["grids"] if g["resolution"] == 6)
                for s in sites)
    groups = [
        {"group": "terrain", "source": "cop-dem-glo-30", "periods": 1,
         "features": ["elevation_min_m", "elevation_mean_m", "elevation_max_m", "elevation_stddev_m"],
         "method": "Aggregate native DEM samples using documented cell-overlap weights; slope requires a separate, specified derivative operator."},
        {"group": "surface_temperature", "source": "MOD11A2", "periods": 46,
         "features": ["surface_temperature_day_k", "surface_temperature_night_k"],
         "method": "Keep source 8-day composites; aggregate quality-screened source pixels per cell separately for day and night."},
        {"group": "albedo", "source": "MCD43A3", "periods": 366,
         "features": ["shortwave_black_sky_albedo", "shortwave_white_sky_albedo"],
         "method": "Keep distinct albedo definitions and source 16-day support; daily outputs are temporally overlapping."},
        {"group": "vegetation", "source": "MOD13Q1", "periods": 23,
         "features": ["ndvi", "evi"],
         "method": "Quality-screened spatial summaries of the source 16-day composites."},
        {"group": "land_cover", "source": "MCD12Q1", "periods": 1,
         "features": ["dominant_igbp_land_cover"],
         "method": "Compute area fractions under the versioned IGBP vocabulary; store dominant class plus its fraction."},
    ]
    for group in groups:
        group["maximum_value_rows"] = cells * group["periods"] * len(group["features"])
    return {
        "status": "Earlier fixed-selection feasibility sketch, not a selected processing cohort",
        "current_discovery_registry": "../variables/candidates.json",
        "h3_resolution": 6, "intersecting_cells": cells,
        "feature_groups": groups,
        "maximum_value_rows": sum(g["maximum_value_rows"] for g in groups),
        "row_count_notes": "Upper bound for this fixed feature selection, including explicit missing rows. Quality, area fractions, source manifests and audit records add storage. Boundary cells require clipping or support labels; do not extrapolate clipped pilot measurements to a full H3 cell.",
        "metadata_per_value": ["source_series", "source_assets", "feature_version", "cell_id",
                               "observation_window", "canonical_unit", "valid_fraction",
                               "quality_flags", "missing_reason", "processing_version"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cached", action="store_true")
    parser.add_argument("--probe-access", action="store_true",
                        help="Check three anonymous file access paths using only four header bytes")
    args = parser.parse_args()
    if args.cached and args.probe_access:
        parser.error("--cached cannot be combined with network access probes")
    CACHE.mkdir(parents=True, exist_ok=True)
    sites = [footprint(site) for site in SITES]
    collections = {}
    for product, (version, _, _) in PRODUCTS.items():
        collections[product] = collection(product, version, args.cached)
        print(f"Collection: {product} {collections[product]['concept_id']}", flush=True)
    for site in sites:
        site["grids"] = grid_inventory(site)
        site["products"] = {}
    futures = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for site in sites:
            for product, (_, static, expected) in PRODUCTS.items():
                future = pool.submit(granule_inventory, site, product, collections[product], static, expected, args.cached)
                futures[future] = (site, product)
            futures[pool.submit(dem_inventory, site, args.cached)] = (site, "cop-dem-glo-30")
        for future in as_completed(futures):
            site, product = futures[future]
            site["products"][product] = future.result()
            row = site["products"][product]
            print(f"{site['key']} {product}: {row.get('granule_count', row.get('item_count'))} catalog records", flush=True)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "analysis_period": TEMPORAL, "scientific_rasters_downloaded": False,
        "dependencies": {"pyproj": pyproj.__version__, "shapely": shapely.__version__, "h3": h3.__version__},
        "collections": collections, "sites": sites,
        "limitations": [
            "Catalog presence does not establish valid pixels, usable quality or scientific comparability.",
            "Archive sizes describe entire overlapping granules, not clipped 100 km squares.",
            "H3 counts include cells partially intersecting each square; analysis must clip/weight boundaries.",
            "Squares are 100 x 100 km in a local WGS84 azimuthal equidistant projection; geodesic area is reported separately.",
            "No PostgreSQL database or scientific measurement index has been populated by this script.",
        ],
    }
    (OUTPUT / "inventory.json").write_text(json.dumps(report, indent=2) + "\n")
    features = [{"type": "Feature", "geometry": s["geometry"], "properties": {
        "key": s["key"], "name": s["name"], "center": s["center"],
        "geodesic_area_km2": s["geodesic_area_km2"],
    }} for s in sites]
    (OUTPUT / "areas.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": features}, indent=2) + "\n")
    (OUTPUT / "index-plan.json").write_text(json.dumps(index_plan(sites), indent=2) + "\n")
    if args.probe_access:
        (OUTPUT / "access-checks.json").write_text(json.dumps(probe_access(sites), indent=2) + "\n")
    print(f"Wrote {OUTPUT / 'inventory.json'}", flush=True)


if __name__ == "__main__":
    main()
