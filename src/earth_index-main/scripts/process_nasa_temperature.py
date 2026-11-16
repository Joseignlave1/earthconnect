# /// script
# requires-python = ">=3.11"
# dependencies = ["rasterio==1.5.2", "numpy==2.5.4", "exactextract==0.3.0", "shapely==2.2.0", "pyproj==3.8.0", "h3==4.5.0"]
# ///
"""Inspect or aggregate NASA area subsets in memory, retaining no raster files."""

import argparse
import csv
import hashlib
import json
import re
import resource
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, build_opener

import numpy as np
import shapely
from exactextract import exact_extract
from exactextract.raster import NumPyRasterSource
from prepare_nasa_subsets import API, NoRedirect, login_credentials, logout, request
from prepare_nasa_subsets import OUT as REQUESTS
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.io import MemoryFile
from rasterio.transform import Affine, array_bounds
from shapely.geometry import mapping, shape

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research/pilots/temperature"
PATTERN = re.compile(
    r"MOD11A2\.061_(LST_Day_1km|LST_Night_1km|QC_Day|QC_Night|Emis_31|Emis_32|Clear_sky_days|Clear_sky_nights)_doy(\d{7})(?:\d{6})?_(.+)\.tif$"
)
NEEDED = {"LST_Day_1km", "LST_Night_1km", "QC_Day", "QC_Night"}
METHOD = "mod11a2-native-strict-qc0-fractional-area-v1"


def safe_bundle(bundle):
    keys = ("file_id", "file_name", "file_size", "file_type", "sha256")
    return {
        "task_id": bundle["task_id"],
        "bundle_type": bundle.get("bundle_type"),
        "files": [
            {key: info[key] for key in keys if key in info} for info in bundle["files"]
        ],
    }


def download_file(task_id, info, bearer):
    """Follow NASA's signed-S3 redirect without forwarding the bearer token."""
    if info["file_size"] > 100 * 1024 * 1024:
        raise RuntimeError("Subset file exceeds the 100 MiB per-file safety bound")
    endpoint = f"{API}/bundle/{task_id}/{info['file_id']}"
    for attempt in range(3):
        try:
            req = Request(
                endpoint,
                headers={"Authorization": bearer, "User-Agent": "earth-index-pilot/1"},
            )
            opener = build_opener(NoRedirect)
            try:
                response = opener.open(req, timeout=60)
            except HTTPError as e:
                if e.code != 302:
                    raise RuntimeError(
                        f"NASA subset download returned HTTP {e.code}"
                    ) from None
                location = e.headers["Location"]
                parsed = urlparse(location)
                if (
                    parsed.scheme != "https"
                    or not parsed.hostname
                    or not parsed.hostname.endswith(".amazonaws.com")
                ):
                    raise RuntimeError("Unexpected NASA subset redirect host") from None
                # Signed URL is never logged or persisted; do not forward Authorization.
                response = opener.open(
                    Request(location, headers={"User-Agent": "earth-index-pilot/1"}),
                    timeout=60,
                )
            with response:
                data = response.read(info["file_size"] + 1)
            if len(data) != info["file_size"]:
                raise RuntimeError("Subset byte length differs from its manifest")
            if (
                not info.get("sha256")
                or hashlib.sha256(data).hexdigest() != info["sha256"]
            ):
                raise RuntimeError("Subset SHA-256 verification failed")
            return data
        except (HTTPError, URLError, TimeoutError):
            if attempt == 2:
                raise RuntimeError(
                    "Subset transfer failed; signed URLs and credentials omitted"
                ) from None
            time.sleep(attempt + 1)
    raise AssertionError("unreachable")


def decode_file(data, info):
    with MemoryFile(data) as memory, memory.open() as src:
        values = src.read(1, masked=False)
        keys = {
            "AREA_OR_POINT",
            "units",
            "long_name",
            "scale_factor",
            "add_offset",
            "_FillValue",
            "valid_range",
            "valid_min",
            "valid_max",
        }
        nodata = src.nodata
        if nodata is not None and not np.isfinite(nodata):
            nodata = str(nodata)
        finite = values[np.isfinite(values)]
        metadata = {
            "file_name": info["file_name"],
            "file_id": info["file_id"],
            "file_size": len(data),
            "sha256": info["sha256"],
            "dtype": str(values.dtype),
            "shape": list(values.shape),
            "crs_wkt": src.crs.to_wkt(),
            "transform": list(src.transform),
            "nodata": nodata,
            "scales": list(src.scales),
            "offsets": list(src.offsets),
            "units": list(src.units),
            "tags": {k: v for k, v in src.tags().items() if k in keys},
            "band_tags": {k: v for k, v in src.tags(1).items() if k in keys},
            "tag_names": list(src.tags()),
            "band_tag_names": list(src.tags(1)),
            "minimum": float(finite.min()) if finite.size else None,
            "maximum": float(finite.max()) if finite.size else None,
        }
    return values, metadata


def file_groups(bundle):
    groups = {}
    for info in bundle["files"]:
        match = PATTERN.search(info["file_name"])
        if not match or match[1] not in NEEDED:
            continue
        layer, day, area = match.groups()
        group = groups.setdefault((day, area), {})
        if layer in group:
            raise RuntimeError("Duplicate layer for the same date and area")
        group[layer] = info
    if any(set(group) != NEEDED for group in groups.values()):
        raise RuntimeError("Missing temperature or QA layer in NASA bundle")
    return groups


def inspect_site(receipt, bearer):
    site = receipt["site"]
    task = request("/task/" + receipt["task_id"], authorization=bearer)
    if task["status"] != "done":
        print(f"{site}: NASA task {task['status']}", flush=True)
        return {"site": site, "status": task["status"]}
    bundle = request("/bundle/" + receipt["task_id"], authorization=bearer)
    # Bundle metadata contains file hashes; do not persist the account/task response.
    (OUT / f"{site}-bundle.json").write_text(
        json.dumps(safe_bundle(bundle), indent=2) + "\n"
    )
    groups = file_groups(bundle)
    if not groups:
        raise RuntimeError(
            "Completed NASA task returned no matching temperature/QA files"
        )
    first = min(groups)

    def fetch(layer):
        info = groups[first][layer]
        return layer, decode_file(
            download_file(receipt["task_id"], info, bearer), info
        )[1]

    with ThreadPoolExecutor(max_workers=4) as pool:
        headers = dict(pool.map(fetch, sorted(NEEDED)))
    result = {
        "site": site,
        "status": "headers_inspected",
        "date_area_groups": len(groups),
        "first_group": first,
        "headers": headers,
    }
    (OUT / f"{site}-headers.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(f"{site}: {len(groups)} date/area groups; first header inspected", flush=True)
    for key, value in headers.items():
        print(
            key,
            {
                k: value[k]
                for k in (
                    "dtype",
                    "shape",
                    "nodata",
                    "scales",
                    "offsets",
                    "units",
                    "minimum",
                    "maximum",
                )
            },
            flush=True,
        )
    return result


def kelvin(values, metadata):
    """Decode exactly once using verified packed/native vs AppEEARS float formats."""
    scale = metadata["scales"][0]
    offset = metadata["offsets"][0]
    if values.dtype == np.dtype("uint16"):
        if not (np.isclose(scale, 1) or np.isclose(scale, 0.02)) or offset != 0:
            raise ValueError("Unexpected packed MOD11A2 scale/offset")
        result = values.astype("float64") * 0.02
        valid = values >= 7500
    elif np.issubdtype(values.dtype, np.floating):
        # The AppEEARS layer catalog declares physical float32 Kelvin outputs.
        # Nondefault scaling or packed-looking float values require manual review.
        if scale != 1 or offset != 0:
            raise ValueError("Unexpected floating MOD11A2 scale/offset")
        result = values.astype("float64")
        valid = np.isfinite(result) & (result >= 150) & (result <= 1310.7)
        if np.any(np.isfinite(result) & (result >= 7500)):
            raise ValueError(
                "Packed-looking floating temperature requires header review"
            )
    else:
        raise ValueError("Unsupported MOD11A2 temperature dtype")
    return np.where(valid, result, np.nan)


def quality_screen(values, qa):
    # QA=0 is GOOD; never apply a nodata=0 raster mask to this bitfield.
    if qa.dtype != np.dtype("uint8"):
        raise ValueError("Expected the native uint8 MOD11A2 quality bitfield")
    return np.where((qa == 0) & np.isfinite(values), values, np.nan)


def composite_window(day):
    start = datetime.strptime(day, "%Y%j").replace(tzinfo=timezone.utc)
    if not day.startswith("2024"):
        raise ValueError("Only nominal 2024 composites belong in this pilot")
    end = min(
        start + timedelta(days=8), datetime(start.year + 1, 1, 1, tzinfo=timezone.utc)
    )
    return start.isoformat(), end.isoformat()


def aligned(decoded):
    reference = decoded["LST_Day_1km"][1]
    for _, metadata in decoded.values():
        if (
            metadata["shape"] != reference["shape"]
            or metadata["transform"] != reference["transform"]
            or CRS.from_wkt(metadata["crs_wkt"]) != CRS.from_wkt(reference["crs_wkt"])
        ):
            raise ValueError("Temperature and quality grids are not aligned")
    crs = CRS.from_wkt(reference["crs_wkt"])
    if not crs.is_projected or crs.to_dict().get("proj") != "sinu":
        raise ValueError(
            "Only native equal-area MODIS sinusoidal subsets are supported"
        )
    return reference


def projected_cells(site, crs_wkt):
    cells = json.loads(
        (ROOT / f"research/pilots/terrain/{site}-cells.geojson").read_text()
    )["features"]
    transformer = Transformer.from_crs(4326, CRS.from_wkt(crs_wkt), always_xy=True)
    result = []
    for cell in cells:
        # Densify geographic straight edges to limit projection chord error.
        geometry = shapely.transform(
            shapely.segmentize(shape(cell["geometry"]), 0.001),
            transformer.transform,
            interleaved=False,
        )
        result.append(
            {
                "type": "Feature",
                "geometry": mapping(geometry),
                "properties": {
                    **cell["properties"],
                    "projected_target_area_m2": geometry.area,
                },
            }
        )
    return result


def aggregate_arrays(decoded, cells):
    metadata = aligned(decoded)
    day = quality_screen(kelvin(*decoded["LST_Day_1km"]), decoded["QC_Day"][0])
    night = quality_screen(kelvin(*decoded["LST_Night_1km"]), decoded["QC_Night"][0])
    # Difference is calculated on paired valid pixels, not mismatched cell means.
    arrays = {"day_k": day, "night_k": night, "paired_day_minus_night_k": day - night}
    bounds = array_bounds(*metadata["shape"], Affine(*metadata["transform"][:6]))
    result = {
        cell["properties"]["cell_id"]: {"cell_id": cell["properties"]["cell_id"]}
        for cell in cells
    }
    areas = {
        cell["properties"]["cell_id"]: cell["properties"]["projected_target_area_m2"]
        for cell in cells
    }
    for name, values in arrays.items():
        raster = NumPyRasterSource(
            values, *bounds, nodata=np.nan, srs_wkt=metadata["crs_wkt"]
        )
        stats = exact_extract(
            raster,
            cells,
            [
                "mean(coverage_weight=area_cartesian)",
                "area=count(coverage_weight=area_cartesian)",
            ],
            include_cols=["cell_id"],
        )
        for row in stats:
            p = row["properties"]
            target = result[p["cell_id"]]
            fraction = p["area"] / areas[p["cell_id"]]
            if fraction > 1.0001 or fraction < 0:
                raise ValueError("Invalid raster support fraction")
            target[name] = p["mean"] if p["area"] > 0 else None
            target[name + "_valid_area_fraction"] = max(0.0, min(1.0, fraction))
    return list(result.values())


def process_site(receipt, bearer):
    site = receipt["site"]
    started = time.monotonic()
    processor_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    cells_path = ROOT / f"research/pilots/terrain/{site}-cells.geojson"
    cells_hash = hashlib.sha256(cells_path.read_bytes()).hexdigest()
    previous_path = OUT / f"{site}-run.json"
    csv_path = OUT / f"{site}.csv"
    if previous_path.exists() and csv_path.exists():
        previous = json.loads(previous_path.read_text())
        if (
            previous.get("task_id") == receipt["task_id"]
            and previous.get("script_sha256") == processor_hash
            and previous.get("cells_sha256") == cells_hash
            and previous.get("csv_sha256")
            == hashlib.sha256(csv_path.read_bytes()).hexdigest()
        ):
            print(f"{site}: reusing verified completed summaries", flush=True)
            return previous
    task = request("/task/" + receipt["task_id"], authorization=bearer)
    if task["status"] != "done":
        print(
            f"{site}: NASA task {task['status']}; no incomplete results imported",
            flush=True,
        )
        return {"site": site, "status": task["status"]}
    bundle = request("/bundle/" + receipt["task_id"], authorization=bearer)
    (OUT / f"{site}-bundle.json").write_text(
        json.dumps(safe_bundle(bundle), indent=2) + "\n"
    )
    groups = file_groups(bundle)
    expected = {f"2024{day:03}" for day in range(1, 367, 8)}
    if {day for day, area in groups} != expected or len(groups) != 46:
        raise ValueError("Expected exactly one area and 46 nominal 2024 composites")
    first = min(groups)

    def fetch_group(key):
        def fetch(layer):
            info = groups[key][layer]
            return layer, decode_file(
                download_file(receipt["task_id"], info, bearer), info
            )

        with ThreadPoolExecutor(max_workers=4) as pool:
            return dict(pool.map(fetch, sorted(NEEDED)))

    # Inspect the actual returned header before processing the remaining dates.
    initial = fetch_group(first)
    reference = aligned(initial)
    cells = projected_cells(site, reference["crs_wkt"])
    first_headers = {name: entry[1] for name, entry in initial.items()}
    (OUT / f"{site}-headers.json").write_text(
        json.dumps(
            {"site": site, "first_group": first, "headers": first_headers},
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    rows = []
    provenance = []

    def append(key, decoded):
        if CRS.from_wkt(aligned(decoded)["crs_wkt"]) != CRS.from_wkt(
            reference["crs_wkt"]
        ):
            raise ValueError("NASA subset CRS changed between composite dates")
        start, end = composite_window(key[0])
        for row in aggregate_arrays(decoded, cells):
            rows.append(
                {
                    "site": site,
                    "window_start": start,
                    "window_end_exclusive": end,
                    **row,
                }
            )
        provenance.append(
            {
                "nominal_date": key[0],
                "area_id": key[1],
                "files": [
                    {
                        "layer": name,
                        **{
                            k: entry[1][k]
                            for k in ("file_name", "file_id", "file_size", "sha256")
                        },
                    }
                    for name, entry in decoded.items()
                ],
            }
        )
        print(f"{site}: summarized {key[0]} ({len(provenance)}/46)", flush=True)

    append(first, initial)
    del initial
    # Two date groups, each with at most four parallel file reads, bound RAM/I/O.
    with ThreadPoolExecutor(max_workers=2) as pool:
        keys = [key for key in sorted(groups) if key != first]
        futures = {}
        iterator = iter(keys)
        for _ in range(2):
            key = next(iterator, None)
            if key is not None:
                futures[key] = pool.submit(fetch_group, key)
        for key in keys:
            decoded = futures.pop(key).result()
            append(key, decoded)
            del decoded
            following = next(iterator, None)
            if following is not None:
                futures[following] = pool.submit(fetch_group, following)
    path = OUT / f"{site}.csv"
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)
    report = {
        "site": site,
        "task_id": receipt["task_id"],
        "script_sha256": processor_hash,
        "cells_sha256": cells_hash,
        "method_version": METHOD,
        "status": "processed_research_pilot",
        "windows": 46,
        "cells": len(cells),
        "cell_window_rows": len(rows),
        "numeric_feature_slots": len(rows) * 3,
        "missing_feature_values": sum(
            row[key] is None
            for row in rows
            for key in ("day_k", "night_k", "paired_day_minus_night_k")
        ),
        "wall_seconds": time.monotonic() - started,
        "csv_bytes": path.stat().st_size,
        "csv_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "verified_subset_payload_bytes": sum(
            f["file_size"] for p in provenance for f in p["files"]
        ),
        "source_files": provenance,
    }
    (OUT / f"{site}-run.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--credentials-file", type=Path, required=True)
    parser.add_argument("--inspect", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    receipts = json.loads((REQUESTS / "receipts.json").read_text())
    bearer = login_credentials(args.credentials_file)
    try:
        results = [
            (inspect_site if args.inspect else process_site)(r, bearer)
            for r in receipts
        ]
    finally:
        logout(bearer)
    name = "inspection.json" if args.inspect else "run.json"
    report = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "method_version": METHOD,
        "status": "processed"
        if all(r["status"] == "processed_research_pilot" for r in results)
        else "waiting_for_NASA",
        "sites": [{"site": r["site"], "status": r["status"]} for r in results],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "peak_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / 1024,
        "source_raster_files_retained": 0,
        "notes": [
            "QA=0 only: good production/data flags, emissivity error <=0.01 and LST error <=1 K.",
            "Composite QA describes majority input flags; it does not re-screen the original daily inputs.",
            "Paired day-minus-night means use the intersection of valid pixels, not the difference of independently masked means.",
            "This is an 8-day clear-sky surface composite, not air temperature or a full diurnal amplitude.",
            "Native MODIS sinusoidal grid, fractional pixel overlap, projected spherical equal-area weights.",
            "Final 2024 composite ends at 2025-01-01 exclusive (six days in the leap year).",
            "CSV is a compact wide ingestion artifact; units/support/provenance are described by this manifest.",
            "Signed download URLs and credentials are never persisted.",
        ],
    }
    (OUT / name).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
