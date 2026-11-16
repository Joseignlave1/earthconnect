# /// script
# requires-python = ">=3.11"
# dependencies = ["pyhdf==0.11.7", "rasterio==1.5.2", "numpy==2.5.4", "exactextract==0.3.0", "shapely==2.2.0", "pyproj==3.8.0", "h3==4.5.0"]
# ///
"""Process native MOD11A2 files with bounded parallel downloads and no raster archive.

Full HDF4 files are a temporary fallback: this driver cannot read remote windows.
Downloads span all three regions; HDF4 decoding is serial to avoid library races.
"""

import argparse
import base64
import csv
import hashlib
import json
import math
import re
import resource
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, build_opener

import numpy as np
from prepare_nasa_subsets import NoRedirect
from process_nasa_temperature import aggregate_arrays, composite_window, projected_cells
from pyhdf.SD import SD, SDC
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.windows import from_bounds as bounds_window
from shapely.geometry import shape

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research/pilots/temperature-native"
CRS_WKT = CRS.from_string("+proj=sinu +R=6371007.181 +units=m +no_defs").to_wkt()
FIELDS = ("day_k", "night_k", "paired_day_minus_night_k")
LAYERS = ("LST_Day_1km", "LST_Night_1km", "QC_Day", "QC_Night")
METHOD = "mod11a2-native-hdf4-strict-qc0-fractional-area-v1"
PROVIDER = "data.lpdaac.earthdatacloud.nasa.gov"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def original_token(credentials_file):
    secret = credentials_file.read_text().strip()
    url = "https://urs.earthdata.nasa.gov/api/users/find_or_create_token"
    req = Request(
        url,
        data=b"",
        headers={
            "Authorization": "Basic " + base64.b64encode(secret.encode()).decode()
        },
    )
    try:
        with build_opener(NoRedirect).open(req, timeout=60) as response:
            return json.load(response)["access_token"]
    except (HTTPError, URLError):
        raise RuntimeError(
            "Earthdata user-token authentication failed; credentials omitted"
        ) from None


def download(granule, token):
    title = granule["title"]
    if not re.fullmatch(r"MOD11A2\.A2024\d{3}\.h\d{2}v\d{2}\.061\.\d+", title):
        raise ValueError("Unexpected original granule identifier")
    url = f"https://{PROVIDER}/lp-prod-protected/MOD11A2.061/{title}/{title}.hdf"
    for attempt in range(3):
        try:
            target = url
            authorization = "Bearer " + token
            opener = build_opener(NoRedirect)
            for _ in range(4):
                req = Request(
                    target,
                    headers={
                        "User-Agent": "earth-index-native-pilot/1",
                        **({"Authorization": authorization} if authorization else {}),
                    },
                )
                try:
                    response = opener.open(req, timeout=90)
                    break
                except HTTPError as e:
                    if e.code not in (302, 303, 307, 308):
                        raise
                    target = e.headers["Location"]
                    parsed = urlparse(target)
                    if (
                        parsed.scheme != "https"
                        or not parsed.hostname
                        or not (
                            parsed.hostname.endswith(".amazonaws.com")
                            or parsed.hostname == "d1nklfio7vscoe.cloudfront.net"
                        )
                    ):
                        raise RuntimeError(
                            "Unexpected data redirect; no credentials forwarded"
                        ) from None
                    authorization = None
            else:
                raise RuntimeError("Too many data redirects")
            with response:
                payload = response.read(100 * 1024 * 1024 + 1)
                expected = response.headers.get("Content-Length")
            if len(payload) > 100 * 1024 * 1024 or payload[:4] != bytes.fromhex(
                "0e031301"
            ):
                raise RuntimeError("Expected a bounded HDF4 source file")
            if expected and len(payload) != int(expected):
                raise RuntimeError("Source download was truncated")
            return payload, {
                "granule_id": granule["id"],
                "title": title,
                "uri": url,
                "file_size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "checksum_provenance": "computed_on_downloaded_original_bytes",
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
            }
        except (HTTPError, URLError, TimeoutError):
            if attempt == 2:
                raise RuntimeError(
                    "Original source transfer failed; secrets/redirect URL omitted"
                ) from None
            time.sleep(attempt + 1)
    raise AssertionError("unreachable")


def grid_transform(struct, shape):
    def pair(key):
        match = re.search(key + r"=\(([^)]+)\)", struct)
        if not match:
            raise ValueError("Missing native grid coordinates")
        return tuple(map(float, match[1].split(",")))

    if "Projection=GCTP_SNSOID" not in struct or not np.isclose(
        pair("ProjParams")[0], 6371007.181
    ):
        raise ValueError("Unsupported native MODIS projection")
    left, top = pair("UpperLeftPointMtrs")
    right, bottom = pair("LowerRightMtrs")
    return from_bounds(left, bottom, right, top, shape[1], shape[0])


def decode(payload, metadata, bbox):
    # One scratch HDF at a time; the temporary directory is deleted even on error.
    with TemporaryDirectory(prefix="earth-index-mod11a2-") as scratch:
        path = Path(scratch) / "source.hdf"
        path.write_bytes(payload)
        src = SD(str(path), SDC.READ)
        try:
            struct = src.attributes()["StructMetadata.0"]
            dataset = src.select("LST_Day_1km")
            try:
                source_shape = tuple(dataset.info()[2])
            finally:
                dataset.endaccess()
            transform = grid_transform(struct, source_shape)
            requested = bounds_window(*bbox, transform=transform)
            left = max(0, math.floor(requested.col_off))
            top = max(0, math.floor(requested.row_off))
            right = min(source_shape[1], math.ceil(requested.col_off + requested.width))
            bottom = min(
                source_shape[0], math.ceil(requested.row_off + requested.height)
            )
            if left >= right or top >= bottom:
                raise ValueError("Catalog source misses the target region")
            local = transform * transform.translation(left, top)
            decoded = {}
            attributes = {}
            for name in LAYERS:
                dataset = src.select(name)
                try:
                    if tuple(dataset.info()[2]) != source_shape:
                        raise ValueError("Native source grids differ")
                    attrs = dataset.attributes()
                    array = dataset.get(
                        start=(top, left), count=(bottom - top, right - left)
                    )
                    if name.startswith("LST_") and (
                        not np.isclose(attrs["scale_factor"], 0.02)
                        or attrs["add_offset"] != 0
                        or attrs["units"] != "K"
                    ):
                        raise ValueError(
                            "Unexpected native temperature units or packing"
                        )
                    attributes[name] = {
                        k: attrs[k]
                        for k in (
                            "units",
                            "valid_range",
                            "_FillValue",
                            "scale_factor",
                            "add_offset",
                        )
                        if k in attrs
                    }
                    decoded[name] = (
                        array,
                        {
                            "shape": list(array.shape),
                            "crs_wkt": CRS_WKT,
                            "transform": list(local),
                            "scales": [float(attrs.get("scale_factor", 1))],
                            "offsets": [float(attrs.get("add_offset", 0))],
                        },
                    )
                finally:
                    dataset.endaccess()
            metadata.update(
                source_shape=list(source_shape),
                source_transform=list(transform),
                read_window=[left, top, right - left, bottom - top],
                source_field_attributes=attributes,
            )
        finally:
            src.end()
    return decoded, metadata


def merge_cell(acc, row):
    for name in FIELDS:
        fraction = row[name + "_valid_area_fraction"]
        if fraction and row[name] is not None:
            acc[name + "_weighted_sum"] += row[name] * fraction
            acc[name + "_valid_area_fraction"] += fraction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--credentials-file", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=8)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    inventory_path = ROOT / "research/pilots/inventory.json"
    inventory = json.loads(inventory_path.read_text())
    token = original_token(args.credentials_file)
    cells = {
        site["key"]: projected_cells(site["key"], CRS_WKT)
        for site in inventory["sites"]
    }
    boxes = {}
    for site, cs in cells.items():
        bounds = np.array([shape(cell["geometry"]).bounds for cell in cs])
        boxes[site] = (
            bounds[:, 0].min(),
            bounds[:, 1].min(),
            bounds[:, 2].max(),
            bounds[:, 3].max(),
        )
    tasks = []
    for site in inventory["sites"]:
        for granule in site["products"]["MOD11A2"]["granules"]:
            day = re.search(r"\.A(\d{7})\.", granule["title"])[1]
            tasks.append((day, site["key"], granule["title"], granule))
    tasks.sort(key=lambda task: task[:3])  # Interleave regions for each date.
    accumulators = {}
    source_files = {}
    bytes_received = 0
    max_file = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {
            i: pool.submit(download, task[3], token)
            for i, task in enumerate(tasks[: args.workers])
        }
        for index, (day, site, title, granule) in enumerate(tasks):
            payload, metadata = pending.pop(index).result()
            following = index + args.workers
            if following < len(tasks):
                pending[following] = pool.submit(download, tasks[following][3], token)
            bytes_received += len(payload)
            max_file = max(max_file, len(payload))
            decoded, metadata = decode(payload, metadata, boxes[site])
            del payload
            summaries = aggregate_arrays(decoded, cells[site])
            del decoded
            source_files.setdefault((site, day), []).append(metadata)
            for row in summaries:
                key = (site, day, row["cell_id"])
                acc = accumulators.setdefault(
                    key,
                    {
                        field + suffix: 0.0
                        for field in FIELDS
                        for suffix in ("_weighted_sum", "_valid_area_fraction")
                    },
                )
                merge_cell(acc, row)
            if index % 16 == 0 or index + 1 == len(tasks):
                progress = {
                    "completed_original_files": index + 1,
                    "total_original_files": len(tasks),
                    "wall_seconds": time.monotonic() - started,
                    "verified_payload_bytes": bytes_received,
                }
                dump(OUT / "progress.json", progress)
                print(
                    f"Original files {index + 1}/{len(tasks)}; all regions interleaved",
                    flush=True,
                )
    del token
    reports = []
    script_hash = digest(Path(__file__))
    for site, site_cells in cells.items():
        days = sorted(day for s, day in source_files if s == site)
        if len(days) != 46:
            raise ValueError("Incomplete year of original source files")
        rows = []
        for day in days:
            start, end = composite_window(day)
            for cell in site_cells:
                cell_id = cell["properties"]["cell_id"]
                acc = accumulators[(site, day, cell_id)]
                row = {
                    "site": site,
                    "window_start": start,
                    "window_end_exclusive": end,
                    "cell_id": cell_id,
                }
                for name in FIELDS:
                    fraction = acc[name + "_valid_area_fraction"]
                    if fraction > 1.0001:
                        raise ValueError("Overlapping native source support")
                    row[name] = (
                        acc[name + "_weighted_sum"] / fraction if fraction else None
                    )
                    row[name + "_valid_area_fraction"] = min(1.0, fraction)
                rows.append(row)
        path = OUT / f"{site}.csv"
        temporary = path.with_suffix(".csv.tmp")
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
        report = {
            "site": site,
            "status": "processed_research_pilot",
            "input_kind": "native_hdf4",
            "script_sha256": script_hash,
            "cells_sha256": digest(
                ROOT / f"research/pilots/terrain/{site}-cells.geojson"
            ),
            "method_version": METHOD,
            "windows": 46,
            "cells": len(site_cells),
            "cell_window_rows": len(rows),
            "numeric_feature_slots": len(rows) * len(FIELDS),
            "missing_feature_values": sum(
                r[name] is None for r in rows for name in FIELDS
            ),
            "csv_bytes": path.stat().st_size,
            "csv_sha256": digest(path),
            "source_files": [
                {"nominal_date": day, "files": source_files[(site, day)]}
                for day in days
            ],
        }
        dump(OUT / f"{site}-run.json", report)
        reports.append({"site": site, "status": report["status"]})
    dump(
        OUT / "run.json",
        {
            "status": "processed",
            "input_kind": "native_hdf4",
            "processor_script": "scripts/process_native_temperature.py",
            "method_version": METHOD,
            "script_sha256": script_hash,
            "aggregation_module_sha256": digest(
                ROOT / "scripts/process_nasa_temperature.py"
            ),
            "inventory_sha256": digest(inventory_path),
            "sites": reports,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "workers": args.workers,
            "source_granules": len(tasks),
            "downloaded_original_payload_bytes": bytes_received,
            "wall_seconds": time.monotonic() - started,
            "peak_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / 1024,
            "source_raster_files_retained": 0,
            "peak_single_scratch_raster_bytes": max_file,
            "notes": [
                "Original HDF4 files are downloaded in parallel, decoded one at a time and immediately deleted.",
                "Only the intersecting native source window is decoded. No raster archive is retained.",
                "Original quality flags are used, including valid QA zero; no missing subset QA is inferred.",
                "Only QA byte zero and physical source-valid temperatures are included.",
                "Native MODIS sinusoidal grid, exact fractional planar overlap, spherical equal-area support.",
                "Day-minus-night is a paired-pixel composite difference, not a full diurnal amplitude.",
                "Null values indicate no valid support after source-range and strict QA filtering.",
                "All H3 supports are clipped to their pilot regions. Final 2024 window is six days.",
                "Source hashes are computed from received original bytes; provider checksums were not supplied.",
                "Earthdata tokens, passwords and signed redirect URLs are never persisted.",
                "AppEEARS jobs are independent and are not required by this direct ingestion path.",
            ],
        },
    )
    print(
        f"Completed three regions in {time.monotonic() - started:.1f}s; no rasters retained",
        flush=True,
    )


if __name__ == "__main__":
    main()
