"""Check completed temperature artifacts; report queued sources explicitly."""

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research/pilots/temperature"
FIELDS = ["day_k", "night_k", "paired_day_minus_night_k"]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", action="store_true")
    args = parser.parse_args()
    output = ROOT / "research/pilots/temperature-native" if args.native else OUT
    run = json.loads((output / "run.json").read_text())
    processor_hash = sha256(
        ROOT
        / (
            "scripts/process_native_temperature.py"
            if args.native
            else "scripts/process_nasa_temperature.py"
        )
    )
    assert run["script_sha256"] == processor_hash
    if args.native:
        assert run["aggregation_module_sha256"] == sha256(
            ROOT / "scripts/process_nasa_temperature.py"
        )
        assert run["inventory_sha256"] == sha256(
            ROOT / "research/pilots/inventory.json"
        )
    expected_dates = {
        datetime.strptime(f"2024{day:03}", "%Y%j")
        .replace(tzinfo=timezone.utc)
        .isoformat()
        for day in range(1, 367, 8)
    }
    results = []
    pending = []
    for site_status in run["sites"]:
        site = site_status["site"]
        if site_status["status"] != "processed_research_pilot":
            pending.append(site_status)
            continue
        report = json.loads((output / f"{site}-run.json").read_text())
        assert report["script_sha256"] == processor_hash
        assert report["cells_sha256"] == sha256(
            ROOT / f"research/pilots/terrain/{site}-cells.geojson"
        )
        path = output / f"{site}.csv"
        assert report["csv_sha256"] == sha256(path)
        rows = list(csv.DictReader(path.open()))
        cells = json.loads(
            (ROOT / f"research/pilots/terrain/{site}-cells.geojson").read_text()
        )["features"]
        expected_cells = {f["properties"]["cell_id"] for f in cells}
        assert len(rows) == len(expected_cells) * 46 == report["cell_window_rows"]
        assert len({(r["cell_id"], r["window_start"]) for r in rows}) == len(rows)
        assert {r["cell_id"] for r in rows} == expected_cells
        assert {r["window_start"] for r in rows} == expected_dates
        for row in rows:
            start = datetime.fromisoformat(row["window_start"])
            expected_end = min(
                start + timedelta(days=8), datetime(2025, 1, 1, tzinfo=timezone.utc)
            )
            assert row["window_end_exclusive"] == expected_end.isoformat()
            assert row["site"] == site
            fractions = {
                key: float(row[key + "_valid_area_fraction"]) for key in FIELDS
            }
            for key in FIELDS:
                assert 0 <= fractions[key] <= 1
                if row[key] == "":
                    assert fractions[key] == 0
                else:
                    value = float(row[key])
                    assert math.isfinite(value)
                    assert fractions[key] > 0
                    if key != "paired_day_minus_night_k":
                        assert 150 <= value <= 1310.7
            assert (
                fractions["paired_day_minus_night_k"]
                <= min(fractions["day_k"], fractions["night_k"]) + 1e-5
            )
        assert len(report["source_files"]) == 46
        expected_files = (2 if site == "sahara" else 1) if args.native else 4
        assert all(
            len(p["files"]) == expected_files
            and all(len(f["sha256"]) == 64 for f in p["files"])
            for p in report["source_files"]
        )
        if args.native:
            for period in report["source_files"]:
                for source in period["files"]:
                    assert len(source["source_field_attributes"]) == 4
                    assert (
                        source["source_field_attributes"]["LST_Day_1km"]["scale_factor"]
                        == 0.02
                    )
                    assert (
                        source["source_field_attributes"]["LST_Night_1km"]["units"]
                        == "K"
                    )
                    assert source["read_window"][2] <= source["source_shape"][1]
                    assert source["read_window"][3] <= source["source_shape"][0]
        results.append({"site": site, "passed": True, "cell_window_rows": len(rows)})
    assert not any(
        p.suffix.lower() in (".tif", ".tiff", ".hdf", ".nc", ".npy", ".npz")
        for p in output.rglob("*")
    )
    result = {
        "status": (
            "passed"
            if len(results) == 3
            else "validated_available_sites_other_sources_pending"
            if results
            else "waiting_for_NASA_no_temperature_values_validated"
        ),
        "sites": results,
        "pending_sources": pending,
        "checks": [
            "processor_and_geometry_hashes",
            "CSV_content_hashes",
            "46_unique_2024_windows_per_cell",
            "finite_values_and_explicit_missingness",
            "valid_area_fractions",
            "paired_pixel_support",
            "source_original_SHA256_manifests"
            if args.native
            else "source_subset_SHA256_manifests",
            "no_retained_rasters",
        ],
        "separate_scientific_fixture_command": "uv run scripts/test_nasa_temperature.py",
    }
    (output / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
