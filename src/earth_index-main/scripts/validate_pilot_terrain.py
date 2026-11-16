"""Validate real pilot outputs, identifiers, provenance hashes and value invariants."""

import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research/pilots/terrain"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate():
    report = json.loads((OUT / "run.json").read_text())
    inventory = json.loads((ROOT / "research/pilots/inventory.json").read_text())
    assert report["script_sha256"] == digest(ROOT / "scripts/process_pilot_terrain.py")
    assert report["inventory_sha256"] == digest(ROOT / "research/pilots/inventory.json")
    assert report["source_raster_files_retained"] == 0
    assert len(report["sites"]) == 3
    results = []
    for site in inventory["sites"]:
        key = site["key"]
        run = next(r for r in report["sites"] if r["site"] == key)
        path = OUT / f"{key}.csv"
        assert run["csv_sha256"] == digest(path)
        assert run["csv_bytes"] == path.stat().st_size
        rows = list(csv.DictReader(path.open()))
        assert len(rows) == run["observation_rows"]
        expected = set(
            next(
                g["cell_ids"]
                for g in site["grids"]
                if g["resolution"] == report["h3_resolution"]
            )
        )
        grouped = {}
        for row in rows:
            assert row["site"] == key and row["time_kind"] == "static"
            assert row["method_version"] == report["method_version"]
            value = float(row["value"])
            assert math.isfinite(value)
            assert 0 <= float(row["valid_area_fraction"]) <= 1 + 1e-6
            assert float(row["target_area_m2"]) > 0
            assert 0 < float(row["fraction_of_full_cell"]) <= 1 + 1e-6
            assert row["feature_key"] not in grouped.setdefault(row["cell_id"], {})
            grouped[row["cell_id"]][row["feature_key"]] = value
        assert set(grouped) == expected
        assert len(grouped) == run["cells"]
        for values in grouped.values():
            assert len(values) == 12
            assert (
                values["terrain.elevation.min"]
                <= values["terrain.elevation.mean"]
                <= values["terrain.elevation.max"]
            )
            assert values["terrain.elevation.stddev"] >= 0
            assert (
                0 <= values["terrain.slope.mean"] <= values["terrain.slope.max"] <= 90
            )
            assert values["terrain.slope.stddev"] >= 0
            assert (
                0
                <= values["terrain.roughness_3x3.mean"]
                <= values["terrain.local_relief_3x3.mean"] / 2 + 1e-4
            )
            east = values["terrain.aspect_eastness.mean"]
            north = values["terrain.aspect_northness.mean"]
            assert east**2 + north**2 <= 1 + 1e-5
        geometry = json.loads((OUT / f"{key}-cells.geojson").read_text())
        assert {f["properties"]["cell_id"] for f in geometry["features"]} == expected
        for asset in run["assets"]:
            assert asset["read_shape"][0] <= asset["source_shape"][0]
            assert asset["read_shape"][1] <= asset["source_shape"][1]
            assert len(asset["decoded_float32_window_sha256"]) == 64
        results.append(
            {"site": key, "cells": len(grouped), "rows": len(rows), "passed": True}
        )
    retained = [
        str(p.relative_to(OUT))
        for p in OUT.rglob("*")
        if p.suffix.lower() in (".tif", ".tiff", ".hdf", ".nc", ".npy", ".npz")
    ]
    assert not retained
    result = {
        "status": "passed",
        "checks": [
            "provenance_hashes",
            "expected_H3_ids_and_row_counts",
            "finite_values_and_coverage",
            "terrain_value_invariants",
            "clipped_geometry_identifiers",
            "bounded_read_shapes",
            "no_retained_rasters",
        ],
        "total_cells": sum(r["cells"] for r in results),
        "total_rows": sum(r["rows"] for r in results),
        "separate_scientific_fixture_command": "uv run scripts/test_pilot_terrain.py",
        "sites": results,
    }
    (OUT / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    validate()
