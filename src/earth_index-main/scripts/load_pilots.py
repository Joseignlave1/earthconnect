# /// script
# requires-python = ">=3.11"
# dependencies = ["psycopg[binary]==3.3.9", "h3==4.5.0"]
# ///
"""Load verified pilot artifacts into PostgreSQL/PostGIS in one transaction.

Uses standard PG* variables or DATABASE_URL. Never prints connection credentials.
Only numeric pilot outputs are imported; all 91 candidate definitions are cataloged.
"""

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import h3
import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb

ROOT = Path(__file__).resolve().parents[1]
SITES = ("atacama", "iceland", "sahara")
TEMPERATURE = ROOT / "research/pilots/temperature-native"
TERRAIN = ROOT / "research/pilots/terrain"


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def jsonb_safe(value):
    """Preserve source strings containing NUL in a reversible JSONB-safe form.

    PostgreSQL rejects U+0000 even in JSONB string escapes. Original manifests
    remain byte-for-byte in the repository and are referenced by their hashes.
    """
    if isinstance(value, str) and "\x00" in value:
        return {
            "earth_index_encoding": "unicode_code_points_for_source_string_with_NUL",
            "code_points": [ord(character) for character in value],
        }
    if isinstance(value, list):
        return [jsonb_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: jsonb_safe(item) for key, item in value.items()}
    return value


def checked(condition, message):
    if not condition:
        raise RuntimeError(message)


def verify_inputs():
    """Fail before database writes if any scientific artifact/code hash changed."""
    paths = [ROOT / "catalog/variables.json", ROOT / "research/pilots/areas.geojson"]
    paths += [ROOT / "research/pilots/inventory.json"]
    for directory, processor in (
        (TERRAIN, "process_pilot_terrain.py"),
        (TEMPERATURE, "process_native_temperature.py"),
    ):
        run = read(directory / "run.json")
        checked(
            read(directory / "validation.json")["status"] == "passed",
            "Pilot validation has not passed",
        )
        checked(
            run["script_sha256"] == digest(ROOT / "scripts" / processor),
            "Processor hash mismatch",
        )
        checked(
            run["inventory_sha256"] == digest(ROOT / "research/pilots/inventory.json"),
            "Inventory hash mismatch",
        )
        paths += [directory / "run.json", ROOT / "scripts" / processor]
        if directory == TEMPERATURE:
            helper = ROOT / "scripts/process_nasa_temperature.py"
            checked(
                run["aggregation_module_sha256"] == digest(helper),
                "Aggregation module hash mismatch",
            )
            paths.append(helper)
        for site in SITES:
            report = read(directory / f"{site}-run.json")
            csv_path = directory / f"{site}.csv"
            checked(report["csv_sha256"] == digest(csv_path), "CSV hash mismatch")
            cells = TERRAIN / f"{site}-cells.geojson"
            if directory == TEMPERATURE:
                checked(
                    report["script_sha256"] == run["script_sha256"],
                    "Per-region processor hash mismatch",
                )
                checked(
                    report["cells_sha256"] == digest(cells),
                    "Cell geometry hash mismatch",
                )
            paths += [csv_path, directory / f"{site}-run.json", cells]
    manifest = {str(p.relative_to(ROOT)): digest(p) for p in sorted(set(paths))}
    return manifest, fingerprint(manifest)


def migrate(conn):
    conn.execute(
        "CREATE TABLE IF NOT EXISTS public.earth_index_migrations (version text PRIMARY KEY, sha256 text NOT NULL, applied_at timestamptz NOT NULL DEFAULT now())"
    )
    for path in sorted((ROOT / "db" / "migrations").glob("*.sql")):
        previous = conn.execute(
            "SELECT sha256 FROM public.earth_index_migrations WHERE version = %s",
            (path.name,),
        ).fetchone()
        if previous:
            checked(previous[0] == digest(path), "An applied migration was modified")
        else:
            conn.execute(path.read_text(), prepare=False)
            conn.execute(
                "INSERT INTO public.earth_index_migrations (version, sha256) VALUES (%s,%s)",
                (path.name, digest(path)),
            )


def catalog(conn, definitions):
    versions = {}
    for variable in definitions:
        conn.execute(
            "INSERT INTO catalog.variables VALUES (%s,%s,%s,%s,%s) ON CONFLICT (variable_key) DO UPDATE SET definition_version=EXCLUDED.definition_version, label=EXCLUDED.label, value_kind=EXCLUDED.value_kind, definition=EXCLUDED.definition",
            (
                variable["id"],
                variable["definitionVersion"],
                variable["label"],
                variable["valueType"],
                Jsonb(variable),
            ),
        )
        for output in variable["pilot"]["outputs"]:
            method = read(ROOT / variable["pilot"]["manifest"])["method_version"]
            definition = {
                "output": output,
                "variable_id": variable["id"],
                "source_id": variable["sourceId"],
                "method_version": method,
                "quantity_notes": variable["notes"],
                "spatial_support": "h3_cell_intersection_with_pilot_region",
                "temporal_kind": variable["temporalKind"],
                "valid_fraction_denominator": "clipped_cell_area_in_the_processing_area_model; see run manifest",
                "quality_policy": "raw_MOD11A2_QA_zero_and_valid_temperature_range"
                if variable["groupId"] == "surface_thermal"
                else "nodata_only; auxiliary_DEM_quality_masks_unavailable",
            }
            definition_hash = fingerprint(definition)
            conn.execute(
                "INSERT INTO catalog.features VALUES (%s,%s) ON CONFLICT DO NOTHING",
                (output["id"], variable["id"]),
            )
            conn.execute(
                "INSERT INTO catalog.feature_versions (feature_key,version,canonical_unit,statistic,definition,definition_sha256) VALUES (%s,1,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                (
                    output["id"],
                    output["unit"],
                    output["statistic"],
                    Jsonb(definition),
                    definition_hash,
                ),
            )
            row = conn.execute(
                "SELECT feature_version_id,definition_sha256 FROM catalog.feature_versions WHERE feature_key=%s AND version=1",
                (output["id"],),
            ).fetchone()
            checked(
                row[1] == definition_hash,
                "Feature definition changed; create a new version before importing",
            )
            versions[output["id"]] = row[0]
    checked(
        len(definitions) == 91 and len(versions) == 15,
        "Unexpected catalog/output count",
    )
    return versions


def geography(conn):
    conn.execute(
        "INSERT INTO geo.grids (grid_key,body,scheme,resolution,library_version,area_method) VALUES ('earth-h3-r6-wgs84-v1','earth','h3',6,%s,'PostGIS geography WGS84 ellipsoidal full-cell area') ON CONFLICT DO NOTHING",
        (h3.__version__,),
    )
    grid = conn.execute(
        "SELECT grid_id FROM geo.grids WHERE grid_key='earth-h3-r6-wgs84-v1'"
    ).fetchone()[0]
    regions = {}
    for feature in read(ROOT / "research/pilots/areas.geojson")["features"]:
        properties = feature["properties"]
        site = properties["key"]
        version = fingerprint(feature["geometry"])
        geometry = json.dumps(feature["geometry"])
        conn.execute(
            "INSERT INTO geo.regions (region_key,version,name,geom,source_metadata) VALUES (%s,%s,%s,ST_Multi(ST_GeomFromGeoJSON(%s)),%s) ON CONFLICT DO NOTHING",
            (site, version, properties["name"], geometry, Jsonb(properties)),
        )
        region = conn.execute(
            "SELECT region_id FROM geo.regions WHERE region_key=%s AND version=%s",
            (site, version),
        ).fetchone()[0]
        regions[site] = region
        for cell in read(TERRAIN / f"{site}-cells.geojson")["features"]:
            p = cell["properties"]
            cell_id = p["cell_id"]
            checked(
                h3.is_valid_cell(cell_id) and h3.get_resolution(cell_id) == 6,
                "Invalid H3 identity",
            )
            boundary = [[lon, lat] for lat, lon in h3.cell_to_boundary(cell_id)]
            boundary.append(boundary[0])
            full = json.dumps({"type": "Polygon", "coordinates": [boundary]})
            lat, lon = h3.cell_to_latlng(cell_id)
            conn.execute(
                "INSERT INTO geo.cells (grid_id,cell_id,geom,centroid,area_m2) SELECT %s,%s,geom,ST_SetSRID(ST_MakePoint(%s,%s),4326)::geography,ST_Area(geom::geography) FROM (SELECT ST_Multi(ST_GeomFromGeoJSON(%s)) AS geom) q ON CONFLICT DO NOTHING",
                (grid, cell_id, lon, lat, full),
            )
            conn.execute(
                "INSERT INTO geo.cell_supports VALUES (%s,%s,%s,ST_Multi(ST_GeomFromGeoJSON(%s)),%s,%s,'pyproj.Geod WGS84 clipped polygon; processing pixel weights use run-specific area model') ON CONFLICT DO NOTHING",
                (
                    grid,
                    region,
                    cell_id,
                    json.dumps(cell["geometry"]),
                    p["target_area_m2"],
                    p["fraction_of_full_cell"],
                ),
            )
    return grid, regions


def window(conn, kind, start=None, end=None):
    key = fingerprint(
        {"kind": kind, "start": start, "end_exclusive": end, "calendar": "gregorian"}
    )
    conn.execute(
        "INSERT INTO data.time_windows (window_sha256,kind,time_start,time_end) VALUES (%s,%s,%s,%s) ON CONFLICT DO NOTHING",
        (key, kind, start, end),
    )
    return conn.execute(
        "SELECT window_id FROM data.time_windows WHERE window_sha256=%s", (key,)
    ).fetchone()[0]


def import_data(conn, import_id, definitions, feature_versions, grid, regions):
    static_window = window(conn, "static")
    time_windows = {}
    outputs = {o["id"]: o for v in definitions for o in v["pilot"]["outputs"]}
    for directory, source_kind in ((TERRAIN, "terrain"), (TEMPERATURE, "temperature")):
        for site in SITES:
            region = regions[site]
            report_path = directory / f"{site}-run.json"
            manifest = {
                "global": read(directory / "run.json"),
                "region": read(report_path),
            }
            run_id = conn.execute(
                "INSERT INTO ingest.runs (import_id,region_id,source_kind,source_manifest_sha256,manifest,status) VALUES (%s,%s,%s,%s,%s,'validated_research_artifact') RETURNING run_id",
                (
                    import_id,
                    region,
                    source_kind,
                    digest(report_path),
                    Jsonb(jsonb_safe(manifest)),
                ),
            ).fetchone()[0]
            feature_keys = [
                key
                for key in feature_versions
                if key.startswith("terrain.") == (source_kind == "terrain")
            ]
            series = {}
            for key in feature_keys:
                support = {
                    "kind": "h3_cell_intersection_with_pilot_region",
                    "geometry_relation": "geo.cell_supports",
                    "area_weights": "spherical_native_pixel_area"
                    if source_kind == "terrain"
                    else "native_MODIS_sinusoidal_equal_area",
                    "notes": manifest["global"].get("notes", []),
                }
                series[key] = conn.execute(
                    "INSERT INTO data.series (feature_version_id,grid_id,region_id,run_id,window_kind,support_metadata) VALUES (%s,%s,%s,%s,%s,%s) RETURNING series_id",
                    (
                        feature_versions[key],
                        grid,
                        region,
                        run_id,
                        "static" if source_kind == "terrain" else "interval",
                        Jsonb(support),
                    ),
                ).fetchone()[0]
            with (directory / f"{site}.csv").open() as stream:
                rows = csv.DictReader(stream)
                if source_kind == "temperature":
                    # Window IDs must be resolved before entering COPY on this connection.
                    rows = list(rows)
                    for row in rows:
                        pair = (row["window_start"], row["window_end_exclusive"])
                        if pair not in time_windows:
                            time_windows[pair] = window(conn, "interval", *pair)
                with conn.cursor().copy(
                    "COPY data.observations (series_id,grid_id,region_id,cell_id,window_id,value_numeric,missing_reason,valid_fraction) FROM STDIN"
                ) as copy:
                    for row in rows:
                        checked(row["site"] == site, "CSV region mismatch")
                        for key in (
                            [row["feature_key"]]
                            if source_kind == "terrain"
                            else feature_keys
                        ):
                            value_column = (
                                "value"
                                if source_kind == "terrain"
                                else outputs[key]["artifactColumn"]
                            )
                            fraction_column = (
                                "valid_area_fraction"
                                if source_kind == "terrain"
                                else value_column + "_valid_area_fraction"
                            )
                            value = (
                                float(row[value_column]) if row[value_column] else None
                            )
                            if source_kind == "terrain":
                                checked(
                                    row["unit"] == outputs[key]["unit"],
                                    "Terrain unit mismatch",
                                )
                            wid = (
                                static_window
                                if source_kind == "terrain"
                                else time_windows[
                                    (row["window_start"], row["window_end_exclusive"])
                                ]
                            )
                            copy.write_row(
                                (
                                    series[key],
                                    grid,
                                    region,
                                    row["cell_id"],
                                    wid,
                                    value,
                                    "no_valid_support_after_source_range_and_quality_screening"
                                    if value is None
                                    else None,
                                    float(row[fraction_column]),
                                )
                            )
            conn.execute(
                "UPDATE data.series SET status='sealed',sealed_at=now() WHERE run_id=%s",
                (run_id,),
            )


def measurements(conn, import_id):
    counts = conn.execute(
        "SELECT count(*),count(o.value_numeric),count(*) FILTER (WHERE o.value_numeric IS NULL),count(DISTINCT o.cell_id),count(DISTINCT o.series_id),count(DISTINCT o.window_id) FROM data.observations o JOIN data.series s USING (series_id) JOIN ingest.runs r USING (run_id) WHERE r.import_id=%s",
        (import_id,),
    ).fetchone()
    checked(
        counts == (143850, 130211, 13639, 959, 45, 47),
        "Imported counts differ from the verified pilot domain",
    )
    checked(
        conn.execute(
            "SELECT count(*) FROM data.series s JOIN ingest.runs r USING (run_id) WHERE r.import_id=%s AND s.status <> 'sealed'",
            (import_id,),
        ).fetchone()[0]
        == 0,
        "Import has unsealed series",
    )
    return dict(
        zip(
            (
                "observation_rows",
                "valid_values",
                "explicit_missing_values",
                "cells",
                "series",
                "time_windows",
            ),
            counts,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    inputs, bundle = verify_inputs()
    definitions = read(ROOT / "catalog/variables.json")["variables"]
    with psycopg.connect(os.getenv("DATABASE_URL", ""), autocommit=True) as conn:
        with conn.transaction():
            conn.execute("SELECT pg_advisory_xact_lock(813397782)")
            migrate(conn)
            if password := os.getenv("API_DB_PASSWORD"):
                conn.execute(
                    sql.SQL("ALTER ROLE earth_index_api PASSWORD {}").format(
                        sql.Literal(password)
                    )
                )
            previous = conn.execute(
                "SELECT import_id FROM ingest.imports WHERE bundle_sha256=%s", (bundle,)
            ).fetchone()
            if previous:
                import_id = previous[0]
            else:
                versions = catalog(conn, definitions)
                grid, regions = geography(conn)
                import_id = conn.execute(
                    "INSERT INTO ingest.imports (bundle_sha256,input_manifest,expected_observations) VALUES (%s,%s,143850) RETURNING import_id",
                    (bundle, Jsonb(inputs)),
                ).fetchone()[0]
                import_data(conn, import_id, definitions, versions, grid, regions)
            counts = measurements(conn, import_id)
        for table in (
            "data.observations",
            "data.series",
            "geo.cells",
            "geo.cell_supports",
        ):
            conn.execute(
                sql.SQL("ANALYZE {}").format(sql.Identifier(*table.split(".")))
            )
        relations = conn.execute(
            "SELECT n.nspname || '.' || c.relname,pg_relation_size(c.oid),pg_total_relation_size(c.oid) FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace WHERE c.relkind='r' AND n.nspname IN ('catalog','geo','ingest','data') ORDER BY pg_total_relation_size(c.oid) DESC"
        ).fetchall()
        report = {
            "status": "loaded_verified_research_pilot_not_published_release",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "import_id": import_id,
            "bundle_sha256": bundle,
            "already_imported": bool(previous),
            "counts": counts,
            "candidate_variables": 91,
            "processed_output_definitions": 15,
            "postgresql_version": conn.execute("SHOW server_version").fetchone()[0],
            "postgis_version": conn.execute("SELECT postgis_lib_version()").fetchone()[
                0
            ],
            "database_bytes": conn.execute(
                "SELECT pg_database_size(current_database())"
            ).fetchone()[0],
            "project_tables_and_indexes_bytes": sum(row[2] for row in relations),
            "relations": [
                {"name": name, "heap_bytes": heap, "total_bytes": total}
                for name, heap, total in relations
            ],
            "notes": [
                "Database bytes include PostgreSQL/PostGIS catalog overhead, not cluster WAL or container images.",
                "Project relation totals include indexes and TOAST metadata.",
                "API role earth_index_api reads research views; credentials remain in ignored .env.",
                "CSV missing values are SQL NULL with explicit reasons, not zero.",
                "No raster data was imported. No scientific publication release is declared.",
            ],
        }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
