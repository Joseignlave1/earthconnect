# /// script
# requires-python = ">=3.11"
# dependencies = ["psycopg[binary]==3.3.9", "h3==4.5.0"]
# ///
"""Integration checks against a loaded PostgreSQL/PostGIS pilot database.

Fixtures run in rolled-back transactions; source observations are never changed.
"""

import csv
import os
import unittest

import psycopg
from load_pilots import ROOT, SITES, TEMPERATURE, TERRAIN, jsonb_safe, read
from psycopg.types.json import Jsonb


class PostgresPilotTests(unittest.TestCase):
    def setUp(self):
        self.conn = psycopg.connect(os.getenv("DATABASE_URL", ""), autocommit=True)
        self.addCleanup(self.conn.close)
        self.import_id = self.conn.execute(
            "SELECT max(import_id) FROM ingest.imports"
        ).fetchone()[0]
        self.assertIsNotNone(self.import_id)

    def test_every_source_value_and_missing_fraction_round_trips(self):
        outputs = {
            output["id"]: output["artifactColumn"]
            for variable in read(ROOT / "catalog/variables.json")["variables"]
            for output in variable["pilot"]["outputs"]
            if output["artifactColumn"]
        }
        expected = {}
        for site in SITES:
            with (TERRAIN / f"{site}.csv").open() as stream:
                for row in csv.DictReader(stream):
                    expected[(site, row["cell_id"], row["feature_key"], None)] = (
                        float(row["value"]) if row["value"] else None,
                        float(row["valid_area_fraction"]),
                        row["unit"],
                    )
            with (TEMPERATURE / f"{site}.csv").open() as stream:
                for row in csv.DictReader(stream):
                    for output, column in outputs.items():
                        expected[
                            (site, row["cell_id"], output, row["window_start"])
                        ] = (
                            float(row[column]) if row[column] else None,
                            float(row[column + "_valid_area_fraction"]),
                            "K",
                        )
        self.assertEqual(len(expected), 143850)
        rows = self.conn.execute(
            "SELECT region_id_external,cell_id,output_id,time_start,value_numeric,valid_area_fraction,unit,missing_reason FROM research.pilot_observations WHERE import_id=%s",
            (self.import_id,),
        )
        for region, cell, output, start, value, fraction, unit, missing in rows:
            key = (region, cell, output, start.isoformat() if start else None)
            self.assertEqual((value, fraction, unit), expected.pop(key))
            self.assertEqual(value is None, missing is not None)
        self.assertFalse(expected)

    def test_boundaries_and_temporal_support(self):
        self.assertEqual(
            self.conn.execute("SELECT count(*) FROM geo.cells").fetchone()[0], 959
        )
        self.assertEqual(
            self.conn.execute(
                "SELECT count(*) FROM geo.cell_supports WHERE ST_IsValid(geom) AND NOT ST_IsEmpty(geom)"
            ).fetchone()[0],
            959,
        )
        self.assertGreater(
            self.conn.execute(
                "SELECT count(*) FROM geo.cell_supports WHERE fraction_of_full_cell < 0.99"
            ).fetchone()[0],
            0,
        )
        self.assertEqual(
            self.conn.execute(
                "SELECT count(*) FROM geo.cell_supports s JOIN geo.cells c USING (grid_id,cell_id) WHERE ST_Area(s.geom::geography) > c.area_m2 * 1.00001"
            ).fetchone()[0],
            0,
        )
        self.assertEqual(
            self.conn.execute(
                "SELECT count(*) FROM data.time_windows WHERE kind='interval' AND time_start='2024-12-26' AND time_end='2025-01-01'"
            ).fetchone()[0],
            1,
        )
        self.assertEqual(
            self.conn.execute(
                "SELECT count(*) FROM data.time_windows WHERE kind='static' AND time_start IS NULL AND time_end IS NULL"
            ).fetchone()[0],
            1,
        )

    def assert_rejected(self, query, parameters=()):
        with self.assertRaises(psycopg.Error) as raised, self.conn.transaction():
            self.conn.execute(query, parameters)
        self.assertIn(raised.exception.sqlstate, {"23514", "23503", "23505", "P0001"})

    def test_source_NUL_metadata_is_preserved_reversibly(self):
        encoded = jsonb_safe({"source": "NA\x00source"})["source"]
        self.assertEqual("".join(map(chr, encoded["code_points"])), "NA\x00source")
        with self.conn.transaction(force_rollback=True):
            stored = self.conn.execute(
                "SELECT %s::jsonb", (Jsonb(encoded),)
            ).fetchone()[0]
            self.assertEqual(stored, encoded)

    def test_constraints_and_sealed_series(self):
        with self.conn.transaction(force_rollback=True):
            template = self.conn.execute(
                "SELECT feature_version_id,grid_id,region_id FROM data.series WHERE window_kind='static' LIMIT 1"
            ).fetchone()
            feature, grid, region = template
            imported = self.conn.execute(
                "INSERT INTO ingest.imports (bundle_sha256,input_manifest,expected_observations) VALUES (%s,%s,1) RETURNING import_id",
                ("a" * 64, Jsonb({"fixture": True})),
            ).fetchone()[0]
            run = self.conn.execute(
                "INSERT INTO ingest.runs (import_id,region_id,source_kind,source_manifest_sha256,manifest,status) VALUES (%s,%s,'terrain','fixture',%s,'validated_research_artifact') RETURNING run_id",
                (imported, region, Jsonb({"fixture": True})),
            ).fetchone()[0]
            series = self.conn.execute(
                "INSERT INTO data.series (feature_version_id,grid_id,region_id,run_id,window_kind,support_metadata) VALUES (%s,%s,%s,%s,'static',%s) RETURNING series_id",
                (feature, grid, region, run, Jsonb({"fixture": True})),
            ).fetchone()[0]
            cell = self.conn.execute(
                "SELECT cell_id FROM geo.cell_supports WHERE grid_id=%s AND region_id=%s LIMIT 1",
                (grid, region),
            ).fetchone()[0]
            static = self.conn.execute(
                "SELECT window_id FROM data.time_windows WHERE kind='static'"
            ).fetchone()[0]
            interval = self.conn.execute(
                "SELECT window_id FROM data.time_windows WHERE kind='interval' LIMIT 1"
            ).fetchone()[0]
            insert = "INSERT INTO data.observations VALUES (%s,%s,%s,%s,%s,%s,%s,%s)"
            for value, reason, fraction in (
                (float("nan"), None, 1),
                (float("inf"), None, 1),
                (-float("inf"), None, 1),
                (1, None, 1.1),
                (1, None, -0.1),
                (None, None, 0),
                (None, "missing", 0.5),
                (1, "missing", 1),
            ):
                self.assert_rejected(
                    insert,
                    (series, grid, region, cell, static, value, reason, fraction),
                )
            self.assert_rejected(
                insert, (series, grid, region, cell, interval, 1, None, 1)
            )
            self.assert_rejected(
                insert, (series, grid, region, "invalid-cell-id", static, 1, None, 1)
            )
            self.conn.execute(insert, (series, grid, region, cell, static, 0, None, 1))
            self.assertEqual(
                self.conn.execute(
                    "SELECT value_numeric FROM data.observations WHERE series_id=%s",
                    (series,),
                ).fetchone()[0],
                0,
            )
            self.assert_rejected(
                insert, (series, grid, region, cell, static, 0, None, 1)
            )
            self.conn.execute(
                "UPDATE data.series SET status='sealed',sealed_at=now() WHERE series_id=%s",
                (series,),
            )
            self.assert_rejected(
                "UPDATE data.observations SET value_numeric=2 WHERE series_id=%s",
                (series,),
            )
            self.assert_rejected(
                "DELETE FROM data.observations WHERE series_id=%s", (series,)
            )
            self.assert_rejected(
                "UPDATE data.series SET support_metadata='{}' WHERE series_id=%s",
                (series,),
            )
            self.assert_rejected(
                "UPDATE catalog.feature_versions SET canonical_unit='wrong' WHERE feature_version_id=%s",
                (feature,),
            )

    def test_api_login_reads_views_and_cannot_write(self):
        password = os.getenv("API_DB_PASSWORD")
        self.assertTrue(password, "Provide API_DB_PASSWORD to verify the API login")
        # DATABASE_URL, when provided, contains the owner user; explicit kwargs override it.
        with psycopg.connect(
            os.getenv("DATABASE_URL", ""),
            user="earth_index_api",
            password=password,
            autocommit=True,
        ) as conn:
            self.assertEqual(
                conn.execute("SHOW default_transaction_read_only").fetchone()[0], "on"
            )
            expected_variables = {
                variable["id"]
                for variable in read(ROOT / "catalog/variables.json")["variables"]
            }
            visible_variables = {
                row[0]
                for row in conn.execute("SELECT variable_key FROM research.variables")
            }
            self.assertTrue(expected_variables.issubset(visible_variables))
            self.assertEqual(
                conn.execute(
                    "SELECT count(*) FROM research.pilot_observations WHERE import_id=%s",
                    (self.import_id,),
                ).fetchone()[0],
                143850,
            )
            self.assertIsNotNone(
                conn.execute(
                    "SELECT public.ST_AsGeoJSON(geom) FROM research.cell_supports LIMIT 1"
                ).fetchone()[0]
            )
            with self.assertRaises(psycopg.Error):
                conn.execute("SELECT * FROM data.observations LIMIT 1")
            with self.assertRaises(psycopg.Error):
                conn.execute("UPDATE research.variables SET label='wrong'")
            conn.execute("SET default_transaction_read_only=off")
            with self.assertRaises(psycopg.Error):
                conn.execute("UPDATE research.variables SET label='wrong'")


if __name__ == "__main__":
    unittest.main()
