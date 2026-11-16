# /// script
# requires-python = ">=3.11"
# dependencies = ["rasterio==1.5.2", "numpy==2.5.4", "exactextract==0.3.0", "shapely==2.2.0", "pyproj==3.8.0", "h3==4.5.0"]
# ///
"""Scientific fixtures. Run: uv run scripts/test_pilot_terrain.py"""

import math
import unittest

import numpy as np
from process_pilot_terrain import GEOD, accumulator, derivatives, merge_stats, summarize
from rasterio.transform import from_origin
from shapely.geometry import box, mapping


class TerrainFixtures(unittest.TestCase):
    def test_fractional_overlap_and_population_variance(self):
        values = np.array([[1.0, 3.0], [5.0, 7.0]])
        feature = {
            "type": "Feature",
            "properties": {"cell_id": "fixture"},
            "geometry": mapping(box(0, 0, 1.5, 2)),
        }
        p = summarize(values, (0, 0, 2, 2), [feature])[0]["properties"]
        # Full left pixels and half of each right pixel. Spherical row weights
        # differ slightly, so independently construct the spherical area weights.
        lats = np.radians(np.array([0.0, 1.0, 2.0]))
        areas = np.sin(lats[1:]) - np.sin(lats[:-1])
        weights = np.array([[areas[1], areas[1] / 2], [areas[0], areas[0] / 2]])
        mean = np.sum(values * weights) / np.sum(weights)
        variance = np.sum(weights * (values - mean) ** 2) / np.sum(weights)
        self.assertAlmostEqual(p["mean"], mean, places=7)
        self.assertAlmostEqual(p["stdev"], math.sqrt(variance), places=7)

    def test_nodata_is_missing_not_zero(self):
        feature = {
            "type": "Feature",
            "properties": {"cell_id": "fixture"},
            "geometry": mapping(box(0, 0, 2, 1)),
        }
        p = summarize(np.array([[0.0, np.nan]]), (0, 0, 2, 1), [feature])[0][
            "properties"
        ]
        self.assertEqual(p["mean"], 0)
        self.assertAlmostEqual(p["area"] / p["total"], 0.5)

    def test_merging_asset_variance(self):
        acc = accumulator()
        for p in (
            {
                "total": 2.0,
                "area": 2.0,
                "mean": 2.0,
                "stdev": 1.0,
                "min": 1.0,
                "max": 3.0,
            },
            {
                "total": 2.0,
                "area": 2.0,
                "mean": 6.0,
                "stdev": 1.0,
                "min": 5.0,
                "max": 7.0,
            },
        ):
            merge_stats(acc, {"properties": p})
        self.assertAlmostEqual(acc["mean"], 4.0)
        self.assertAlmostEqual(acc["m2"] / acc["area"], 5.0)

    def test_flat_terrain_has_zero_slope_and_undefined_aspect(self):
        d = derivatives(
            np.full((5, 5), 100.0, dtype="float32"),
            from_origin(0, 1, 1 / 3600, 1 / 3600),
        )
        self.assertTrue(np.all(d["slope"][1:-1, 1:-1] == 0))
        self.assertTrue(np.isnan(d["aspect_eastness"]).all())
        self.assertTrue(np.isnan(d["slope"][[0, -1], :]).all())

    def test_east_rising_plane_faces_west(self):
        transform = from_origin(0, 0.001, 1 / 3600, 1 / 3600)
        dx = GEOD.inv(0, 0, 1 / 3600, 0)[2]
        values = np.broadcast_to(np.arange(5, dtype="float32") * dx * 2, (5, 5)).copy()
        d = derivatives(values, transform)
        np.testing.assert_allclose(
            d["slope"][1:-1, 1:-1], np.degrees(np.arctan(2)), atol=1e-5
        )
        np.testing.assert_allclose(d["aspect_eastness"][1:-1, 1:-1], -1, atol=1e-6)
        np.testing.assert_allclose(d["aspect_northness"][1:-1, 1:-1], 0, atol=1e-6)
        values[2, 2] = np.nan
        self.assertTrue(np.isnan(derivatives(values, transform)["slope"]).all())


if __name__ == "__main__":
    unittest.main()
