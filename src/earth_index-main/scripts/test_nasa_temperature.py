# /// script
# requires-python = ">=3.11"
# dependencies = ["rasterio==1.5.2", "numpy==2.5.4", "exactextract==0.3.0", "shapely==2.2.0", "pyproj==3.8.0", "h3==4.5.0"]
# ///
"""Scientific and credential-handling fixtures; no NASA login or network needed."""

import hashlib
import io
import unittest
from datetime import datetime
from unittest.mock import patch
from urllib.error import HTTPError

import numpy as np
from process_nasa_temperature import (
    aggregate_arrays,
    aligned,
    composite_window,
    download_file,
    kelvin,
    quality_screen,
)
from rasterio.crs import CRS
from rasterio.transform import from_origin
from shapely.geometry import box, mapping

CRS_WKT = CRS.from_string("+proj=sinu +R=6371007.181 +units=m +no_defs").to_wkt()


def metadata():
    return {
        "scales": [1.0],
        "offsets": [0.0],
        "shape": [1, 2],
        "transform": list(from_origin(0, 1, 1, 1)),
        "crs_wkt": CRS_WKT,
    }


def fixture(day, night, polygon=None):
    decoded = {
        name: (np.array([values], dtype=dtype), metadata())
        for name, values, dtype in [
            ("LST_Day_1km", day, "float32"),
            ("LST_Night_1km", night, "float32"),
            ("QC_Day", [0, 0], "uint8"),
            ("QC_Night", [0, 0], "uint8"),
        ]
    }
    polygon = box(0, 0, 2, 1) if polygon is None else polygon
    cells = [
        {
            "type": "Feature",
            "geometry": mapping(polygon),
            "properties": {
                "cell_id": "fixture",
                "projected_target_area_m2": polygon.area,
            },
        }
    ]
    return decoded, cells


class TemperatureFixtures(unittest.TestCase):
    def test_packed_and_physical_values_decode_once(self):
        packed = kelvin(np.array([12500, 0], dtype="uint16"), metadata())
        physical = kelvin(np.array([250.0, 0.0], dtype="float32"), metadata())
        np.testing.assert_allclose(packed, physical, equal_nan=True)
        self.assertEqual(packed[0], 250.0)
        with self.assertRaises(ValueError):
            kelvin(np.array([12500.0], dtype="float32"), metadata())

    def test_quality_zero_is_good(self):
        result = quality_screen(
            np.full(4, 300.0), np.array([0, 1, 2, 64], dtype="uint8")
        )
        self.assertEqual(result[0], 300.0)
        self.assertTrue(np.isnan(result[1:]).all())

    def test_leap_year_final_composite_is_six_days(self):
        start, end = map(datetime.fromisoformat, composite_window("2024361"))
        self.assertEqual((end - start).days, 6)
        self.assertEqual(end.isoformat(), "2025-01-01T00:00:00+00:00")

    def test_difference_uses_paired_pixels(self):
        decoded, cells = fixture([300.0, 320.0], [280.0, 0.0])
        result = aggregate_arrays(decoded, cells)[0]
        self.assertEqual(result["day_k"], 310.0)
        self.assertEqual(result["night_k"], 280.0)
        self.assertEqual(result["paired_day_minus_night_k"], 20.0)
        self.assertEqual(result["paired_day_minus_night_k_valid_area_fraction"], 0.5)

    def test_fractional_pixel_overlap(self):
        decoded, cells = fixture([300.0, 330.0], [290.0, 290.0], box(0, 0, 1.5, 1))
        result = aggregate_arrays(decoded, cells)[0]
        self.assertAlmostEqual(result["day_k"], 310.0)
        self.assertAlmostEqual(result["day_k_valid_area_fraction"], 1.0)

    def test_all_missing_remains_null(self):
        decoded, cells = fixture([0.0, 0.0], [0.0, 0.0])
        result = aggregate_arrays(decoded, cells)[0]
        self.assertIsNone(result["day_k"])
        self.assertEqual(result["day_k_valid_area_fraction"], 0.0)

    def test_misaligned_quality_grid_is_rejected(self):
        decoded, _ = fixture([300.0, 300.0], [290.0, 290.0])
        decoded["QC_Day"][1]["transform"][2] = 1
        with self.assertRaises(ValueError):
            aligned(decoded)

    def test_signed_download_gets_no_bearer_header(self):
        payload = b"test"
        info = {
            "file_id": "fixture",
            "file_size": 4,
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        calls = []

        class Opener:
            def open(self, req, timeout):
                calls.append(req)
                if len(calls) == 1:
                    raise HTTPError(
                        req.full_url,
                        302,
                        "Found",
                        {
                            "Location": "https://example.s3.amazonaws.com/file?signature=example"
                        },
                        None,
                    )
                return io.BytesIO(payload)

        with patch("process_nasa_temperature.build_opener", return_value=Opener()):
            self.assertEqual(download_file("task", info, "Bearer fixture"), payload)
        self.assertEqual(calls[0].get_header("Authorization"), "Bearer fixture")
        self.assertIsNone(calls[1].get_header("Authorization"))


if __name__ == "__main__":
    unittest.main()
