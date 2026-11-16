# /// script
# requires-python = ">=3.11"
# dependencies = ["pyhdf==0.11.7", "rasterio==1.5.2", "numpy==2.5.4", "exactextract==0.3.0", "shapely==2.2.0", "pyproj==3.8.0", "h3==4.5.0"]
# ///
"""Native HDF4, cross-tile weighting and authentication boundary fixtures."""

import io
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from urllib.error import HTTPError

import numpy as np
from process_native_temperature import (
    FIELDS,
    decode,
    download,
    grid_transform,
    merge_cell,
)
from pyhdf.SD import SD, SDC

STRUCT = "UpperLeftPointMtrs=(0,1)\nLowerRightMtrs=(2,0)\nProjection=GCTP_SNSOID\nProjParams=(6371007.181,0,0)"


class NativeFixtures(unittest.TestCase):
    def test_native_grid(self):
        transform = grid_transform(STRUCT, (1, 2))
        self.assertEqual(transform.a, 1.0)
        self.assertEqual(transform.e, -1.0)
        self.assertEqual(transform.f, 1.0)

    def test_cross_tile_weighted_merge(self):
        acc = {
            name + suffix: 0.0
            for name in FIELDS
            for suffix in ("_weighted_sum", "_valid_area_fraction")
        }
        for value, fraction in ((200.0, 0.25), (300.0, 0.75)):
            row = {name: value for name in FIELDS}
            row.update({name + "_valid_area_fraction": fraction for name in FIELDS})
            merge_cell(acc, row)
        self.assertEqual(acc["day_k_valid_area_fraction"], 1.0)
        self.assertEqual(acc["day_k_weighted_sum"], 275.0)

    def test_real_hdf4_decoding_preserves_good_zero_quality(self):
        with TemporaryDirectory() as scratch:
            path = Path(scratch) / "fixture.hdf"
            src = SD(str(path), SDC.WRITE | SDC.CREATE)
            src.attr("StructMetadata.0").set(SDC.CHAR8, STRUCT)
            for name in ("LST_Day_1km", "LST_Night_1km", "QC_Day", "QC_Night"):
                science = name.startswith("LST_")
                dataset = src.create(name, SDC.UINT16 if science else SDC.UINT8, (1, 2))
                dataset[:] = np.array(
                    [[12500, 0]] if science else [[0, 3]],
                    dtype="uint16" if science else "uint8",
                )
                if science:
                    dataset.scale_factor = 0.02
                    dataset.add_offset = 0.0
                    dataset.units = "K"
                dataset.endaccess()
            src.end()
            decoded, metadata = decode(path.read_bytes(), {}, (0, 0, 2, 1))
        self.assertEqual(decoded["LST_Day_1km"][0].tolist(), [[12500, 0]])
        self.assertEqual(decoded["QC_Day"][0].tolist(), [[0, 3]])
        self.assertEqual(metadata["read_window"], [0, 0, 2, 1])

    def test_cdn_redirect_strips_authentication(self):
        payload = bytes.fromhex("0e031301") + b"fixture"
        calls = []

        class Response(io.BytesIO):
            def __init__(self, content):
                super().__init__(content)
                self.headers = {"Content-Length": str(len(content))}

        class Opener:
            def open(self, req, timeout):
                calls.append(req)
                if len(calls) == 1:
                    raise HTTPError(
                        req.full_url,
                        303,
                        "See Other",
                        {"Location": "https://d1nklfio7vscoe.cloudfront.net/fixture"},
                        None,
                    )
                return Response(payload)

        with patch("process_native_temperature.build_opener", return_value=Opener()):
            received, _ = download(
                {"id": "fixture", "title": "MOD11A2.A2024001.h11v11.061.2024011194050"},
                "fixture-token",
            )
        self.assertEqual(received, payload)
        self.assertEqual(calls[0].get_header("Authorization"), "Bearer fixture-token")
        self.assertIsNone(calls[1].get_header("Authorization"))


if __name__ == "__main__":
    unittest.main()
