#!/usr/bin/env python3
"""Offline checks for the bounded numeric raster reader and audit classifiers."""

import struct
import unittest

from inspect_mars_numeric_sources import decode_vicar
from inspect_planetary_sources import numeric, sample_points


class NumericRasterTests(unittest.TestCase):
    def test_decodes_numeric_float_and_preserves_missing_sentinel(self):
        header = "LBLSIZE=256 FORMAT='REAL' ORG='BSQ' NB=1 NS=2 NL=1 NBB=0 NLB=0 RECSIZE=8 REALFMT='RIEEE'"
        payload = header.encode().ljust(256, b" ") + struct.pack("<ff", 0.0, -32768.0)
        _, values = decode_vicar(payload)
        self.assertEqual(values, [0.0, -32768.0])
        with self.assertRaises(ValueError):
            decode_vicar(payload[:-1])

    def test_decodes_big_endian_integer(self):
        header = "LBLSIZE=256 FORMAT='HALF' ORG='BSQ' NB=1 NS=2 NL=1 NBB=0 NLB=0 RECSIZE=4 INTFMT='HIGH'"
        payload = header.encode().ljust(256, b" ") + struct.pack(">hh", -123, 456)
        self.assertEqual(decode_vicar(payload)[1], [-123, 456])

    def test_rejects_xml_errors_and_unsupported_pixel_layout(self):
        with self.assertRaises(ValueError):
            decode_vicar(b'<ServiceException code="InvalidFormat"/>')
        header = "LBLSIZE=256 FORMAT='REAL' ORG='BIP' NB=3 NS=1 NL=1 NBB=0 RECSIZE=12 REALFMT='RIEEE'"
        with self.assertRaises(ValueError):
            decode_vicar(header.encode().ljust(256, b" ") + bytes(12))

    def test_missing_numeric_samples_do_not_erase_real_zero(self):
        for missing in ["nan", "NoData", "", None, -32768, -3.4e38]:
            self.assertIsNone(numeric(missing))
        self.assertEqual(numeric("0"), 0)

    def test_native_projection_samples_are_not_interpreted_as_earth_degrees(self):
        metadata = {"extent": {"xmin": -300000, "xmax": 300000, "ymin": -300000, "ymax": 300000}}
        points, system = sample_points(metadata)
        self.assertEqual(system, "native_projected_coordinates")
        self.assertEqual(points[0], [0, 0])
        self.assertTrue(any(abs(x) > 180 for x, _ in points))


if __name__ == "__main__":
    unittest.main()
