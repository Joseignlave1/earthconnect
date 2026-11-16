import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio.transform import from_origin

from expansion_features import contexts as reference_contexts
from raster_source_cache import atomic_json
from world_features import contexts, warp_array, read_projected, sha_file, valid_checkpoint
import world_grid as grid


class WorldImportTests(unittest.TestCase):
    def test_native_tiles_share_exact_grid_edges(self):
        a, b = grid.transform('h17v08'), grid.transform('h18v08')
        self.assertAlmostEqual(a.c + grid.SIZE * a.a, b.c, places=7)
        c = grid.transform('h17v09')
        self.assertAlmostEqual(a.f + grid.SIZE * a.e, c.f, places=7)
        self.assertLess(abs(grid.CELL - 4000), .2)

    def test_polar_padding_and_whole_earth_area(self):
        self.assertFalse(grid.domain_mask('h00v00').any())
        self.assertTrue(grid.domain_mask('h18v00').any())
        count = sum(int(grid.domain_mask(key).sum()) for key in grid.tiles())
        measured = count * grid.CELL ** 2
        self.assertLess(abs(measured / (4 * np.pi * grid.RADIUS ** 2) - 1), .0002)

    def test_fast_contexts_match_reference_with_missing_values(self):
        rng = np.random.default_rng(7)
        a = rng.normal(size=(36, 36))
        a[:4, :10] = np.nan
        a[19, 18] = np.nan
        expected = reference_contexts(a, 16)
        actual = contexts(a, 16)
        for name in expected:
            np.testing.assert_allclose(actual[name], expected[name], atol=1e-12, equal_nan=True)

    def test_area_weighted_mean_and_support_are_separate(self):
        a = np.array([[1., np.nan], [3., np.nan]])
        tr = from_origin(0, 2, 1, 1)
        target = from_origin(0, 2, 2, 2)
        mean = warp_array(a, tr, (1, 1), target)
        support = warp_array(np.isfinite(a).astype(float), tr, (1, 1), target, nodata=None, fill=0)
        self.assertAlmostEqual(mean.item(), 2)
        self.assertAlmostEqual(support.item(), .5)

    def test_global_projection_reads_constant_geographic_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'source.tif'
            with rasterio.open(path, 'w', driver='GTiff', width=40, height=40, count=1,
                               dtype='float32', nodata=-9999, crs='EPSG:4326',
                               transform=from_origin(-20, 20, 1, 1)) as dst:
                dst.write(np.full((40, 40), 3., dtype='float32'), 1)
            actual = read_projected(path, 'h18v08')
            np.testing.assert_allclose(actual, 3.)

    def test_resume_rejects_corrupt_or_partial_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / 'run'
            (out / 'sources').mkdir(parents=True)
            (out / 'checkpoints').mkdir()
            (out / 'sources/h18v08.json').write_text('{}')
            (out / 'shared.json').write_text('{}')
            layers = []
            for i in range(25):
                file = root / f'{i}.tif'
                file.write_bytes(b'complete')
                layers.append(dict(path=file.name, file_sha256=sha_file(file)))
            checkpoint = dict(run_identity='stable', layers=layers,
                              source_sha256=sha_file(out / 'sources/h18v08.json'),
                              shared_sha256=sha_file(out / 'shared.json'))
            with patch('world_features.ROOT', root):
                self.assertIsNone(valid_checkpoint(out, 'h18v08', 'stable'))
                atomic_json(out / 'checkpoints/h18v08.json', checkpoint)
                self.assertIsNotNone(valid_checkpoint(out, 'h18v08', 'stable'))
                self.assertIsNone(valid_checkpoint(out, 'h18v08', 'changed'))
                (root / '3.tif').write_bytes(b'interrupted write')
                self.assertIsNone(valid_checkpoint(out, 'h18v08', 'stable'))
                (root / '3.tif').write_bytes(b'complete')
                (out / 'sources/h18v08.json').write_text('{"changed":true}')
                self.assertIsNone(valid_checkpoint(out, 'h18v08', 'stable'))


if __name__ == '__main__':
    unittest.main()
