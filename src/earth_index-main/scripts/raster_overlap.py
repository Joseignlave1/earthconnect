"""Reusable conservative raster-to-cell weights, using exactextract's geometry.

Construct once for a fixed source grid and target geometry. Applying changing
values/quality masks uses single-threaded NumPy reductions, without revisiting
polygons. Both Cartesian and spherical-area weights come from exactextract
itself, so they follow the reference processor's area convention.
"""
from dataclasses import dataclass
import math

import numpy as np
from exactextract import exact_extract
from exactextract.raster import NumPyRasterSource
from rasterio.transform import array_bounds


@dataclass(frozen=True)
class OverlapPlan:
    source_shape: tuple
    target_shape: tuple
    source_ids: np.ndarray
    target_ids: np.ndarray
    weights: np.ndarray
    total: np.ndarray

    @classmethod
    def build(cls, source_shape, transform, crs_wkt, features, *, geographic=False, target_shape=None):
        source_shape = tuple(source_shape)
        if len(source_shape) != 2 or min(source_shape) < 1:
            raise ValueError('Expected a nonempty two-dimensional source grid')
        if transform.b or transform.d or transform.a <= 0 or transform.e >= 0:
            raise ValueError('Expected a north-up source grid without rotation')
        count = len(features)
        if not count:
            raise ValueError('Expected target cells')
        if target_shape is None:
            side = math.isqrt(count)
            if side*side != count:
                raise ValueError('Specify target_shape for a nonsquare target')
            target_shape = (side, side)
        target_shape = tuple(target_shape)
        if len(target_shape) != 2 or math.prod(target_shape) != count:
            raise ValueError('Target shape does not match feature count')
        if sorted(f['properties']['i'] for f in features) != list(range(count)):
            raise ValueError('Target feature IDs must uniquely cover 0..N-1')
        source = NumPyRasterSource(np.ones(source_shape, dtype='uint8'),
                                  *array_bounds(*source_shape, transform), srs_wkt=crs_wkt)
        weighting = 'area_spherical_m2' if geographic else 'area_cartesian'
        records = exact_extract(source, features, ['cell_id', f'coverage(coverage_weight={weighting})'], include_cols=['i'])
        source_ids = []; target_ids = []; weights = []
        for record in records:
            props = record['properties']
            ids = np.asarray(props['cell_id'], dtype=np.int64)
            areas = np.asarray(props['coverage'], dtype=np.float64)
            source_ids.append(ids)
            target_ids.append(np.full(ids.size, props['i'], dtype=np.int64))
            weights.append(areas)
        source_ids = np.concatenate(source_ids)
        target_ids = np.concatenate(target_ids)
        weights = np.concatenate(weights)
        total = np.bincount(target_ids, weights=weights, minlength=count).reshape(target_shape)
        for array in (source_ids, target_ids, weights, total):
            array.setflags(write=False)
        return cls(source_shape, target_shape, source_ids, target_ids, weights, total)

    @property
    def nbytes(self):
        return sum(a.nbytes for a in (self.source_ids, self.target_ids, self.weights, self.total))

    def aggregate(self, values):
        if np.ma.isMaskedArray(values):
            values = np.ma.asarray(values, dtype=np.float64).filled(np.nan)
        values = np.asarray(values, dtype=np.float64)
        if values.shape != self.source_shape:
            raise ValueError('Values do not match the source grid used to build weights')
        if np.isinf(values).any():
            raise ValueError('Infinite values must be normalized to NODATA before aggregation')
        selected = values.ravel()[self.source_ids]
        valid = ~np.isnan(selected)
        # Keep input order fixed; never sum partial means or average away QC masks.
        areas = np.where(valid, self.weights, 0.)
        count = math.prod(self.target_shape)
        accepted = np.bincount(self.target_ids, weights=areas, minlength=count)
        sums = np.bincount(self.target_ids, weights=np.where(valid, selected, 0.)*areas, minlength=count)
        mean = np.divide(sums, accepted, out=np.full(count, np.nan), where=accepted > 0)
        return mean.reshape(self.target_shape), accepted.reshape(self.target_shape), self.total
