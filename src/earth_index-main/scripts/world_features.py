"""Bounded global native-tile processing for the existing 25 environmental bands.

The global grid and GDAL area resampling are explicitly versioned separately
from the local exact-polygon pilot. No physical source gaps are filled.
"""
import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'

from datetime import datetime
import hashlib
import json
from pathlib import Path
import time
import warnings

import numpy as np
from pyhdf.SD import SD, SDC
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

import priority_pilot as p
from expansion_features import reduce4
from raster_source_cache import asset_path, atomic_json
import thermal_import as thermal
import thermal_pilot as reference
import world_grid as grid

ROOT = Path(__file__).resolve().parents[1]


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def warp_array(values, src_transform, shape, dst_transform, *, src_crs=grid.CRS,
               method=Resampling.average, nodata=np.nan, fill=np.nan):
    dest = np.full(shape, fill, dtype=np.float64)
    reproject(values, dest, src_transform=src_transform, src_crs=src_crs,
              dst_transform=dst_transform, dst_crs=grid.CRS,
              src_nodata=nodata, dst_nodata=fill, resampling=method,
              num_threads=1, warp_mem_limit=64, ERROR_THRESHOLD=0)
    return dest


def read_projected(path, key, *, fine=False, halo=0, categorical=False, quality=False):
    n = grid.SIZE * (4 if fine else 1) + 2 * halo
    with rasterio.open(path) as source:
        return warp_array(rasterio.band(source, 1), source.transform, (n, n),
                          grid.transform(key, fine=fine, halo=halo), src_crs=source.crs,
                          method=Resampling.nearest if categorical else Resampling.average,
                          nodata=source.nodata, fill=0 if quality else np.nan)


def box_sum(a, width):
    integral = np.pad(a, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    return (integral[width:, width:] - integral[:-width, width:]
            - integral[width:, :-width] + integral[:-width, :-width])


def contexts(height, side):
    """Same 5/21-native-km definitions, separable extrema and integral sums."""
    result = {}
    for width in (5, 21):
        radius = width // 2
        start = 10 - radius
        stop = start + side
        finite = np.isfinite(height)
        counts = box_sum(finite.astype(float), width)[start:stop, start:stop]
        sums = box_sum(np.where(finite, height, 0.), width)[start:stop, start:stop]
        low = np.where(finite, height, np.inf)
        high = np.where(finite, height, -np.inf)
        for axis in (0, 1):
            low = np.min(np.lib.stride_tricks.sliding_window_view(low, width, axis=axis), axis=-1)
            high = np.max(np.lib.stride_tricks.sliding_window_view(high, width, axis=axis), axis=-1)
        valid = counts >= .9 * width * width
        relief = np.where(valid, high[start:stop, start:stop] - low[start:stop, start:stop], np.nan)
        means = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)
        tpi = np.where(valid, height[10:10+side, 10:10+side] - means, np.nan)
        result[f'relief_{width}km_m'] = reduce4(relief, minimum=.5)
        result[f'tpi_{width}km_m'] = reduce4(tpi, minimum=.5)
    return result


def terrain(key, shared):
    side = grid.SIZE * 4
    valid = read_projected(shared / 'terrain-valid.tif', key, fine=True, halo=10, quality=True)
    height = read_projected(shared / 'elevation.tif', key, fine=True, halo=10)
    slope = read_projected(shared / 'slope.tif', key, fine=True, halo=10)
    fine_domain = grid.domain_mask(key, fine=True, halo=10)
    height[~fine_domain] = np.nan
    slope[~fine_domain] = np.nan
    valid[~fine_domain] = 0
    crop = np.s_[10:-10, 10:-10]
    result = dict(elevation_median_m=reduce4(height[crop], valid[crop]),
                  slope_mean_deg=reduce4(slope[crop], valid[crop]),
                  terrain_valid_fraction=np.clip(reduce4(valid[crop]), 0, 1))
    result.update(contexts(height, side))
    return result


def cover(path, key):
    n = grid.SIZE
    if path is None:
        return dict(bare_fraction=np.full((n, n), np.nan),
                    permanent_snow_ice_fraction=np.full((n, n), np.nan),
                    landcover_valid_fraction=np.zeros((n, n)))
    handle = SD(str(path), SDC.READ)
    try:
        tr = p.grid_transform(handle.attributes()['StructMetadata.0'], (2400, 2400))
        arrays = []
        for name in ('LC_Type1', 'QC'):
            ds = handle.select(name)
            try:
                arrays.append(ds.get())
            finally:
                ds.endaccess()
    finally:
        handle.end()
    lc, qc = arrays
    good = np.isin(qc, [0, 2]) & (lc >= 1) & (lc <= 17)
    shape = (n, n)
    result = {name: warp_array(np.where(good, (lc == code).astype(float), np.nan),
                              tr, shape, grid.transform(key))
              for name, code in [('bare_fraction', 16), ('permanent_snow_ice_fraction', 15)]}
    result['landcover_valid_fraction'] = np.clip(warp_array(good.astype(float), tr, shape,
                                                           grid.transform(key), nodata=None, fill=0), 0, 1)
    return result


def surface(key, items, cache):
    means = [[], [], []]
    fractions = [[], [], []]
    dates = []
    days = []
    shape = (grid.SIZE, grid.SIZE)
    tr = grid.transform(key)
    for item in sorted(items, key=lambda i: i['id']):
        arrays = {}
        source_transform = None
        for band in reference.BANDS:
            with rasterio.open(asset_path(cache, item['assets'][band]['href'])) as src:
                if src.width != 1200 or src.height != 1200:
                    raise ValueError('Unexpected thermal source shape')
                if source_transform is not None and source_transform != src.transform:
                    raise ValueError('Thermal source band alignment mismatch')
                source_transform = src.transform
                arrays[band] = src.read(1, masked=False)
        day = np.where((arrays['LST_Day_1km'] >= 7500) & (arrays['QC_Day'] == 0),
                       arrays['LST_Day_1km'].astype(float) * .02 - 273.15, np.nan)
        night = np.where((arrays['LST_Night_1km'] >= 7500) & (arrays['QC_Night'] == 0),
                         arrays['LST_Night_1km'].astype(float) * .02 - 273.15, np.nan)
        for k, values in enumerate((day, night, (day + night) / 2)):
            means[k].append(warp_array(values, source_transform, shape, tr))
            fractions[k].append(np.clip(warp_array(np.isfinite(values).astype(float),
                                                  source_transform, shape, tr, nodata=None, fill=0), 0, 1))
        date = item['properties']['start_datetime'][:10]
        dates.append(date)
        days.append(min(8, (datetime(2024, 1, 1) - datetime.fromisoformat(date)).days))
    summaries = [reference.summarize(np.stack(means[k]), np.stack(fractions[k]), dates, days)
                 for k in range(3)] if dates else [[np.full(shape, np.nan) for _ in range(3)]
                                                  + [np.zeros(shape)] for _ in range(3)]
    return {key: array for key, array, _, _ in thermal.layers_from_summaries(summaries)}


def process_tile(job):
    """Publish all outputs before the final checkpoint; incomplete tiles redo safely."""
    key, out_text, identity = job
    out = Path(out_text)
    shared = out / 'shared'
    source = json.loads((out / 'sources' / f'{key}.json').read_text())
    start = time.perf_counter()
    cpu = time.process_time()
    with rasterio.Env(GDAL_CACHEMAX=64 * 1024 * 1024, GDAL_NUM_THREADS='1'):
        values = terrain(key, shared)
        values.update(cover(out / source['landcover_path'] if source['landcover_path'] else None, key))
        values.update(surface(key, source['items'], out / 'raw-cache' / key))
        for field in ('coldest_month_air_c', 'warmest_month_air_c', 'annual_precip_mm'):
            a = read_projected(shared / (field + '.tif'), key)
            support = read_projected(shared / (field + '-valid.tif'), key, quality=True)
            a[support < .999] = np.nan
            values[field] = a
        values['permafrost_fraction'] = read_projected(shared / 'permafrost.tif', key)
        values['permafrost_valid_fraction'] = np.clip(read_projected(shared / 'permafrost-valid.tif', key, quality=True), 0, 1)
        values['dominant_lithology_code'] = read_projected(shared / 'lithology.tif', key, categorical=True)
    specs = {l['feature_key']: l for l in json.loads((ROOT / 'research/thermal-pilot/manifest.json').read_text())['layers']}
    if set(values) != set(specs):
        raise ValueError('Global feature set differs from all 25 pilot bands')
    domain = grid.domain_mask(key)
    layers = []
    folder = out / 'tiles' / key
    folder.mkdir(parents=True, exist_ok=True)
    tr = grid.transform(key)
    for name, a in sorted(values.items()):
        a[~domain] = np.nan
        if np.isinf(a).any():
            raise ValueError(f'Infinite physical values: {key}/{name}')
        spec = specs[name].copy()
        if name == 'annual_precip_mm':
            # The pilot's .1mm encoding overflows above 6553.4mm/year. Global
            # wet tropical cells need a wider range while retaining two bytes.
            spec.update(scale=1., offset=-1.)
        codes, error = p.quantize(a, spec['scale'], spec['offset'])
        path = folder / (name + '.tif')
        temporary = path.with_suffix('.part.tif')
        with rasterio.open(temporary, 'w', driver='GTiff', width=grid.SIZE, height=grid.SIZE,
                           count=1, dtype='uint16', nodata=0, crs=grid.CRS, transform=tr,
                           compress='deflate', predictor=2, tiled=True, blockxsize=128, blockysize=128) as dst:
            dst.write(codes, 1)
        temporary.replace(path)
        layer = {k: spec[k] for k in ('feature_key', 'variable_key', 'unit', 'scale', 'offset', 'nodata')}
        layer.update(site_key=key, path=str(path.relative_to(ROOT)), file_sha256=sha_file(path),
                     codes_sha256=hashlib.sha256(codes.tobytes()).hexdigest(),
                     grid_definition=dict(shape=[grid.SIZE, grid.SIZE], transform=list(tr)[:6],
                                          crs_wkt=rasterio.crs.CRS.from_string(grid.CRS).to_wkt(), version=grid.VERSION),
                     valid_cells=int(np.isfinite(a).sum()), max_rounding_error=error,
                     physical_min=float(np.nanmin(a)) if np.isfinite(a).any() else None,
                     physical_max=float(np.nanmax(a)) if np.isfinite(a).any() else None,
                     source=dict(method='global-native-area-v1', source_inventory=f'sources/{key}.json',
                                 source_inventory_sha256=sha_file(out / 'sources' / f'{key}.json'),
                                 scientific_methods='METHODS.md'))
        layers.append(layer)
    checkpoint = dict(site_key=key, run_identity=identity, layers=layers,
                      seconds=time.perf_counter() - start, cpu_seconds=time.process_time() - cpu,
                      domain_cells=int(domain.sum()), source_sha256=sha_file(out / 'sources' / f'{key}.json'),
                      shared_sha256=sha_file(out / 'shared.json'))
    atomic_json(out / 'checkpoints' / f'{key}.json', checkpoint)
    return checkpoint


def valid_checkpoint(out, key, identity):
    try:
        result = json.loads((out / 'checkpoints' / f'{key}.json').read_text())
        if (result['run_identity'] != identity or len(result['layers']) != 25
                or result['source_sha256'] != sha_file(out / 'sources' / f'{key}.json')
                or result['shared_sha256'] != sha_file(out / 'shared.json')):
            return None
        if all(sha_file(ROOT / l['path']) == l['file_sha256'] for l in result['layers']):
            return result
    except (OSError, ValueError, KeyError):
        pass
    return None
