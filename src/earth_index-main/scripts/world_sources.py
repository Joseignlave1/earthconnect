"""Restartable, checksummed global inputs; no access tokens are written to reports."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import zipfile

import h5py
import numcodecs
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window

import priority_pilot as p
from raster_source_cache import SourceCache, atomic_json
import thermal_pilot as reference
from world_features import sha_file
import world_grid as grid

POWER = 'https://nasa-power.s3.us-west-2.amazonaws.com/merra2/temporal/power_merra2_climatology_temporal_lst.zarr/'


def log(message):
    print(datetime.utcnow().isoformat(timespec='seconds') + 'Z ' + message, flush=True)


def fetch(url, path):
    """A completed file is usable only with matching atomic checksum metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = path.with_name(path.name + '.json')
    try:
        old = json.loads(meta.read_text())
        if old['url'] == url and old['bytes'] == path.stat().st_size and old['sha256'] == sha_file(path):
            return old
    except (OSError, ValueError, KeyError):
        pass
    tmp = path.with_name(path.name + '.part')
    for attempt in range(6):
        try:
            digest = hashlib.sha256()
            size = 0
            with urlopen(Request(url, headers={'User-Agent': 'EarthIndex/world-import'}), timeout=120) as response, tmp.open('wb') as stream:
                expected = response.headers.get('Content-Length')
                while block := response.read(4 * 1024 * 1024):
                    stream.write(block)
                    digest.update(block)
                    size += len(block)
                if expected and size != int(expected):
                    raise ValueError('Truncated source transfer')
            tmp.replace(path)
            result = dict(url=url, bytes=size, sha256=digest.hexdigest())
            atomic_json(meta, result)
            return result
        except Exception:
            tmp.unlink(missing_ok=True)
            if attempt == 5:
                raise RuntimeError(f'Public source transfer failed: {path.name}') from None
            time.sleep(min(30, 2 ** attempt))


def thermal_catalog(out):
    directory = out / 'catalog'
    directory.mkdir(exist_ok=True)
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(0, 365, 8)]

    def one(date):
        file = directory / (date.strftime('%Y-%j') + '.json')
        if file.exists():
            return json.loads(file.read_text())
        url = reference.STAC + '/search?' + urlencode(dict(collections='modis-11A2-061',
            datetime=date.strftime('%Y-%m-%dT00:00:00Z/') + date.strftime('%Y-%m-%dT23:59:59Z'),
            query=json.dumps({'platform': {'eq': 'terra'}}), limit=1000))
        items = []
        while url:
            data = reference.get_json(url)
            for item in data['features']:
                items.append(dict(id=item['id'], properties=dict(start_datetime=item['properties']['start_datetime']),
                                  assets={b: dict(href=item['assets'][b]['href']) for b in reference.BANDS}))
            url = next((link['href'] for link in data.get('links', []) if link['rel'] == 'next'), None)
            if url and not url.startswith(reference.STAC + '/'):
                raise ValueError('Unexpected catalog pagination host')
        atomic_json(file, items)
        log(f'Catalog saved {date.date()}: {len(items)} thermal granules')
        return items

    with ThreadPoolExecutor(max_workers=4) as pool:
        groups = list(pool.map(one, dates))
    if sum(bool(group) for group in groups) < 40:
        raise ValueError('Global annual thermal catalog is unexpectedly incomplete')
    result = {}
    for item in (i for group in groups for i in group):
        import re
        match = re.fullmatch(r'MOD11A2\.A2023\d{3}\.(h\d{2}v\d{2})\.061\.\d+', item['id'])
        if not match:
            raise ValueError('Unexpected global thermal product/year')
        result.setdefault(match[1], []).append(item)
    if len({i['id'] for group in result.values() for i in group}) != sum(map(len, result.values())):
        raise ValueError('Duplicate thermal granules')
    return result


def cover_catalog(out):
    path = out / 'catalog' / 'landcover.json'
    if path.exists():
        return json.loads(path.read_text())
    url = 'https://cmr.earthdata.nasa.gov/search/granules.json?' + urlencode(dict(short_name='MCD12Q1', version='061',
          temporal='2024-01-01T00:00:00Z,2024-01-02T00:00:00Z', page_size=1000))
    data = reference.get_json(url)['feed']['entry']
    if len(data) >= 1000:
        raise ValueError('Land-cover catalog pagination required')
    import re
    result = {re.search(r'\.(h\d{2}v\d{2})\.', g['title'])[1]: dict(title=g['title']) for g in data}
    atomic_json(path, result)
    return result


def write_raster(path, values, tr, *, nodata=np.nan, dtype='float32'):
    tmp = path.with_name(path.stem + '.part.tif')
    with rasterio.open(tmp, 'w', driver='GTiff', height=values.shape[0], width=values.shape[1],
                       count=1, dtype=dtype, nodata=nodata, crs='EPSG:4326', transform=tr,
                       compress='deflate', tiled=True, blockxsize=256, blockysize=256) as dst:
        dst.write(values.astype(dtype), 1)
    tmp.replace(path)


def power(out):
    """Fetch only the two small climatology arrays from the public bulk store."""
    folder = out / 'raw-cache' / 'power'
    record = fetch(POWER + '.zmetadata', folder / 'metadata')
    metadata = json.loads((folder / 'metadata').read_text())['metadata']
    records = [record]

    def decode(name, indices):
        spec = metadata[name + '/.zarray']
        suffix = '.'.join(map(str, indices))
        path = folder / (name + '-' + suffix)
        rec = fetch(POWER + name + '/' + suffix, path)
        payload = numcodecs.get_codec(spec['compressor']).decode(path.read_bytes())
        for filt in reversed(spec.get('filters') or []):
            payload = numcodecs.get_codec(filt).decode(payload)
        a = np.frombuffer(payload, dtype=spec['dtype']).reshape(spec['chunks'], order=spec['order'])
        return a, rec

    coordinates = {}
    for name in ('time', 'lat', 'lon'):
        a, rec = decode(name, [0])
        coordinates[name] = a
        records.append(rec)
    if not np.array_equal(coordinates['time'], np.arange(1, 14)):
        raise ValueError('Unexpected POWER climatology month order')
    lat, lon = coordinates['lat'], coordinates['lon']
    if not (len(lat) == 361 and len(lon) == 576 and np.allclose(np.diff(lat), .5) and np.allclose(np.diff(lon), .625)):
        raise ValueError('Unexpected POWER native climate grid')
    tr = from_origin(lon[0] - .3125, lat[-1] + .25, .625, .5)
    baseline = metadata['.zattrs']
    start, end = (int(baseline['time_coverage_' + k][:4]) for k in ('start', 'end'))
    import calendar
    days = np.mean([366 if calendar.isleap(y) else 365 for y in range(start, end + 1)])
    for name, unit in [('T2M', 'C'), ('PRECTOTCORR', 'mm/day')]:
        spec = metadata[name + '/.zarray']
        if spec['shape'] != [13, 361, 576] or metadata[name + '/.zattrs']['units'] != unit:
            raise ValueError('Unexpected POWER source encoding')
        values = np.full(spec['shape'], np.nan, dtype='float32')
        jobs = [(0, row, col) for row in range(math.ceil(361 / spec['chunks'][1]))
                for col in range(math.ceil(576 / spec['chunks'][2]))]
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(decode, name, index): index for index in jobs}
            for future in as_completed(futures):
                index = futures[future]
                array, rec = future.result()
                records.append(rec)
                y, x = index[1] * spec['chunks'][1], index[2] * spec['chunks'][2]
                nr, nc = min(array.shape[1], 361 - y), min(array.shape[2], 576 - x)
                values[:, y:y+nr, x:x+nc] = array[:, :nr, :nc]
        attrs = metadata[name + '/.zattrs']
        values[(values < attrs['valid_min']) | (values > attrs['valid_max'])] = np.nan
        monthly = values[:12]
        # Native monthly climatology first; no independent 4km climate information.
        valid = np.isfinite(monthly).all(axis=0)
        if name == 'T2M':
            fields = dict(coldest_month_air_c=np.where(valid, np.min(monthly, axis=0), np.nan),
                          warmest_month_air_c=np.where(valid, np.max(monthly, axis=0), np.nan))
        else:
            fields = dict(annual_precip_mm=np.where(valid & (values[12] >= 0), values[12] * days, np.nan))
        for key, a in fields.items():
            # Longitude is periodic: retain the -180-centered support at +180.
            wrapped = np.concatenate((a, a[:, :1]), axis=1)
            write_raster(out / 'shared' / (key + '.tif'), wrapped[::-1], tr)
            write_raster(out / 'shared' / (key + '-valid.tif'), np.isfinite(wrapped[::-1]), tr,
                         nodata=None, dtype='uint8')
    atomic_json(out / 'sources' / 'power.json', dict(records=records, metadata=baseline,
                 monthly_coordinates=coordinates['time'].tolist(), annual_precip_days=float(days)))


def terrain_mask(shared):
    path = shared / 'terrain-valid.tif'
    tmp = path.with_suffix('.part.tif')
    with rasterio.open(shared / 'slope.tif') as src:
        profile = src.profile | dict(dtype='uint8', nodata=None, compress='deflate', count=1, tiled=True,
                                     blockxsize=256, blockysize=256)
        with rasterio.open(tmp, 'w', **profile) as dst:
            for _, window in dst.block_windows(1):
                a = src.read(1, window=window, masked=True)
                dst.write((~np.ma.getmaskarray(a) & np.isfinite(a.data)).astype('uint8'), 1, window=window)
    tmp.replace(path)


def permafrost(shared):
    with h5py.File(shared / 'permafrost.nc', 'r') as source:
        data = source['PFR']
        lat, lon = source['lat'][:], source['lon'][:]
        # CF coordinates are float32; whole-vector roundoff reaches ~3e-5 deg.
        if (data.shape != (1, len(lat), len(lon))
                or not np.allclose(lat, float(lat[0]) + np.arange(len(lat)) * .01, atol=4e-5, rtol=0)
                or not np.allclose(lon, float(lon[0]) + np.arange(len(lon)) * .01, atol=4e-5, rtol=0)):
            raise ValueError('Unexpected CCI permafrost CF coordinates')
        tr = from_origin(round(float(lon[0]) - .005, 4), round(float(lat[-1]) + .005, 4), .01, .01)
        profile = dict(driver='GTiff', width=len(lon), height=len(lat), count=1, crs='EPSG:4326',
                       transform=tr, compress='deflate', tiled=True, blockxsize=256, blockysize=256)
        paths = [shared / 'permafrost.tif', shared / 'permafrost-valid.tif']
        temps = [p.with_suffix('.part.tif') for p in paths]
        with rasterio.open(temps[0], 'w', **profile, dtype='float32', nodata=np.nan) as dst, \
                rasterio.open(temps[1], 'w', **profile, dtype='uint8') as quality:
            for row in range(0, len(lat), 128):
                n = min(128, len(lat) - row)
                a = data[0, len(lat)-row-n:len(lat)-row, :][::-1]
                good = a <= 100
                window = Window(0, row, len(lon), n)
                dst.write(np.where(good, a.astype('float32') / 100, np.nan), 1, window=window)
                quality.write(good.astype('uint8'), 1, window=window)
        for tmp, path in zip(temps, paths):
            tmp.replace(path)


def lithology(shared):
    with zipfile.ZipFile(shared / 'lithology.zip') as z:
        lines = z.read('glim_wgs84_0point5deg.txt.asc').decode().splitlines()
    a = np.loadtxt(io.StringIO('\n'.join(lines[6:])), dtype='int16')
    if a.shape != (360, 720):
        raise ValueError('Unexpected GLiM source shape')
    a[a == 15] = -9999
    write_raster(shared / 'lithology.tif', a, from_origin(-180, 90, .5, .5), nodata=-9999, dtype='int16')


def shared_sources(out):
    checkpoint = out / 'shared.json'
    if checkpoint.exists():
        try:
            old = json.loads(checkpoint.read_text())
            if all(sha_file(out / name) == checksum for name, checksum in old['files'].items()):
                return old
        except (OSError, ValueError, KeyError):
            pass
    shared = out / 'shared'
    shared.mkdir(exist_ok=True)
    inputs = [(p.TOPO_BASE + 'elevation/elevation_1KMmd_GMTEDmd.tif', shared / 'elevation.tif'),
              (p.TOPO_BASE + 'slope/slope_1KMmn_GMTEDmd.tif', shared / 'slope.tif'),
              (p.PERMA_URL, shared / 'permafrost.nc'), (p.GLIM_URL, shared / 'lithology.zip')]
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(lambda job: fetch(*job), inputs))
    log('Shared terrain, permafrost and lithology downloads verified')
    terrain_mask(shared)
    log('Terrain support mask prepared')
    permafrost(shared)
    lithology(shared)
    log('Permafrost and lithology prepared')
    power(out)
    log('Global bulk POWER climatology prepared')
    files = {str(path.relative_to(out)): sha_file(path) for path in shared.glob('*')
             if path.suffix in ('.tif', '.nc', '.zip')}
    files['sources/power.json'] = sha_file(out / 'sources' / 'power.json')
    result = dict(files=files, records=records, power_baseline='2001-2020 public global climatology')
    atomic_json(checkpoint, result)
    return result


def tile_sources(key, items, granule, out, token_file, workers=4):
    folder = out / 'raw-cache' / key
    folder.mkdir(parents=True, exist_ok=True)
    hrefs = [i['assets'][b]['href'] for i in items for b in reference.BANDS]
    records = SourceCache(folder).fetch_all(hrefs, workers=workers) if hrefs else []
    land = None
    land_path = None
    if granule:
        path = folder / (granule['title'] + '.hdf')
        meta = path.with_suffix('.json')
        try:
            land = json.loads(meta.read_text())
            if land['sha256'] != sha_file(path):
                land = None
        except (OSError, ValueError, KeyError):
            land = None
        if land is None:
            payload, land = p.nasa_download(granule['title'], Path(token_file).read_text().strip())
            tmp = path.with_suffix('.part')
            tmp.write_bytes(payload)
            tmp.replace(path)
            atomic_json(meta, land)
        land_path = str(path.relative_to(out))
    result = dict(tile=key, items=items, thermal=[{k: r[k] for k in ('href', 'sha256', 'bytes')} for r in records],
                  landcover=land, landcover_path=land_path)
    inventory = out / 'sources' / (key + '.json')
    if inventory.exists():
        old = json.loads(inventory.read_text())
        if old != result:
            raise ValueError('Previously inventoried source bytes changed; refusing to mix versions')
    else:
        atomic_json(inventory, result)
    return result
