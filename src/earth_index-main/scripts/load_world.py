# /// script
# requires-python = ">=3.11"
# dependencies = ["psycopg[binary]==3.3.9", "h3==4.5.0", "rasterio==1.5.2", "numpy==2.5.4"]
# ///
"""Verify and transactionally import the completed world rasters into PostGIS.

One transaction per native region, with byte-exact WKB verification before commit.
Restarting reuses verified committed regions. Publication happens only at the end.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import struct
import time
import uuid

import numpy as np
import rasterio
from psycopg.types.json import Jsonb

from load_raster_pilot import ROOT, connection
from load_pilots import migrate
import world_grid as grid

SRID = 930001


def atomic_json(path, value):
    temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.part')
    try:
        temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(4 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def log(message):
    print(datetime.now(timezone.utc).isoformat(timespec='seconds') + ' ' + message, flush=True)


def check_wkb(payload, layer):
    """PostGIS RFC2 transport format, one in-db U16 band; no float expansion."""
    data = bytes(payload)
    if not data or data[0] not in (0, 1):
        raise ValueError('Invalid raster WKB byte order')
    order = '<' if data[0] else '>'
    header = struct.unpack(order + 'BHHddddddiHH', data[:61])
    _, version, bands, sx, sy, x, y, skewx, skewy, srid, width, height = header
    g = layer['grid_definition']
    expected = g['transform']
    if (version != 0 or bands != 1 or srid != SRID or [height, width] != g['shape']
            or [sx, skewx, x, skewy, sy, y] != expected):
        raise ValueError('PostGIS raster header disagrees with declared grid')
    if (len(data) != 64 + width * height * 2 or data[61] & 128
            or not data[61] & 64 or data[61] & 15 != 6 or struct.unpack(order+'H', data[62:64])[0] != 0):
        raise ValueError('Expected one inline 16BUI band with NODATA zero')
    codes = np.frombuffer(data, dtype=order+'u2', offset=64).astype('<u2', copy=False)
    if hashlib.sha256(codes.tobytes()).hexdigest() != layer['codes_sha256']:
        raise ValueError('PostGIS code roundtrip mismatch')


def validate(out):
    start = time.perf_counter()
    manifest = json.loads((out/'manifest.json').read_text())
    plan = json.loads((out/'plan.json').read_text())
    if manifest['run_identity'] != hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest():
        raise ValueError('Frozen processor/plan identity mismatch')
    for name, checksum in plan['dependencies'].items():
        if digest(ROOT/'scripts'/name) != checksum:
            raise ValueError(f'Frozen scientific dependency changed: {name}')
    if digest(out/'METHODS.md') != plan['methods_sha256']:
        raise ValueError('Scientific methods document changed')
    if digest(ROOT/'research/thermal-pilot/manifest.json') != plan['spec_sha256']:
        raise ValueError('Quantization specification changed')
    for name, checksum in plan['thermal_catalog_hashes'].items():
        if digest(out/'catalog'/name) != checksum:
            raise ValueError('Source catalog hash mismatch')
    if digest(out/'catalog/landcover.json') != plan['landcover_catalog_sha256']:
        raise ValueError('Land-cover catalog hash mismatch')
    shared = json.loads((out/'shared.json').read_text())
    for name, checksum in shared['files'].items():
        if digest(out/name) != checksum:
            raise ValueError('Shared source hash mismatch')
    shared_sha = digest(out/'shared.json')
    expected_keys = set(plan['tiles'])
    if set(manifest['tiles']) != expected_keys or manifest['grid'] != grid.definition():
        raise ValueError('Global grid/domain identity mismatch')
    specs = {l['feature_key'] for l in json.loads((ROOT/'research/thermal-pilot/manifest.json').read_text())['layers']}
    groups = {key: {} for key in expected_keys}
    for layer in manifest['layers']:
        group = groups[layer['site_key']]
        if layer['feature_key'] in group:
            raise ValueError('Duplicate layer')
        group[layer['feature_key']] = layer
    checked_codes = valid_values = domain_cells = 0
    for index, key in enumerate(sorted(groups)):
        group = groups[key]
        if set(group) != specs or len(group) != 25:
            raise ValueError('Incomplete feature set')
        checkpoint = json.loads((out/'checkpoints'/f'{key}.json').read_text())
        if (checkpoint['run_identity'] != manifest['run_identity'] or checkpoint['shared_sha256'] != shared_sha
                or checkpoint['source_sha256'] != digest(out/'sources'/f'{key}.json')
                or {l['feature_key']: l for l in checkpoint['layers']} != group):
            raise ValueError('Invalid region checkpoint')
        domain = grid.domain_mask(key)
        domain_cells += int(domain.sum())
        values = {}
        for name, layer in group.items():
            path = (ROOT/layer['path']).resolve()
            if not path.is_relative_to(out) or digest(path) != layer['file_sha256']:
                raise ValueError('Raster file path/hash mismatch')
            with rasterio.open(path) as source:
                a = source.read(1)
                g = layer['grid_definition']
                if (source.count != 1 or source.dtypes != ('uint16',) or source.nodata != 0
                        or list(source.shape) != [grid.SIZE, grid.SIZE]
                        or source.transform != grid.transform(key)
                        or list(source.transform)[:6] != g['transform']
                        or source.crs != rasterio.crs.CRS.from_string(grid.CRS)
                        or source.crs != rasterio.crs.CRS.from_wkt(g['crs_wkt'])):
                    raise ValueError('Declared grid disagrees with TIFF')
            if (int((a != 0).sum()) != layer['valid_cells'] or (a[~domain] != 0).any()
                    or hashlib.sha256(a.astype('<u2').tobytes()).hexdigest() != layer['codes_sha256']
                    or layer['max_rounding_error'] > layer['scale']/2+1e-8):
                raise ValueError('Raster codes/validity/quantization mismatch')
            v = np.where(a == 0, np.nan, a.astype(float)*layer['scale']+layer['offset'])
            if np.isinf(v).any():
                raise ValueError('Nonfinite physical values')
            if name.endswith('_fraction') and np.any((v < -1e-9) | (v > 1+1e-9)):
                raise ValueError('Coverage/fraction outside 0..1')
            if (name.startswith('relief_') or name == 'annual_precip_mm') and np.any(v < -1e-9):
                raise ValueError('Negative unsigned physical quantity')
            if name == 'dominant_lithology_code' and not set(a[a != 0]).issubset({*range(1,15),16}):
                raise ValueError('Unexpected lithology category')
            values[name] = v
            checked_codes += a.size
            valid_values += layer['valid_cells']
        if np.any(values['surface_low_c'] > values['surface_high_c']+.0200001):
            raise ValueError('Inverted thermal bounds')
        error = np.abs(values['surface_high_c']-values['surface_low_c']-values['surface_range_c'])
        if np.any(error > .0300001):
            raise ValueError('Thermal range consistency failed')
        if np.any(values['coldest_month_air_c'] > values['warmest_month_air_c']+.0200001):
            raise ValueError('Inverted air-climate bounds')
        if index % 40 == 0:
            log(f'File validation {index+1}/{len(groups)} regions')
    if domain_cells != manifest['total_domain_cells']:
        raise ValueError('Global domain-cell count mismatch')
    identity = {k:v for k,v in manifest.items() if k not in ('generated_at','status','publication','source_rasters_retained')}
    bundle = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',',':')).encode()).hexdigest()
    report = dict(status='passed', bundle_sha256=bundle, regions=len(groups), layers=len(manifest['layers']),
                  checked_codes=checked_codes, valid_values=valid_values, domain_cells=domain_cells,
                  wall_seconds=time.perf_counter()-start)
    atomic_json(out/'database-file-validation.json', report)
    log(f'All {checked_codes:,} U16 codes and {len(groups)} source/output checkpoints validated')
    return manifest, groups, bundle, report


def snapshot(conn, exclude):
    return conn.execute('''SELECT bundle_sha256,count(*),sum(valid_cells),
        md5(string_agg(layer_id::text||file_sha256,',' ORDER BY layer_id))
        FROM catalog.raster_layers WHERE bundle_sha256<>%s GROUP BY bundle_sha256 ORDER BY bundle_sha256''', (exclude,)).fetchall()


def load(out, data, groups, bundle, file_report, publish):
    start = time.perf_counter()
    ns = 'raster_'+bundle[:32]
    status = dict(status='loading', bundle_sha256=bundle, schema_name=ns, datasource_id='earth-'+bundle[:12],
                  total_regions=len(groups), completed_regions=0, publish_requested=publish)
    methods_sha = json.loads((out/'plan.json').read_text())['methods_sha256']
    with connection() as conn:
        conn.commit()
        conn.autocommit = True
        if not conn.execute('SELECT pg_try_advisory_lock(4182026,3)').fetchone()[0]:
            raise RuntimeError('Another global database loader is running')
        with conn.transaction():
            conn.execute('SELECT pg_advisory_xact_lock(4182026,1)')
            migrate(conn)
            before = snapshot(conn, bundle)
            atomic_json(out/'database-before-load.json', [list(row) for row in before])
            crs = rasterio.crs.CRS.from_string(grid.CRS).to_wkt()
            previous = conn.execute('SELECT srtext,proj4text FROM spatial_ref_sys WHERE srid=%s', (SRID,)).fetchone()
            if previous and previous != (crs, grid.CRS):
                raise ValueError('Global SRID collision')
            conn.execute('''INSERT INTO spatial_ref_sys(srid,auth_name,auth_srid,srtext,proj4text)
                VALUES (%s,'earth_index',%s,%s,%s) ON CONFLICT(srid) DO NOTHING''', (SRID,SRID,crs,grid.CRS))
            # Do not remove an already-published complete bundle on an idempotent rerun.
            conn.execute('INSERT INTO catalog.raster_bundle_publication VALUES(%s,false) ON CONFLICT DO NOTHING', (bundle,))
        verified = reused = 0
        for index, (key, group) in enumerate(sorted(groups.items())):
            with conn.transaction():
                conn.execute("SET LOCAL postgis.gdal_enabled_drivers='GTiff'")
                previous = dict(conn.execute('SELECT feature_key,file_sha256 FROM catalog.raster_layers WHERE bundle_sha256=%s AND site_key=%s', (bundle,key)).fetchall())
                if previous and previous != {name:l['file_sha256'] for name,l in group.items()}:
                    raise ValueError('Committed region is incomplete or conflicts with source identity')
                if not previous:
                    with conn.pipeline():
                        for name, layer in sorted(group.items()):
                            g = layer['grid_definition'].copy()
                            g.update(projected_pixel_area_m2=grid.CELL**2, area_note='MODIS sinusoidal equal-area sphere; NODATA outside Earth.',
                                     global_shape=data['grid']['global_shape'], global_origin=data['grid']['origin'],
                                     native_tile_indices=list(grid.tile_indices(key)), nodata=0)
                            provenance = layer['source'] | dict(methods_sha256=methods_sha,
                                source_inventory_path=str((out/'sources'/f'{key}.json').relative_to(ROOT)),
                                power_climatology_baseline='2001-2020', thermal_year=2023,landcover_year=2024)
                            conn.execute('''WITH inserted AS (
                                INSERT INTO catalog.raster_layers(bundle_sha256,site_key,feature_key,variable_key,unit,scale,offset_value,
                                    srid,grid_definition,provenance,valid_cells,max_rounding_error,file_sha256)
                                VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING layer_id)
                                INSERT INTO data.raster_tiles SELECT layer_id,0,0,ST_FromGDALRaster(%s,%s) FROM inserted''',
                                (bundle,key,name,layer['variable_key'],layer['unit'],layer['scale'],layer['offset'],SRID,
                                 Jsonb(g),Jsonb(provenance),layer['valid_cells'],layer['max_rounding_error'],layer['file_sha256'],
                                 (ROOT/layer['path']).read_bytes(),SRID))
                rows = conn.execute('''SELECT l.feature_key,ST_AsBinary(t.rast)
                    FROM catalog.raster_layers l JOIN data.raster_tiles t USING(layer_id)
                    WHERE l.bundle_sha256=%s AND l.site_key=%s ORDER BY l.feature_key''',(bundle,key)).fetchall()
                if len(rows) != 25:
                    raise ValueError('Incomplete region raster data')
                for name, payload in rows:
                    check_wkb(payload, group[name])
                    verified += grid.SIZE**2
                tr = grid.transform(key)
                conn.execute('''INSERT INTO catalog.raster_region_coverage VALUES(%s,%s,public.sinusoidal_footprint(%s,%s,%s,%s))
                    ON CONFLICT(bundle_sha256,site_key) DO NOTHING''',
                    (bundle,key,tr.c,tr.f+grid.SIZE*tr.e,tr.c+grid.SIZE*tr.a,tr.f))
                if previous:
                    reused += 1
            status.update(completed_regions=index+1, reused_regions=reused, verified_codes=verified,
                          updated_at=datetime.now(timezone.utc).isoformat())
            atomic_json(out/'database-load-status.json', status)
            if index % 10 == 0 or index+1 == len(groups):
                log(f'Database committed and byte-verified {index+1}/{len(groups)} regions ({(index+1)*25} layers)')
        with conn.transaction():
            totals = conn.execute('''SELECT count(*),sum(l.valid_cells),sum(pg_column_size(t.rast))
                FROM catalog.raster_layers l JOIN data.raster_tiles t USING(layer_id) WHERE bundle_sha256=%s''',(bundle,)).fetchone()
            if totals[0] != len(data['layers']) or totals[1] != file_report['valid_values']:
                raise ValueError('Global database totals mismatch')
            if snapshot(conn,bundle) != before:
                raise ValueError('Existing bundles changed during global load')
            if publish:
                conn.execute('UPDATE catalog.raster_bundle_publication SET published=true WHERE bundle_sha256=%s', (bundle,))
            conn.execute('SELECT catalog.refresh_api_datasources()')
            conn.execute('ANALYZE catalog.raster_layers')
            conn.execute('ANALYZE data.raster_tiles')
            conn.execute('ANALYZE catalog.raster_region_coverage')
        # The serving dataset is immutable; prepare visibility maps/statistics.
        conn.execute('VACUUM (FREEZE, ANALYZE) catalog.raster_layers')
        conn.execute('VACUUM (FREEZE, ANALYZE) data.raster_tiles')
        report = status | dict(status='completed', published=bool(conn.execute('SELECT published FROM catalog.raster_bundle_publication WHERE bundle_sha256=%s',(bundle,)).fetchone()[0]),
                              layers=totals[0],valid_values=totals[1],raster_datum_bytes=totals[2],
                              database_bytes=conn.execute('SELECT pg_database_size(current_database())').fetchone()[0],
                              checked_codes=verified,wall_seconds=time.perf_counter()-start,
                              existing_bundles_unchanged=True,coverage_regions=len(groups))
        atomic_json(out/'database.json',report)
        atomic_json(out/'database-load-status.json',report)
        log(f'Global database load complete: {report["datasource_id"]}, {report["layers"]} layers')
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'research/world-4km')
    parser.add_argument('--publish',action='store_true',help='Expose completed datasource through Metadata/Query')
    parser.add_argument('--validate-only',action='store_true')
    args = parser.parse_args()
    out = args.output.resolve()
    data,groups,bundle,report = validate(out)
    if not args.validate_only:
        load(out,data,groups,bundle,report,args.publish)


if __name__=='__main__':
    main()
