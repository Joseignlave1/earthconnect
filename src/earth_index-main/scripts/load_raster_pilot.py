# /// script
# requires-python = ">=3.11"
# dependencies = ["psycopg[binary]==3.3.9", "h3==4.5.0", "rasterio==1.5.2", "numpy==2.5.4"]
# ///
"""Import the verified derived tiles through standard PostGIS raster functions.

docker compose run --build --rm --entrypoint python loader scripts/load_raster_pilot.py --report /reports/priority-raster-load.json
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from rasterio.crs import CRS

import numpy as np
import psycopg
from psycopg.types.json import Jsonb
import rasterio

from load_pilots import migrate

ROOT=Path(__file__).resolve().parents[1]
PILOT=ROOT/'research/priority-pilot'


def sha(data):return hashlib.sha256(data).hexdigest()


def connection():
    # Host uv runs can reuse private local settings; Docker supplies PG* directly.
    if not os.getenv('PGHOST') and not os.getenv('DATABASE_URL') and (ROOT/'.env').exists():
        settings=dict(line.split('=',1) for line in (ROOT/'.env').read_text().splitlines() if line and not line.startswith('#') and '=' in line)
        os.environ['DATABASE_URL']=settings['DATABASE_URL']
    return psycopg.connect(os.getenv('DATABASE_URL',''))


def manifest(pilot=PILOT):
    data=json.loads((pilot/'manifest.json').read_text())
    if sha((ROOT/data.get('processor_path','scripts/priority_pilot.py')).read_bytes())!=data['script_sha256']:
        raise ValueError('Scientific processor hash changed; reprocess before importing')
    for path,checksum in data.get('dependency_files',{}).items():
        if sha((ROOT/path).read_bytes())!=checksum:raise ValueError('Scientific dependency hash changed')
    for layer in data['layers']:
        path=ROOT/layer['path']
        if sha(path.read_bytes())!=layer['file_sha256']:raise ValueError('Tile file hash mismatch')
        with rasterio.open(path) as src:
            a=src.read(1)
            if src.count!=1 or src.dtypes!=('uint16',) or src.nodata!=0 or src.transform.b or src.transform.d or src.transform.a<=0 or src.transform.e>=0:
                raise ValueError('Invalid quantized grid')
            site=next(site for site in data['sites'] if site['key']==layer['site_key'])
            declared=layer.get('grid_definition',dict(shape=[100,100],transform=[1000,0,-50000,0,-1000,50000],crs_wkt=site['crs_wkt']))
            if list(src.shape)!=declared['shape'] or list(src.transform)[:6]!=declared['transform'] or src.crs!=CRS.from_wkt(declared['crs_wkt']):
                raise ValueError('Raster disagrees with declared grid')
            if src.crs!=CRS.from_wkt(site['crs_wkt']):raise ValueError('Region/raster CRS mismatch')
            if int((a!=0).sum())!=layer['valid_cells']:raise ValueError('Valid-cell count mismatch')
            if layer['max_rounding_error']>layer['scale']/2+1e-8:raise ValueError('Quantization error exceeded')
            if sha(a.astype('<u2').tobytes())!=layer['codes_sha256']:raise ValueError('Code hash mismatch')
    for site in data['sites']:
        grids=[l.get('grid_definition',dict(shape=[100,100],transform=[1000,0,-50000,0,-1000,50000],crs_wkt=site['crs_wkt'])) for l in data['layers'] if l['site_key']==site['key']]
        if not grids or any(g!=grids[0] for g in grids):raise ValueError('Region bands use different grids')
    identity={k:v for k,v in data.items() if k not in ['generated_at','wall_seconds']}
    return data,sha(json.dumps(identity,sort_keys=True,separators=(',',':')).encode())


def load(conn,data,bundle):
    conn.execute('SELECT pg_advisory_xact_lock(4182026, 2)')
    migrate(conn)
    conn.execute("SET LOCAL postgis.gdal_enabled_drivers = 'GTiff'")
    for key,label in [('quality.valid_area_fraction','Accepted source area fraction'),('ground.permafrost_fraction','Modeled permafrost area fraction')]:
        conn.execute('INSERT INTO catalog.variables VALUES (%s,1,%s,\'number\',%s) ON CONFLICT (variable_key) DO NOTHING',
                     (key,label,Jsonb({'id':key,'label':label,'availability':'research_priority_pilot','unit':'1'})))
    site_lookup={s['key']:s for s in data['sites']}
    for site in data['sites']:
        existing=conn.execute('SELECT srtext,proj4text FROM spatial_ref_sys WHERE srid=%s',(site['srid'],)).fetchone()
        if existing and existing!=(site['crs_wkt'],site['crs_proj4']):raise ValueError('Custom pilot CRS collision')
        conn.execute('INSERT INTO spatial_ref_sys (srid,auth_name,auth_srid,srtext,proj4text) VALUES (%s,\'earth_index\',%s,%s,%s) ON CONFLICT (srid) DO NOTHING',
                     (site['srid'],site['srid'],site['crs_wkt'],site['crs_proj4']))
    reused=0
    for layer in data['layers']:
        site=site_lookup[layer['site_key']]
        previous=conn.execute('SELECT layer_id,file_sha256 FROM catalog.raster_layers WHERE bundle_sha256=%s AND site_key=%s AND feature_key=%s',
                              (bundle,layer['site_key'],layer['feature_key'])).fetchone()
        if previous:
            if previous[1]!=layer['file_sha256']:raise ValueError('Layer identity conflict')
            reused+=1;continue
        grid=layer.get('grid_definition',{'crs_wkt':site['crs_wkt'],'shape':[100,100],'transform':[1000,0,-50000,0,-1000,50000]}).copy()
        grid.update(projected_pixel_area_m2=abs(grid['transform'][0]*grid['transform'][4]),
              area_note='Local AEQD metre squares; geodesic areas differ slightly.',site_center_lon_lat=site['center'],nodata=0)
        layer_id=conn.execute('''INSERT INTO catalog.raster_layers
            (bundle_sha256,site_key,feature_key,variable_key,unit,scale,offset_value,srid,grid_definition,
             provenance,valid_cells,max_rounding_error,file_sha256)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING layer_id''',
            (bundle,layer['site_key'],layer['feature_key'],layer['variable_key'],layer['unit'],layer['scale'],layer['offset'],
             site['srid'],Jsonb(grid),Jsonb(layer['source']),layer['valid_cells'],layer['max_rounding_error'],layer['file_sha256'])).fetchone()[0]
        conn.execute('INSERT INTO data.raster_tiles VALUES (%s,0,0,ST_FromGDALRaster(%s,%s))',
                     (layer_id,(ROOT/layer['path']).read_bytes(),site['srid']))
        row=conn.execute('SELECT ST_DumpValues(rast,1,true),ST_SRID(rast) FROM data.raster_tiles WHERE layer_id=%s',(layer_id,)).fetchone()
        roundtrip=np.nan_to_num(np.asarray(row[0],dtype=float),nan=0).astype('<u2')
        if sha(roundtrip.tobytes())!=layer['codes_sha256'] or row[1]!=site['srid']:raise ValueError('PostGIS raster roundtrip failed')
    counts=conn.execute('''SELECT count(*),sum(l.valid_cells),sum(pg_column_size(t.rast))
        FROM catalog.raster_layers l JOIN data.raster_tiles t USING(layer_id) WHERE l.bundle_sha256=%s''',(bundle,)).fetchone()
    if counts[0]!=len(data['layers']):raise ValueError('Incomplete raster bundle')
    conn.execute('ANALYZE catalog.raster_layers');conn.execute('ANALYZE data.raster_tiles')
    return {'bundle_sha256':bundle,'layers':counts[0],'valid_values':counts[1],'raster_datum_bytes':counts[2],
            'reused_layers':reused,'schema':'standard PostGIS 16BUI rasters; one layer/tile per feature and area'}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--report',type=Path,default=PILOT/'database.json')
    parser.add_argument('--pilot-dir',type=Path,default=PILOT)
    args=parser.parse_args();start=time.monotonic();data,bundle=manifest(args.pilot_dir)
    with connection() as conn:report=load(conn,data,bundle)
    report['wall_seconds']=time.monotonic()-start
    args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
