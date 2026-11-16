# /// script
# requires-python = ">=3.11"
# dependencies = ["psycopg[binary]==3.3.9", "h3==4.5.0", "rasterio==1.5.2", "numpy==2.5.4"]
# ///
"""Check every imported code and compare complete SQL profile matches to NumPy.

docker compose run --rm --entrypoint python loader scripts/test_raster_pilot.py --report /reports/priority-raster-validation.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import psycopg
from psycopg import sql
import rasterio

from load_raster_pilot import ROOT, PILOT, connection, manifest

TOLERANCE=1e-9


def predicate(conditions):
    clauses=[];params=[]
    for field,limits in conditions.items():
        for direction,value in limits.items():
            operator=sql.SQL('>=') if direction=='min' else sql.SQL('<=')
            clauses.append(sql.SQL('{} {} %s').format(sql.Identifier(field),operator))
            params.append(value-TOLERANCE if direction=='min' else value+TOLERANCE)
    return sql.SQL(' AND ').join(clauses),params


def numpy_mask(arrays,conditions):
    mask=np.ones((100,100),dtype=bool)
    for field,limits in conditions.items():
        a=arrays[field];mask &= np.isfinite(a)
        if 'min' in limits:mask &= a>=limits['min']-TOLERANCE
        if 'max' in limits:mask &= a<=limits['max']+TOLERANCE
    return mask


def run(conn,data,bundle,profiles):
    started=time.monotonic();arrays={};checked=0
    for layer in data['layers']:
        with rasterio.open(ROOT/layer['path']) as src:expected=src.read(1)
        row=conn.execute('''SELECT ST_DumpValues(t.rast,1,true), ST_BandPixelType(t.rast,1),
                ST_BandNoDataValue(t.rast,1), ST_ScaleX(t.rast), ST_ScaleY(t.rast), ST_SRID(t.rast)
                FROM data.raster_tiles t JOIN catalog.raster_layers l USING(layer_id)
                WHERE bundle_sha256=%s AND site_key=%s AND feature_key=%s''',
                (bundle,layer['site_key'],layer['feature_key'])).fetchone()
        values=np.asarray(row[0],dtype=float)
        if not np.array_equal(np.nan_to_num(values,nan=0).astype('uint16'),expected):raise AssertionError('Pixel roundtrip mismatch')
        if np.count_nonzero(np.isfinite(values))!=layer['valid_cells']:raise AssertionError('Missingness mismatch')
        if row[1:5]!=('16BUI',0.0,1000.0,-1000.0):raise AssertionError('Wrong raster metadata')
        expected_srid=next(s['srid'] for s in data['sites'] if s['key']==layer['site_key'])
        if row[5]!=expected_srid:raise AssertionError('Wrong raster CRS')
        decoded=np.where(expected==0,np.nan,expected.astype(float)*layer['scale']+layer['offset'])
        arrays.setdefault(layer['site_key'],{})[layer['feature_key']]=decoded
        checked+=expected.size
    results={};examples={}
    for name,profile in profiles.items():
        where,params=predicate(profile['conditions'])
        query=sql.SQL('''SELECT site_key,x,y FROM research.priority_pilot_cells
            WHERE bundle_sha256=%s AND {} ORDER BY site_key,y,x''').format(where)
        t=time.monotonic();rows=conn.execute(query,[bundle,*params]).fetchall();seconds=time.monotonic()-t
        expected=[]
        for site in sorted(arrays):
            mask=numpy_mask(arrays[site],profile['conditions'])
            yy,xx=np.where(mask)
            expected += [(site,int(x)+1,int(y)+1) for y,x in zip(yy,xx)]
        if rows!=expected:raise AssertionError('SQL and independent NumPy profile masks differ')
        results[name]={'label':profile['label'],'query_wall_seconds':seconds,'sites':{}}
        for site in arrays:
            selected=[(x,y) for s,x,y in rows if s==site]
            center=bool(numpy_mask(arrays[site],profile['conditions'])[50,50])
            near=sum((x-50.5)**2+(y-50.5)**2<=25 for x,y in selected)
            results[name]['sites'][site]={'matching_cells':len(selected),'eligible_fraction_of_10000_cells':len(selected)/10000,
                                        'reference_point_cell_matches':center,'matches_within_5km_of_reference':near}
            if selected:
                x,y=min(selected,key=lambda p:(p[0]-50.5)**2+(p[1]-50.5)**2)
                srid=next(s['srid'] for s in data['sites'] if s['key']==site)
                lon,lat=conn.execute('''SELECT ST_X(g),ST_Y(g) FROM
                    (SELECT ST_Transform(ST_SetSRID(ST_MakePoint(%s,%s),%s),4326) AS g) q''',
                    (-50000+(x-.5)*1000,50000-(y-.5)*1000,srid)).fetchone()
                examples[name+'--'+site]={'site_key':site,'profile':name,'x':x,'y':y,'center_lon_lat':[lon,lat],
                                         'values':{k:float(v[y-1,x-1]) if np.isfinite(v[y-1,x-1]) else None for k,v in arrays[site].items()}}
    # A no-observation permafrost tile must stay NULL after scale/offset decoding.
    if not np.isnan(arrays['atacama']['permafrost_fraction']).all():raise AssertionError('Atacama permafrost should be outside domain')
    missing=conn.execute('''SELECT count(*) FROM research.priority_pilot_cells
         WHERE bundle_sha256=%s AND site_key='atacama' AND permafrost_fraction IS NOT NULL''',(bundle,)).fetchone()[0]
    if missing:raise AssertionError('Missing permafrost became observed zero')
    # The wide SQL interface and geometry lookups must decode the same cells.
    point_checks=0
    for site in data['sites']:
        key=site['key']
        row=conn.execute('''SELECT * FROM research.priority_pilot_cells
            WHERE bundle_sha256=%s AND site_key=%s AND x=51 AND y=51''',(bundle,key)).fetchone()
        names=[d.name for d in conn.execute('SELECT * FROM research.priority_pilot_cells LIMIT 0').description]
        wide=dict(zip(names,row))
        for feature,a in arrays[key].items():
            expected=a[50,50];actual=wide[feature]
            if not ((actual is None and not np.isfinite(expected)) or
                    (actual is not None and abs(actual-expected)<TOLERANCE)):
                raise AssertionError('Wide SQL view value mismatch')
        # Original grid center is on four pixel corners: floor ownership is x=y=51.
        for x,y in [(51,51),(1,1),(100,100)]:
            px=-50000+(x-.5)*1000;py=50000-(y-.5)*1000
            if x==51 and y==51:px=py=0
            values=conn.execute('''SELECT l.feature_key,ST_Value(t.rast,1,
                ST_SetSRID(ST_MakePoint(%s,%s),l.srid),true)*l.scale+l.offset_value
                FROM data.raster_tiles t JOIN catalog.raster_layers l USING(layer_id)
                WHERE l.bundle_sha256=%s AND l.site_key=%s''',(px,py,bundle,key)).fetchall()
            for feature,actual in values:
                expected=arrays[key][feature][y-1,x-1]
                if not ((actual is None and not np.isfinite(expected)) or
                        (actual is not None and abs(actual-expected)<TOLERANCE)):
                    raise AssertionError('Geometry point lookup differs from pixel lookup')
                point_checks+=1
        codes=arrays[key]['dominant_lithology_code']
        if not set(codes[np.isfinite(codes)]).issubset({*range(1,15),16}):
            raise AssertionError('Categorical geology codes were averaged or missing class became data')
    # Forced grid/data-type errors must be rejected by database invariants.
    layer_id=conn.execute('SELECT layer_id FROM catalog.raster_layers WHERE bundle_sha256=%s LIMIT 1',(bundle,)).fetchone()[0]
    rejected=0
    for expression in [sql.SQL('ST_SetSRID(rast,4326)'),sql.SQL('ST_SetScale(rast,2000,-1000)')]:
        try:
            with conn.transaction():
                conn.execute(sql.SQL('UPDATE data.raster_tiles SET rast={} WHERE layer_id=%s').format(expression),(layer_id,))
        except psycopg.Error:rejected+=1
        else:raise AssertionError('Invalid raster update was accepted')
    # Exercise the same inherited read-only role used by the API.
    conn.execute('SAVEPOINT reader_check');conn.execute('SET LOCAL ROLE earth_index_reader')
    reader_count=conn.execute('SELECT count(*) FROM research.raster_layers WHERE bundle_sha256=%s',(bundle,)).fetchone()[0]
    if reader_count!=len(data['layers']):raise AssertionError('Reader cannot discover layers')
    conn.execute('ROLLBACK TO SAVEPOINT reader_check')
    size=conn.execute('''SELECT pg_total_relation_size('catalog.raster_layers'),pg_total_relation_size('data.raster_tiles'),
         pg_database_size(current_database())''').fetchone()
    centers={s:{k:float(v[50,50]) if np.isfinite(v[50,50]) else None for k,v in a.items()} for s,a in arrays.items()}
    return {'status':'passed','bundle_sha256':bundle,'layers_verified':len(data['layers']),'pixel_codes_verified':checked,
            'sql_profile_coordinate_sets_verified':len(profiles),'invalid_grid_updates_rejected':rejected,
            'geometry_point_values_verified':point_checks,'wide_reference_cells_verified':len(data['sites']),
            'wall_seconds':time.monotonic()-started,'profiles':results,'reference_cells':centers,'matching_examples':examples,
            'storage':{'raster_layers_relation_bytes':size[0],'raster_tiles_relation_bytes':size[1],'whole_database_bytes':size[2]},
            'limitations':[f"{len(data['sites'])} selected regions only; not a blind/global recovery or ranking benchmark.",
                           'Climate remains native 0.5x0.625 degree modeled climatology; no 1km climate detail.',
                           'Mean slope and bare fraction do not prove a fully flat, connected, traversable footprint.',
                           'Permafrost missingness is not observed absence. No crater, basalt, mineral or dune recognition claim.']}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--report',type=Path,default=PILOT/'validation.json')
    parser.add_argument('--pilot-dir',type=Path,default=PILOT)
    args=parser.parse_args();data,bundle=manifest(args.pilot_dir);priorities=json.loads((args.pilot_dir/'priorities.json').read_text())
    with connection() as conn:report=run(conn,data,bundle,priorities['profiles'])
    report['profile_definitions']=priorities['profiles']
    args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ['status','layers_verified','pixel_codes_verified','wall_seconds','profiles','storage']},indent=2))


if __name__=='__main__':main()
