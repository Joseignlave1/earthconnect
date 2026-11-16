# /// script
# requires-python = ">=3.11"
# dependencies = ["duckdb==2.0.0"]
# ///
"""Build the immutable serving file, atomically, from the validated DOUBLE import.

uv run db/prepare_duckdb.py
The frozen source measurements and source-metadata.{pb,json} are never modified.
"""
import argparse
import base64
import json
import os
from pathlib import Path
import time
import duckdb

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=ROOT/'research/world-4km/duckdb-double/world-double.duckdb')
    parser.add_argument('--output', type=Path, default=ROOT/'db/runtime/earth.duckdb')
    parser.add_argument('--metadata', type=Path, default=ROOT/'db/runtime/source-metadata.json')
    parser.add_argument('--spatial', type=Path, default=ROOT/'db/runtime/spatial.duckdb_extension')
    args = parser.parse_args()
    started = time.perf_counter()
    metadata = json.loads(args.metadata.read_text())
    datasource, = metadata['datasources']
    assert datasource['revision'] == json.loads(duckdb.connect(str(args.source),read_only=True).sql('SELECT identity FROM layout_identity').fetchall()[0][0])['bundle']
    grids = {g['siteKey']:g for g in datasource['grids']}
    coverage = {c['siteKey']:c for c in datasource['coverage']}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    staging = args.output.with_suffix('.building.duckdb')
    for stale in [staging, Path(str(staging)+'.wal')]:
        if stale.exists(): stale.unlink()
    conn = duckdb.connect(str(staging))
    conn.execute('SET threads=8')
    conn.execute("SET memory_limit='8GB'")
    conn.execute("LOAD '"+str(args.spatial.resolve()).replace("'","''")+"'")
    conn.execute("ATTACH '"+str(args.source.resolve()).replace("'","''")+"' AS source_db (READ_ONLY)")
    # DOUBLE values are already in physical units; only category codes are integral.
    assert conn.sql('SELECT count(*) FROM source_db.cells WHERE dominant_lithology_code != trunc(dominant_lithology_code)').fetchall()[0][0] == 0
    conn.execute('CREATE TABLE measurements AS SELECT * REPLACE(dominant_lithology_code::SMALLINT AS dominant_lithology_code) FROM source_db.cells')
    conn.execute('CREATE TABLE regions(region_id USMALLINT,site_key VARCHAR,width UINTEGER,height UINTEGER,srid INTEGER,crs_wkt VARCHAR,a DOUBLE,b DOUBLE,c DOUBLE,d DOUBLE,e DOUBLE,f DOUBLE,coverage GEOMETRY)')
    regions = conn.sql('SELECT * FROM source_db.regions ORDER BY region_id').fetchall()
    for region_id, site_key in regions:
        g = grids[site_key]
        conn.execute('INSERT INTO regions SELECT ?,?,?,?,?,?,?,?,?,?,?,?,ST_GeomFromWKB(?)', [region_id,site_key,g['width'],g['height'],g['srid'],g['crsWkt'],*g['transform'],base64.b64decode(coverage[site_key]['wkb'])])
    conn.execute('CREATE TABLE api_metadata(payload BLOB)')
    conn.execute('INSERT INTO api_metadata VALUES (?)',[args.metadata.with_suffix('.pb').read_bytes()])
    conn.execute('CREATE TABLE layout_identity AS SELECT * FROM source_db.layout_identity')
    # Match the previous safe sinusoidal footprint algorithm, including curved
    # domain clipping at the antimeridian/poles. 4 km cells span <.05 deg latitude;
    # two segments on each edge meet the original maximum sampling interval.
    conn.execute("CREATE MACRO sinu_distance(_xmin,_xmax) AS CASE WHEN _xmin<=0 AND _xmax>=0 THEN 0 ELSE least(abs(_xmin),abs(_xmax)) END")
    conn.execute("CREATE MACRO sinu_limit(_xmin,_xmax) AS degrees(acos(least(1.,sinu_distance(_xmin,_xmax)/(pi()*6371007.181))))")
    conn.execute("CREATE MACRO sinu_lo(_xmin,_ymin,_xmax) AS greatest(-90.,degrees(_ymin/6371007.181),-sinu_limit(_xmin,_xmax))")
    conn.execute("CREATE MACRO sinu_hi(_xmin,_xmax,_ymax) AS least(90.,degrees(_ymax/6371007.181),sinu_limit(_xmin,_xmax))")
    conn.execute("CREATE MACRO sinu_point(_x,_lat) AS ST_Point(greatest(-180.,least(180.,degrees(_x/(6371007.181*greatest(cos(radians(_lat)),1e-15))))),_lat)")
    lo = 'sinu_lo(_xmin,_ymin,_xmax)'
    hi = 'sinu_hi(_xmin,_xmax,_ymax)'
    mid = f'(({lo}+{hi})/2)'
    vertices = [f'sinu_point({x},{lat})' for x,lat in [('_xmin',lo),('_xmin',mid),('_xmin',hi),('_xmax',hi),('_xmax',mid),('_xmax',lo),('_xmin',lo)]]
    conn.execute(f"CREATE MACRO sinusoidal_footprint(_xmin,_ymin,_xmax,_ymax) AS CASE WHEN {lo}<{hi} THEN ST_MakePolygon(ST_MakeLine([{','.join(vertices)}])) ELSE NULL END")
    conn.execute('''CREATE VIEW cells AS
      SELECT m.*,r.site_key, (pixel % r.width + 1)::UINTEGER x,(pixel // r.width + 1)::UINTEGER y,
        CASE WHEN abs(r.a*((pixel%r.width)+.5)+r.c) > pi()*6371007.181*cos((r.e*((pixel//r.width)+.5)+r.f)/6371007.181)
          THEN NULL ELSE sinusoidal_footprint(r.a*(pixel%r.width)+r.c,r.e*((pixel//r.width)+1)+r.f,
                 r.a*((pixel%r.width)+1)+r.c,r.e*(pixel//r.width)+r.f) END geom
      FROM measurements m JOIN regions r USING(region_id)''')
    # Verify every field and NULL after changing the storage layout.
    names = [r[0] for r in conn.sql('DESCRIBE measurements').fetchall()]
    checksum = ','.join('"'+name+'"::DOUBLE' for name in names)
    source = conn.sql(f'SELECT count(*),bit_xor(hash({checksum})) FROM source_db.cells').fetchall()[0]
    actual = conn.sql(f'SELECT count(*),bit_xor(hash({checksum})) FROM measurements').fetchall()[0]
    assert actual == source and actual[0] == 35_550_640
    assert len(regions) == len(grids) == len(coverage) == 460
    conn.execute('DETACH source_db')
    conn.execute('CHECKPOINT')
    conn.close()
    os.replace(staging,args.output)
    print(json.dumps(dict(path=str(args.output),rows=actual[0],regions=len(regions),bytes=args.output.stat().st_size,seconds=time.perf_counter()-started,all_columns_verified=True),indent=2))


if __name__ == '__main__': main()
