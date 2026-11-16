# /// script
# requires-python = ">=3.11"
# dependencies = ["psycopg[binary]==3.3.9", "h3==4.5.0", "rasterio==1.5.2", "numpy==2.5.4"]
# ///
"""Verify every current-grid thermal bundle code and SQL physical summary."""
import json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from psycopg import sql
import rasterio
from load_raster_pilot import ROOT,connection,manifest


def main():
    pilot=ROOT/'research/thermal-pilot';data,bundle=manifest(pilot)
    expected={};checked=0
    with connection() as conn:
        ns=conn.execute('SELECT schema_name::text FROM catalog.api_datasources WHERE bundle_sha256=%s',(bundle,)).fetchone()[0]
        for layer in data['layers']:
            with rasterio.open(ROOT/layer['path']) as src:code=src.read(1)
            row=conn.execute('''SELECT ST_DumpValues(t.rast,1,false),l.scale,l.offset_value
              FROM data.raster_tiles t JOIN catalog.raster_layers l USING(layer_id)
              WHERE bundle_sha256=%s AND site_key=%s AND feature_key=%s''',(bundle,layer['site_key'],layer['feature_key'])).fetchone()
            if not row or not np.array_equal(np.asarray(row[0],dtype='uint16'),code):raise AssertionError('Imported raster code mismatch')
            if (row[1],row[2])!=(layer['scale'],layer['offset']):raise AssertionError('Physical decode mismatch')
            checked+=code.size
            if layer['feature_key'].startswith('surface_'):
                values=code[code>0].astype(float)*layer['scale']+layer['offset']
                expected.setdefault(layer['feature_key'],[]).append(values)
        summaries={}
        for key,chunks in expected.items():
            values=np.concatenate(chunks)
            row=conn.execute(sql.SQL('SELECT count({0}),min({0}),max({0}),avg({0}) FROM {1}.cells').format(sql.Identifier(key),sql.Identifier(ns))).fetchone()
            if row[0]!=len(values):raise AssertionError('SQL missingness mismatch')
            if len(values) and not np.allclose(row[1:],[values.min(),values.max(),values.mean()],rtol=0,atol=1e-8):raise AssertionError('SQL statistics differ from independent arrays')
            summaries[key]={'valid_cells':row[0],'min':row[1],'max':row[2],'mean':row[3]}
        rows=conn.execute(sql.SQL('SELECT count(*),count(DISTINCT site_key) FROM {}.cells').format(sql.Identifier(ns))).fetchone()
        if rows!=(270000,27):raise AssertionError('Thermal revision changed the pilot grid')
    report={'status':'passed','generated_at':datetime.now(timezone.utc).isoformat(),'bundle_sha256':bundle,
            'datasource_id':'earth-'+bundle[:12],'layers':len(data['layers']),'codes_verified':checked,
            'rows':rows[0],'patches':rows[1],'surface_summaries':summaries}
    (pilot/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
