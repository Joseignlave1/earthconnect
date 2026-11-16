# /// script
# requires-python = ">=3.11"
# dependencies = ["psycopg[binary]==3.3.9", "h3==4.5.0", "rasterio==1.5.2", "numpy==2.5.4"]
# ///
"""Load a verified expansion and prepare SQL views without changing demo defaults.

The new schema is queryable through the existing SQL driver using qualified names.
Publication to Metadata is an explicit later step after validation.
"""
import argparse
import json
from pathlib import Path
import time

from load_raster_pilot import ROOT,connection,manifest,load


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--pilot-dir',type=Path,default=ROOT/'research/expansion-4km');ap.add_argument('--publish',action='store_true')
    args=ap.parse_args();start=time.perf_counter();data,bundle=manifest(args.pilot_dir)
    with connection() as conn:
        original_ids=[row[0] for row in conn.execute('SELECT datasource_id FROM catalog.api_datasources ORDER BY datasource_id')]
        report=load(conn,data,bundle)
        conn.execute('INSERT INTO catalog.raster_bundle_publication VALUES (%s,%s) ON CONFLICT (bundle_sha256) DO UPDATE SET published=EXCLUDED.published',(bundle,args.publish))
        conn.execute('SELECT catalog.refresh_api_datasources()')
        if not args.publish:
            conn.execute('DELETE FROM catalog.api_datasources WHERE bundle_sha256=%s',(bundle,))
            assert [r[0] for r in conn.execute('SELECT datasource_id FROM catalog.api_datasources ORDER BY datasource_id')]==original_ids
        else:
            conn.execute('UPDATE catalog.api_datasources SET name=%s,description=%s WHERE bundle_sha256=%s',
                (f'Earth 4 km expansion — {len(data["sites"])} patches',f'Timed validation dataset: {len(data["sites"])} 400 x 400 km footprints, 4 km cells. Variable support differs; inspect provenance.',bundle))
    report.update(wall_seconds=time.perf_counter()-start,schema_name='raster_'+bundle[:32],datasource_id='earth-'+bundle[:12],published=args.publish,
        access_note='Use fully qualified schema.cells through the existing SQL driver while staged; Metadata/demo datasource selection unchanged.')
    (args.pilot_dir/'database.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
