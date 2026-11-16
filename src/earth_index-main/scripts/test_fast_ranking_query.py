# /// script
# requires-python = ">=3.11"
# dependencies = ["psycopg[binary]==3.3.9", "h3==4.5.0", "rasterio==1.5.2", "numpy==2.5.4"]
# ///
"""Verify array expansion coordinates, NODATA, zero and missing-band padding."""
import json
from pathlib import Path
import unittest

import numpy as np
from psycopg import sql

from evaluate_ranking_pilot import OUT,arrays
from fast_ranking_query import cell_source,required_fields
from load_raster_pilot import ROOT,connection,manifest

SPEC=json.loads((OUT/'spec.json').read_text())
ALL_WEIGHTS={name:1/len(SPEC['components']) for name in SPEC['components']}


class ArrayRankingSource(unittest.TestCase):
    def fixture(self,feature):
        with connection() as conn:
            conn.execute('SET LOCAL ROLE earth_index_reader')
            body=cell_source(SPEC,ALL_WEIGHTS).as_string(conn)
            body=body.replace('research.raster_layers','fake_layers').replace('research.raster_tiles','fake_tiles')
            statement=sql.SQL('''WITH fake_layers AS (
                SELECT 1::bigint AS layer_id,'fixture'::text AS site_key,{}::text AS feature_key,'fixture'::text AS bundle_sha256
            ),fake_tiles AS (
                SELECT 1::bigint AS layer_id,0.001::float8 AS scale,-0.001::float8 AS offset_value,
                    ST_SetValue(ST_AddBand(ST_MakeEmptyRaster(100,100,-50000,50000,1000,-1000,0,0,910001),
                         '16BUI'::text,0,0),1,2,1,1) AS rast
            ) SELECT * FROM ({}) source ORDER BY y,x''').format(sql.Literal(feature),sql.SQL(body))
            cursor=conn.execute(statement,('fixture',));names=[d.name for d in cursor.description]
            return [dict(zip(names,row)) for row in cursor.fetchall()]

    def test_nodata_zero_and_absent_bands_remain_aligned(self):
        rows=self.fixture('bare_fraction')
        self.assertEqual(len(rows),10000)
        self.assertEqual((rows[0]['x'],rows[0]['y']),(1,1));self.assertIsNone(rows[0]['bare_fraction'])
        self.assertEqual((rows[1]['x'],rows[1]['y']),(2,1));self.assertEqual(rows[1]['bare_fraction'],0)
        self.assertIsNone(rows[1]['slope_mean_deg'])
        self.assertEqual((rows[100]['x'],rows[100]['y']),(1,2));self.assertIsNone(rows[100]['bare_fraction'])
        self.assertEqual((rows[-1]['x'],rows[-1]['y']),(100,100))
        self.assertEqual(sum(r['bare_fraction'] is not None for r in rows),1)

    def test_all_requested_bands_absent_do_not_drop_pixels(self):
        rows=self.fixture('unrequested_feature')
        self.assertEqual(len(rows),10000)
        for field in required_fields(SPEC,ALL_WEIGHTS):self.assertTrue(all(r[field] is None for r in rows))

    def test_all_real_cells_and_fields_match_verified_grids(self):
        data,bundle=manifest(ROOT/'research/stress-pilot');expected=arrays(data)
        fields=required_fields(SPEC,ALL_WEIGHTS);seen={k:np.zeros((100,100),dtype=int) for k in expected}
        checked=0
        with connection() as conn:
            conn.execute('SET LOCAL ROLE earth_index_reader');conn.execute("SET LOCAL statement_timeout='10s'")
            with conn.cursor(name='verify_array_cells') as cursor:
                cursor.execute(cell_source(SPEC,ALL_WEIGHTS),(bundle,))
                names=[d.name for d in cursor.description];indices={n:i for i,n in enumerate(names)}
                while batch:=cursor.fetchmany(2000):
                    for key in {row[0] for row in batch}:
                        rows=[r for r in batch if r[0]==key]
                        xx=np.array([r[indices['x']]-1 for r in rows]);yy=np.array([r[indices['y']]-1 for r in rows])
                        np.add.at(seen[key],(yy,xx),1)
                        for field in fields:
                            got=np.asarray([r[indices[field]] for r in rows],dtype=float)
                            want=expected[key][field][yy,xx]
                            self.assertTrue(np.allclose(got,want,rtol=0,atol=1e-9,equal_nan=True),field)
                            checked+=len(rows)
        self.assertTrue(all(np.all(counts==1) for counts in seen.values()))
        self.assertEqual(checked,270000*len(fields))
        print(f'Verified {checked} decoded values and all 270000 pixel positions')


if __name__=='__main__':unittest.main()
