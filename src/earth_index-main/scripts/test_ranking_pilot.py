# /// script
# requires-python = ">=3.11"
# dependencies = ["psycopg[binary]==3.3.9", "h3==4.5.0", "rasterio==1.5.2", "numpy==2.5.4"]
# ///
"""Properties that matter for fuzzy ranking: co-location and honest missingness."""
import json
import math
import unittest

import numpy as np
from psycopg import sql

from evaluate_ranking_pilot import OUT,component_sql,gate_numpy,query,scores_numpy
from load_raster_pilot import connection

SPEC=json.loads((OUT/'spec.json').read_text())


class RankingProperties(unittest.TestCase):
    def test_colocated_preferences_beat_disjoint_extrema(self):
        low,_,_=scores_numpy({'heat':np.array([.9,.9]),'flat':np.array([.9,.9])},{'heat':.5,'flat':.5})
        disjoint,_,_=scores_numpy({'heat':np.array([.99,.01]),'flat':np.array([.01,.99])},{'heat':.5,'flat':.5})
        self.assertAlmostEqual(float(low.mean()),.9)
        self.assertAlmostEqual(float(disjoint.mean()),math.sqrt(.0099))
        self.assertGreater(float(low.mean()),float(disjoint.mean()))

    def test_missing_dimension_bounds_do_not_reward_missingness(self):
        low,high,evidence=scores_numpy({'heat':np.array([.9]),'flat':np.array([np.nan])},{'heat':.5,'flat':.5})
        self.assertEqual(low.item(),0)
        self.assertAlmostEqual(high.item(),math.sqrt(.9))
        self.assertEqual(evidence.item(),.5)
        for actual in [0,.1,.5,1]:
            score,_,_=scores_numpy({'heat':np.array([.9]),'flat':np.array([actual])},{'heat':.5,'flat':.5})
            self.assertLessEqual(low.item(),score.item());self.assertGreaterEqual(high.item(),score.item())

    def test_unknown_gate_is_distinct_from_observed_contradiction(self):
        a={'landcover_valid_fraction':np.array([1,1,0,1]),'terrain_valid_fraction':np.array([1,0,1,1]),
           'bare_fraction':np.array([1,1,np.nan,0]),'slope_mean_deg':np.array([1,np.nan,1,np.nan])}
        self.assertEqual(gate_numpy(a,SPEC).tolist(),[1,2,2,0])

    def test_sql_fraction_preserves_null_and_physical_zero(self):
        expression=component_sql(SPEC['components']['frozen'])
        with connection() as conn:
            rows=conn.execute(sql.SQL('''SELECT {} FROM (VALUES
                (NULL::float8,1.0),(0.0,1.0),(1.0,1.0),(0.9,0.0))
                f(permafrost_fraction,permafrost_valid_fraction)''').format(expression)).fetchall()
        self.assertEqual([r[0] for r in rows],[None,0,1,None])

    def test_sql_windows_preserve_colocation_and_unknown_eligibility(self):
        source='''SELECT site_key,x,y,1.0 AS landcover_valid_fraction,
            CASE WHEN site_key='unknown' THEN 0.0 ELSE 1.0 END AS terrain_valid_fraction,
            1.0 AS bare_fraction,
            CASE WHEN site_key='unknown' THEN NULL::float8
                 WHEN site_key='coincident' THEN -8.0*ln(0.9)
                 WHEN y<=5 THEN -8.0*ln(0.01) ELSE -8.0*ln(0.99) END AS slope_mean_deg,
            CASE WHEN site_key<>'separated' THEN 25.0+5.0*ln(9.0)
                 WHEN y<=5 THEN 25.0+5.0*ln(99.0) ELSE 25.0+5.0*ln(1.0/99.0) END AS warmest_month_air_c
            FROM (VALUES ('coincident'),('separated'),('unknown')) s(site_key)
            CROSS JOIN generate_series(1,10) x CROSS JOIN generate_series(1,10) y'''
        with connection() as conn:
            text=query(SPEC,{'heat':.5,'flat':.5}).as_string(conn)
            text=text.replace('SELECT * FROM research.priority_pilot_cells WHERE bundle_sha256=%s',source)
            cursor=conn.execute(text);names=[c.name for c in cursor.description]
            rows={r['site_key']:r for r in [dict(zip(names,row)) for row in cursor.fetchall()]}
        self.assertAlmostEqual(rows['coincident']['score_lower'],.9)
        self.assertAlmostEqual(rows['separated']['score_lower'],math.sqrt(.0099)/2)
        self.assertEqual(rows['unknown']['score_lower'],0)
        self.assertEqual(rows['unknown']['unknown_cells'],100)
        self.assertAlmostEqual(rows['unknown']['score_upper'],math.sqrt(.9))


if __name__=='__main__':unittest.main()
