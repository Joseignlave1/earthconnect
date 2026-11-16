# /// script
# requires-python = ">=3.11"
# dependencies = ["psycopg[binary]==3.3.9", "h3==4.5.0", "rasterio==1.5.2", "numpy==2.5.4"]
# ///
"""Run one optimized pilot ranking query and return supported/uncertain windows.

docker compose run --rm --entrypoint python loader scripts/rank_pilot.py --profile heat_priority --geographic --limit 10
"""
import argparse
import json

from evaluate_ranking_pilot import OUT,TOL,diverse,order,run_sql,sha
from fast_ranking_query import query
from load_raster_pilot import connection


def main():
    raw=(OUT/'spec.json').read_bytes();spec=json.loads(raw)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',choices=spec['profiles'],default='hot_balanced')
    parser.add_argument('--mode',choices=['supported','uncertain'],default='supported')
    parser.add_argument('--geographic',action='store_true',help='Return one best window per sampled patch')
    parser.add_argument('--limit',type=int,default=20)
    args=parser.parse_args()
    if args.limit<1 or args.limit>2700:parser.error('limit must be 1..2700')
    with connection() as conn:statement=query(spec,spec['profiles'][args.profile]['weights'],conn)
    rows,timing=run_sql(statement,spec['source_bundle_sha256'])
    if rows is None:raise RuntimeError('Ranking query exceeded the 10-second statement limit')
    if args.mode=='supported':
        rows=[r for r in rows if r['score_lower']>TOL];field='score_lower'
    else:
        rows=[r for r in rows if r['score_upper']>r['score_lower']+TOL];field='score_upper'
    eligible=len(rows);ranked=diverse(rows,field) if args.geographic else order(rows,field)
    print(json.dumps({'profile':args.profile,'mode':args.mode,'bundle_sha256':spec['source_bundle_sha256'],
        'spec_sha256':sha(raw),'geographic_grouping':args.geographic,'query_seconds':timing['wall_seconds'],
        'eligible_windows':eligible,'ranked_candidates':len(ranked),'returned':min(args.limit,len(ranked)),
        'window_width_km':spec['window_pixels'],'results':ranked[:args.limit]},indent=2))


if __name__=='__main__':main()
