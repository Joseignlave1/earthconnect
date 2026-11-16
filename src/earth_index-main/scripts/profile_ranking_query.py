# /// script
# requires-python = ">=3.11"
# dependencies = ["psycopg[binary]==3.3.9", "h3==4.5.0", "rasterio==1.5.2", "numpy==2.5.4"]
# ///
"""Capture a query plan for the original ranking SQL without changing it."""
import argparse
import json
from pathlib import Path

from psycopg import sql

from evaluate_ranking_pilot import OUT,query
from load_raster_pilot import connection


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--fast',action='store_true')
    args=parser.parse_args();spec=json.loads((OUT/'spec.json').read_text())
    statement=query(spec,spec['profiles']['hot_balanced']['weights'])
    with connection() as conn:
        if args.fast:
            from fast_ranking_query import query as fast_query
            statement=fast_query(spec,spec['profiles']['hot_balanced']['weights'],conn)
        conn.execute('SET LOCAL ROLE earth_index_reader')
        conn.execute("SET LOCAL statement_timeout='30s'")
        result=conn.execute(sql.SQL('EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON, SETTINGS) {}').format(statement),
                            (spec['source_bundle_sha256'],)).fetchone()[0]
    args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(result,indent=2)+'\n')
    def visit(node,depth=0):
        print('  '*depth+node['Node Type']+f" rows={node.get('Actual Rows')} loops={node.get('Actual Loops')} time_ms={node.get('Actual Total Time')}")
        for child in node.get('Plans',[]):visit(child,depth+1)
    print('Execution ms',result[0]['Execution Time']);visit(result[0]['Plan'])


if __name__=='__main__':main()
