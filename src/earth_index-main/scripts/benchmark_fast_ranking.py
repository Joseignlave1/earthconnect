# /// script
# requires-python = ">=3.11"
# dependencies = ["psycopg[binary]==3.3.9", "h3==4.5.0", "rasterio==1.5.2", "numpy==2.5.4"]
# ///
"""Paired benchmarks of original/optimized SQL, preserving the frozen ranking rules."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import json
from pathlib import Path
import statistics
import time

from psycopg import sql

from evaluate_ranking_pilot import OUT,arrays,diverse,index_fingerprint,order,query as original_query,run_sql,sha,verify,windows_numpy
from fast_ranking_query import query as fast_query,required_fields
from load_raster_pilot import ROOT,connection,manifest


def identities(rows):return [(r['site_key'],r['wx'],r['wy']) for r in rows]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,default=ROOT/'research/pilots/database/fast-ranking-benchmark.json')
    parser.add_argument('--artifacts-dir',type=Path,default=ROOT/'research/pilots/database/fast-ranking-artifacts')
    parser.add_argument('--repetitions',type=int,default=3,choices=range(1,6))
    args=parser.parse_args();start=time.monotonic();raw=(OUT/'spec.json').read_bytes();spec=json.loads(raw)
    old=json.loads((ROOT/'research/pilots/database/ranking-evaluation.json').read_text())
    if sha(raw)!=old['spec_sha256'] or sha((ROOT/'scripts/evaluate_ranking_pilot.py').read_bytes())!=old['processor_sha256']:
        raise ValueError('Frozen scoring specification or original implementation changed')
    data,bundle=manifest(ROOT/'research/stress-pilot')
    if bundle!=old['bundle_sha256']:raise ValueError('Source data changed')
    args.artifacts_dir.mkdir(parents=True,exist_ok=True)
    before=index_fingerprint();all_arrays=arrays(data);profiles={}
    for name,profile in spec['profiles'].items():
        weights=profile['weights'];expected,_=windows_numpy(all_arrays,spec,weights)
        with connection() as conn:
            statements={'original':original_query(spec,weights),'fast':fast_query(spec,weights,conn)}
            (args.artifacts_dir/f'{name}.sql').write_text(statements['fast'].as_string(conn)+';\n')
        timings={'original':[],'fast':[]};results={};errors=[]
        for repetition in range(args.repetitions):
            paths=['original','fast'] if repetition%2==0 else ['fast','original']
            for path in paths:
                rows,timing=run_sql(statements[path],bundle);timings[path].append(timing)
                if rows is None:raise RuntimeError(f'{name} {path} timed out; benchmark incomplete')
                errors.append(verify(rows,expected));results[path]=rows
                print(f"{name} {path}: {timing['wall_seconds']:.3f}s",flush=True)
        # Counts and bound values agree with NumPy for every window; also require
        # the entire deterministic ranking, including ties and uncertainty order.
        for field in ('score_lower','score_upper'):
            if identities(order(results['original'],field))!=identities(order(results['fast'],field)):
                raise AssertionError('Window ranking changed')
            if identities(diverse(results['original'],field))!=identities(diverse(results['fast'],field)):
                raise AssertionError('Geographic ranking changed')
        medians={k:statistics.median(t['wall_seconds'] for t in values) for k,values in timings.items()}
        profiles[name]={'label':profile['label'],'bands_read':len(required_fields(spec,weights)),
            'timings':timings,'median_seconds':medians,'speedup':medians['original']/medians['fast'],
            'windows_verified':len(expected),'max_sql_numpy_error':max(errors),'full_lower_upper_rank_order_unchanged':True}
    # A distinct four-client load test follows the isolated paired measurements.
    with connection() as conn:
        concurrent_statements={name:fast_query(spec,p['weights'],conn) for name,p in spec['profiles'].items()}
    started=time.monotonic()
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending={name:pool.submit(run_sql,statement,bundle) for name,statement in concurrent_statements.items()}
        concurrent={}
        for name,task in pending.items():
            rows,timing=task.result()
            if rows is None:raise RuntimeError('Concurrent optimized query timed out')
            expected,_=windows_numpy(all_arrays,spec,spec['profiles'][name]['weights'])
            verify(rows,expected);concurrent[name]=timing
    concurrent_seconds=time.monotonic()-started
    after=index_fingerprint()
    if before!=after:raise AssertionError('Indexes changed during benchmark')
    report={'status':'passed','generated_at':datetime.now(timezone.utc).isoformat(),'bundle_sha256':bundle,
        'spec_sha256':sha(raw),'processor_sha256':sha(Path(__file__).read_bytes()),
        'fast_builder_sha256':sha((ROOT/'scripts/fast_ranking_query.py').read_bytes()),
        'profiles':profiles,'concurrent_4_clients':concurrent,'concurrent_batch_seconds_including_verification':concurrent_seconds,
        'index_fingerprint_before':before,'index_fingerprint_after':after,'wall_seconds':time.monotonic()-start,
        'method':'Alternating original/optimized execution order, same frozen source data and scoring, fresh connections, reader role, 10-second statement timeout. Returns all 2700 windows. Warm-cache local measurements, not production latency percentiles.',
        'limits':['Optimized source assumes the existing aligned 100x100 single-tile-per-layer pilot constraints.',
                  'SQL result equivalence does not improve scientific resolution, coverage or calibration.',
                  'No persistent cache, extra index, extension, source download or quantization change.']}
    args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:{'seconds':v['median_seconds'],'speedup':v['speedup']} for k,v in profiles.items()},indent=2))


if __name__=='__main__':main()
