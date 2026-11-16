# /// script
# requires-python = ">=3.11"
# dependencies = ["psycopg[binary]==3.3.9", "h3==4.5.0", "rasterio==1.5.2", "numpy==2.5.4"]
# ///
"""Evaluate frozen filters, rejection causes, spatial coherence and SQL load.

Run after load_raster_pilot.py and test_raster_pilot.py --pilot-dir research/stress-pilot.
Selection expectations are hypotheses about environments, not planetary-analog
ground truth. No thresholds are learned or modified here.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np
import psycopg
from psycopg import sql
import rasterio

from load_raster_pilot import ROOT,connection,manifest
from test_raster_pilot import numpy_mask,predicate,TOLERANCE

PILOT=ROOT/'research/stress-pilot'


def connected(mask):
    """4-neighbor connected component sizes on the existing matching pixel mask."""
    remaining=set(zip(*np.where(mask)));sizes=[]
    while remaining:
        stack=[remaining.pop()];size=0
        while stack:
            y,x=stack.pop();size+=1
            for neighbor in [(y-1,x),(y+1,x),(y,x-1),(y,x+1)]:
                if neighbor in remaining:remaining.remove(neighbor);stack.append(neighbor)
        sizes.append(size)
    return {'components':len(sizes),'largest_component_cells':max(sizes,default=0)}


def diagnostics(data,profiles):
    arrays={};output={}
    for layer in data['layers']:
        with rasterio.open(ROOT/layer['path']) as src:code=src.read(1)
        arrays.setdefault(layer['site_key'],{})[layer['feature_key']]=np.where(code==0,np.nan,code.astype(float)*layer['scale']+layer['offset'])
    for site in data['sites']:
        key=site['key'];a=arrays[key];by_profile={}
        for name,profile in profiles.items():
            conditions=profile['conditions'];mask=numpy_mask(a,conditions)
            observed=np.ones((100,100),bool);failures={}
            for field,limits in conditions.items():
                finite=np.isfinite(a[field]);observed &= finite
                passed=numpy_mask(a,{field:limits})
                failures[field]={'missing_cells':int((~finite).sum()),
                    'observed_threshold_failures':int((finite&~passed).sum())}
            by_profile[name]={'matching_cells':int(mask.sum()),'matching_fraction':float(mask.mean()),
                'complete_required_data_cells':int(observed.sum()),
                'missing_any_required_data_cells':int((~observed).sum()),
                'dominant_individual_failure':max(failures,key=lambda f:sum(failures[f].values())),
                'individual_failures_nonexclusive':failures,**connected(mask)}
        finite_medians={field:float(np.median(values[np.isfinite(values)])) if np.isfinite(values).any() else None for field,values in a.items()}
        output[key]={'name':site['name'],'role':site['role'],'center':site['center'],
            'plausible_profiles':site['plausible_profiles'],'rationale':site['rationale'],
            'patch_median_of_usable_pixel_values':finite_medians,'profiles':by_profile}
    return output


def measured_query(bundle,site_keys,profile):
    where,params=predicate(profile['conditions']);started=time.monotonic()
    try:
        with connection() as conn:
            conn.execute('SET LOCAL ROLE earth_index_reader')
            conn.execute("SET LOCAL statement_timeout='10s'")
            row=conn.execute(sql.SQL('''SELECT count(*) FROM research.priority_pilot_cells
                WHERE bundle_sha256=%s AND site_key=ANY(%s) AND {}''').format(where),
                [bundle,site_keys,*params]).fetchone()
        return {'status':'passed','matching_cells':row[0],'wall_seconds':time.monotonic()-started}
    except psycopg.errors.QueryCanceled:
        return {'status':'timeout','timeout_seconds':10,'wall_seconds':time.monotonic()-started}


def benchmarks(data,bundle,profiles,results):
    keys=[s['key'] for s in data['sites']];output=[]
    for count in (3,9,18,len(keys)):
        selected=keys[:count]
        for name,profile in profiles.items():
            samples=[]
            for _ in range(3):
                result=measured_query(bundle,selected,profile)
                if result['status']=='passed' and result['matching_cells']!=sum(results[k]['profiles'][name]['matching_cells'] for k in selected):
                    raise AssertionError('Benchmark count differs from independently evaluated arrays')
                samples.append(result)
            times=[s['wall_seconds'] for s in samples if s['status']=='passed']
            output.append({'patches':count,'profile':name,'samples':samples,
                           'median_successful_wall_seconds':statistics.median(times) if times else None})
            print(f'Benchmarked {count} patches / {name}: '+','.join(s['status'] for s in samples),flush=True)
    tasks=[(name,profile) for name,profile in profiles.items()]+[next(iter(profiles.items()))]
    start=time.monotonic()
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending=[(name,pool.submit(measured_query,bundle,keys,profile)) for name,profile in tasks]
        concurrent=[{'profile':name,**task.result()} for name,task in pending]
    for row in concurrent:
        if row['status']=='passed' and row['matching_cells']!=sum(r['profiles'][row['profile']]['matching_cells'] for r in results.values()):
            raise AssertionError('Concurrent query returned an incorrect count')
    return {'sequential':output,'concurrent_4_clients':concurrent,
            'concurrent_batch_wall_seconds':time.monotonic()-start,
            'scope':'Count queries via the existing wide raster view, fresh local connections, reader role, 10-second statement limit. Three sequential samples per size/profile; warm-cache order, not a production latency distribution.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,default=ROOT/'research/pilots/database/stress-evaluation.json')
    args=parser.parse_args();data,bundle=manifest(PILOT)
    frozen=json.loads((PILOT/'priorities.json').read_text());plan=json.loads((PILOT/'sites.json').read_text())
    original=(ROOT/frozen['baseline_file']).read_bytes()
    if hashlib.sha256(original).hexdigest()!=frozen['baseline_sha256'] or json.loads(original)['profiles']!=frozen['profiles']:
        raise AssertionError('Frozen filters changed')
    if hashlib.sha256((PILOT/'priorities.json').read_bytes()).hexdigest()!=data['frozen_priorities_sha256']:
        raise AssertionError('Filters changed after processing')
    if hashlib.sha256((PILOT/'sites.json').read_bytes()).hexdigest()!=data['site_plan_sha256']:
        raise AssertionError('Predeclared sampling plan changed after processing')
    results=diagnostics(data,frozen['profiles']);minimum=plan['patch_support_threshold_cells']
    controls=[key for key,r in results.items() if r['role']=='negative_control']
    summary={name:{'patches_with_any_match':[k for k,r in results.items() if r['profiles'][name]['matching_cells']>0],
                   'new_plausible_patches_with_support':[k for k,r in results.items() if r['role']=='plausible_match' and r['profiles'][name]['matching_cells']>=minimum],
                   'negative_control_patches_with_support':[k for k in controls if results[k]['profiles'][name]['matching_cells']>=minimum]}
             for name in frozen['profiles']}
    report={'status':'completed','generated_at':datetime.now(timezone.utc).isoformat(),'bundle_sha256':bundle,
        'frozen_profiles_sha256':data['frozen_priorities_sha256'],'patches':len(results),'new_patches':len(plan['sites']),
        'negative_control_patches':len(controls),'support_threshold_cells':minimum,'summary':summary,'sites':results,
        'benchmarks':benchmarks(data,bundle,frozen['profiles'],results),
        'limits':['Purposive small environmental sample; not global precision/recall or certified analog labels.',
                  'Pixel connectivity on 1km mean values does not establish traversability at finer scales.',
                  'Missing required measurements reject eligibility and are reported separately from observed threshold failures.',
                  'Geology and relief are imported context; frozen queries do not test their discriminating power.']}
    args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ['status','patches','new_patches','negative_control_patches','summary']},indent=2))


if __name__=='__main__':main()
