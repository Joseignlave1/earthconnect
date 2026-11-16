# /// script
# requires-python = ">=3.11"
# dependencies = ["rasterio==1.5.2", "numpy==2.5.4", "exactextract==0.3.0", "shapely==2.2.0", "pyproj==3.8.0", "h3==4.5.0", "pyhdf==0.11.7", "h5py==3.16.0"]
# ///
"""Scientific equivalence and tile-worker scaling on cached real thermal data."""
import thermal_import as t  # thread limits must precede NumPy imports
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np
from pyproj import CRS, Transformer
from raster_source_cache import SourceCache,atomic_json

ROOT=t.ROOT


def site_at(x,y,key):
    lon,lat=Transformer.from_crs(t.p.CRS_WKT,4326,always_xy=True).transform(x,y)
    crs=CRS.from_proj4(f'+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m')
    return dict(key=key,center=[lon,lat],crs_wkt=crs.to_wkt('WKT1_GDAL'))


def layer_codes(result):
    return {key:t.p.quantize(a,step,offset)[0] for key,a,step,offset in result['layers']}


def tile_job(job):
    site,items,cache=job
    result=t.compute_tile(site,items,cache,cell_m=4000,side_km=400)
    return dict(site=site['key'],hashes={k:t.p.sha(a.tobytes()) for k,a in layer_codes(result).items()},**result['stats'])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cache-dir',type=Path,default=Path('/tmp/earth-overlap-benchmark-cache'))
    ap.add_argument('--report',type=Path,default=ROOT/'research/import-profile/optimized.json')
    ap.add_argument('--workers',type=int,nargs='+',default=[1,4,8,16])
    ap.add_argument('--scaling-repeats',type=int,default=2)
    args=ap.parse_args()
    sample=json.loads((ROOT/'research/import-profile/sample-items.json').read_text())
    annual=json.loads((ROOT/'research/thermal-pilot/sources/atacama-catalog.json').read_text())
    annual=sorted([i for i in annual if '.h11v11.' in i['id']],key=lambda i:i['id'])
    hrefs=[item['assets'][band]['href'] for item in annual+[g['item'] for g in sample] for band in t.reference.BANDS]
    before=time.perf_counter();downloads=SourceCache(args.cache_dir).fetch_all(hrefs,workers=8)
    report=dict(status='running',dependencies={f:t.p.sha((ROOT/f).read_bytes()) for f in t.DEPENDENCIES},
                download_seconds=time.perf_counter()-before,download_bytes=sum(d['bytes'] for d in downloads if not d['cached']),
                files=len(downloads),sample_checks=[],scaling=[])
    def checkpoint(stage):
        atomic_json(args.report,report);print(json.dumps(dict(stage=stage,**{k:v for k,v in report.items() if k in ('download_bytes','download_seconds')})),flush=True)
    checkpoint('downloaded')
    for site_key in sorted(set(g['site']['key'] for g in sample)):
        groups=[g for g in sample if g['site']['key']==site_key]
        result=t.compute_tile(groups[0]['site'],[g['item'] for g in groups],str(args.cache_dir),cell_m=4000,side_km=400,verify=True)
        report['sample_checks'].append(dict(site=site_key,**result['stats']))
        checkpoint('equivalence_'+site_key)
    span=1111950.5196667;x=-20015109.354+11.5*span;y=10007554.677-11.5*span
    center=site_at(x,y,'annual-center')
    reference=t.compute_tile(center,annual,str(args.cache_dir),cell_m=4000,side_km=400,engine='reference')
    optimized=t.compute_tile(center,annual,str(args.cache_dir),cell_m=4000,side_km=400)
    codes=layer_codes(reference);fast_codes=layer_codes(optimized)
    for key in codes:np.testing.assert_array_equal(codes[key],fast_codes[key],err_msg=key)
    for (_,a,_,_),(_,b,_,_) in zip(reference['layers'],optimized['layers']):
        np.testing.assert_allclose(a,b,rtol=2e-13,atol=1e-9,equal_nan=True)
    report['annual']=dict(dates=len(annual),cells=10000,reference=reference['stats'],optimized=optimized['stats'],
                          all_nine_quantized_layers_identical=True,hashes={k:t.p.sha(a.tobytes()) for k,a in codes.items()})
    checkpoint('annual_equivalence')
    # Distinct overlapping output windows stress scheduling, not geographic coverage.
    sites=[site_at(x+dx,y+dy,f'window-{r}-{c}') for r,dy in enumerate(np.linspace(-180000,180000,4))
           for c,dx in enumerate(np.linspace(-180000,180000,4))]
    expected={}
    for repeat in range(args.scaling_repeats):
        for workers in args.workers if repeat%2==0 else args.workers[::-1]:
            start=time.perf_counter()
            with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as processes:
                results=list(processes.map(tile_job,[(site,annual,str(args.cache_dir)) for site in sites],chunksize=1))
            elapsed=time.perf_counter()-start;cpu=sum(r['cpu_seconds'] for r in results)
            for result in results:
                previous=expected.setdefault(result['site'],result['hashes'])
                if previous!=result['hashes']:raise AssertionError('Worker count changed quantized output')
            report['scaling'].append(dict(workers=workers,repeat=repeat,tiles=len(sites),composites_per_tile=len(annual),wall_seconds=elapsed,
                task_cpu_seconds=cpu,average_busy_cores=cpu/elapsed,samples=results))
            checkpoint(f'workers_{workers}_{repeat}')
    report['status']='complete'
    report['caveats']='Thermal CPU only, cached inputs. Scaling uses 16 distinct overlapping 400km windows in one native tile; not 16 geographically independent downloads. Includes geometry/weight builds and process startup. CPU utilization excludes interpreter startup outside tasks.'
    checkpoint('complete')


if __name__=='__main__':main()
