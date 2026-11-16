# /// script
# requires-python = ">=3.11"
# dependencies = ["rasterio==1.5.2", "numpy==2.5.4", "exactextract==0.3.0", "shapely==2.2.0", "pyproj==3.8.0", "h3==4.5.0", "pyhdf==0.11.7", "h5py==3.16.0"]
# ///
"""Check 4 km base sources before the full thermal transfer completes."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import json
import sys
import re
import time
import numpy as np
import rasterio
import expansion_features as base
import import_expansion as runner
import thermal_import as t
import priority_pilot as p

out=p.ROOT/'research/expansion-4km';cache=out/'raw-cache'
plan=json.loads((out/'plan.json').read_text());old=json.loads((p.ROOT/'research/thermal-pilot/manifest.json').read_text())
catalog={re.search(r'h\d{2}v\d{2}',g['title'])[0]:g for g in json.loads((out/'landcover-catalog.json').read_text())}
path=out/'source-precheck.json'
report=json.loads(path.read_text()) if path.exists() else {}
for key in (sys.argv[1:] or ['atacama','haughton','antarctica']):
    site=next((s for s in plan['sites'] if s['key']==key),None)
    if site is None:continue
    fs,_=t.grid_features(site,cell_m=4000,side_km=400,target_crs=base.GEO)
    projected,_=t.grid_features(site,cell_m=4000,side_km=400)
    values={};durations={}
    for name,fn in [('terrain',lambda:base.terrain(site,cache)),('cover',lambda:base.cover(site,projected,cache,catalog)),
                    ('climate',lambda:base.climate(fs,[out/'sources'/name for _,name in runner.climate_requests(site)])),
                    ('permafrost',lambda:base.permafrost(fs,cache)),('lithology',lambda:base.lithology(site,cache))]:
        start=time.perf_counter();v,_=fn();values.update(v);durations[name]=time.perf_counter()-start
    differences={}
    for l in old['layers']:
        if l['site_key']!=key or l['feature_key'] not in values or l['feature_key']=='dominant_lithology_code':continue
        with rasterio.open(p.ROOT/l['path']) as src:a=src.read(1)
        a=np.where(a==0,np.nan,a.astype(float)*l['scale']+l['offset'])
        expected=base.reduce4(a[2:98,2:98]);actual=values[l['feature_key']][38:62,38:62]
        mask=np.isfinite(expected)&np.isfinite(actual);d=np.abs(expected[mask]-actual[mask])
        differences[l['feature_key']]=dict(shared=int(mask.sum()),p95=float(np.quantile(d,.95)) if d.size else None,max=float(d.max()) if d.size else None)
    report[key]=dict(stage_seconds=durations,valid={k:int(np.isfinite(a).sum()) for k,a in values.items()},shared_differences=differences)
    print(json.dumps({key:report[key]}),flush=True)
(out/'source-precheck.json').write_text(json.dumps(report,indent=2)+'\n')
