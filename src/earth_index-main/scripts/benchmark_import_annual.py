# /// script
# requires-python = ">=3.11"
# dependencies = ["rasterio==1.5.2", "numpy==2.5.4", "exactextract==0.3.0", "shapely==2.2.0", "pyproj==3.8.0", "h3==4.5.0", "pyhdf==0.11.7", "h5py==3.16.0"]
# ///
"""Measure a complete available thermal year for one 400km/4km region.

Reuses the frozen QC, overlap, temporal summaries, and quantization. Scientific
outputs remain in memory; only unsigned benchmark provenance/timings persist.
"""
import json
import multiprocessing
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
from pyproj import CRS, Transformer
import rasterio
from shapely.geometry import shape, box

import priority_pilot as p
import thermal_pilot as t
from benchmark_import_pipeline import HttpPool, features, digest, decode_local, save

ROOT = Path(__file__).resolve().parents[1]


def aggregate_job(job):
    group, cache = job
    start=time.perf_counter(); cpu=time.process_time()
    arrays,tr,hashes,decode_seconds,decode_cpu,_=decode_local(group,None,Path(cache),4000)
    fs,_,_=features(group['site'],4000)
    means=[];valids=[]
    for array in arrays:
        mean,valid,total=p.aggregate(array,tr,p.CRS_WKT,fs)
        means.append(mean); valids.append(valid)
    return dict(date=group['item']['properties']['start_datetime'][:10], means=means, valids=valids,
                seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu,
                decode_seconds=decode_seconds,decoded_hashes=hashes)


def main():
    items=json.loads((ROOT/'research/thermal-pilot/sources/atacama-catalog.json').read_text())
    items=sorted([i for i in items if '.h11v11.' in i['id']],key=lambda i:i['id'])
    # Center the region inside a single source tile: no omitted neighboring pieces.
    span=1111950.5196667
    x=-20015109.354+11.5*span; y=10007554.677-11.5*span
    lon,lat=Transformer.from_crs(p.CRS_WKT,4326,always_xy=True).transform(x,y)
    crs=CRS.from_proj4(f'+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m')
    site=dict(key='h11v11-center-400km',center=[lon,lat],crs_wkt=crs.to_wkt('WKT1_GDAL'))
    fs,bbox,_=features(site,4000)
    cache=Path('/tmp/earth-import-annual-cache');cache.mkdir(exist_ok=True)
    assets={};groups=[];signer=t.Signer();pool=HttpPool()
    for item in items:
        group=dict(site=site,item=item,keys={})
        for band in t.BANDS:
            href=item['assets'][band]['href'];key=digest(href.encode())[:20]
            assets[key]=dict(href=href,signed=signer.sign(href),band=band)
            group['keys'][band]=key
        groups.append(group)
    def fetch(pair):
        key,asset=pair;payload,headers,stats=pool.fetch(asset['signed']);(cache/(key+'.tif')).write_bytes(payload)
        return dict(key=key,href=asset['href'],sha256=digest(payload),**stats)
    start=time.perf_counter()
    with ThreadPoolExecutor(max_workers=8) as threads: downloads=list(threads.map(fetch,assets.items()))
    download_seconds=time.perf_counter()-start;pool.close()
    with rasterio.open(cache/(groups[0]['keys'][t.BANDS[0]]+'.tif')) as src:
        if not box(*src.bounds).contains(box(*bbox)):raise RuntimeError('Annual sample extends outside source tile')
    print(json.dumps(dict(stage='downloaded',files=len(downloads),bytes=sum(r['bytes'] for r in downloads),seconds=download_seconds)),flush=True)
    start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('spawn')) as workers:
        results=list(workers.map(aggregate_job,[(g,str(cache)) for g in groups],chunksize=1))
    processing_seconds=time.perf_counter()-start
    start=time.perf_counter(); area=np.array([shape(f['geometry']).area for f in fs]).reshape(100,100)
    dates=[r['date'] for r in results];days=[min(8,(datetime(2024,1,1)-datetime.fromisoformat(d)).days) for d in dates]
    summaries=[t.summarize(np.stack([r['means'][k] for r in results]),
                           np.clip(np.stack([r['valids'][k]/area for r in results]),0,1),dates,days) for k in range(3)]
    paired,day,night=summaries[2],summaries[0],summaries[1]
    values=[day[0],night[0],paired[0],night[1],day[2],day[2]-night[1],day[3],night[3],paired[3]]
    layers=[]
    for i,array in enumerate(values):
        step,offset=(.001,-.001) if i>=6 else ((.02,-.02) if i==5 else (.02,-150.02))
        codes,error=p.quantize(array,step,offset)
        layers.append(dict(sha256=digest(codes.tobytes()),valid_cells=int(np.count_nonzero(codes)),max_rounding_error=error))
    report=dict(site=site,cell_side_m=4000,region_side_km=400,cells=10000,
        dates=dates,files=len(downloads),download_bytes=sum(r['bytes'] for r in downloads),
        download_workers=8,download_seconds=download_seconds,downloads=downloads,
        processing_workers=8,processing_seconds=processing_seconds,
        task_cpu_seconds=sum(r['cpu_seconds'] for r in results),
        summary_quantization_seconds=time.perf_counter()-start,layers=layers,
        validation='Entire region is inside source tile; frozen QC, area overlap, annual support rules and quantization reused.',
        caveat='One available 2023 tile-year, 44 composites; no production DB writes, terrain processing, or source-boundary mosaicking.')
    save(ROOT/'research/import-profile/annual.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('site','downloads','dates','layers')}),flush=True)


if __name__=='__main__':main()
