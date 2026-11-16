# /// script
# requires-python = ">=3.11"
# dependencies = ["rasterio==1.5.2", "numpy==2.5.4", "exactextract==0.3.0", "shapely==2.2.0", "pyproj==3.8.0", "h3==4.5.0", "pyhdf==0.11.7", "h5py==3.16.0"]
# ///
"""Build a new pilot revision with bounded public MODIS COG window reads.

2023 clear-sky composite summaries, not all-weather climatology or true extremes.
Original rasters and ephemeral public access signatures are never retained.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
import json
import logging
import multiprocessing
from pathlib import Path
import re
import time
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

import numpy as np
import rasterio
from rasterio.windows import from_bounds, Window
from shapely.geometry import shape
import priority_pilot as p

ROOT = p.ROOT
OUT = ROOT / 'research/thermal-pilot'
BASE = ROOT / 'research/stress-pilot/manifest.json'
STAC = 'https://planetarycomputer.microsoft.com/api/stac/v1'
YEAR = 2023
BANDS = ('LST_Day_1km', 'QC_Day', 'LST_Night_1km', 'QC_Night')
METHOD = 'mod11a2-terra-cog-2023-qc0-area-overlap-v1'


def get_json(url):
    for attempt in range(3):
        try:
            with urlopen(Request(url, headers={'User-Agent': 'EarthIndex/0.1 thermal-pilot'}), timeout=30) as response:
                return json.load(response)
        except Exception:
            if attempt == 2:
                raise RuntimeError('Public MODIS metadata request failed; access URLs omitted') from None
            time.sleep(attempt+1)


def inventory(site):
    path = OUT/'sources'/f"{site['key']}-catalog.json"
    if path.exists():
        return json.loads(path.read_text())
    bbox = p.bounds_of(p.cells(site, 'EPSG:4326'))
    query = {'collections': 'modis-11A2-061', 'bbox': ','.join(map(str,bbox)),
             'datetime': f'{YEAR}-01-01T00:00:00Z/{YEAR}-12-31T23:59:59Z',
             'query': json.dumps({'platform': {'eq': 'terra'}}), 'limit': 1000}
    url = STAC+'/search?'+urlencode(query)
    items = []
    while url:
        result = get_json(url)
        items.extend(result['features'])
        url = next((x['href'] for x in result.get('links',[]) if x['rel']=='next'), None)
        if url and not url.startswith(STAC+'/'):
            raise ValueError('Unexpected catalog pagination host')
    # Catalog tile footprints are conservative: actual native windows are checked.
    for item in items:
        if not re.fullmatch(r'MOD11A2\.A2023\d{3}\.h\d{2}v\d{2}\.061\.\d+',item['id']):
            raise ValueError('Unexpected platform/year/product in thermal inventory')
    p.save(path, items)
    return items


class Signer:
    def __init__(self): self.tokens = {}
    def sign(self, href):
        u = urlparse(href)
        if u.scheme != 'https' or not u.hostname.endswith('.blob.core.windows.net') or u.query:
            raise ValueError('Unexpected unsigned MODIS asset URL')
        account, container = u.hostname.split('.')[0], u.path.split('/')[1]
        key = account, container
        if key not in self.tokens:
            self.tokens[key] = get_json(f'https://planetarycomputer.microsoft.com/api/sas/v1/token/{account}/{container}')['token']
        return href+'?'+self.tokens[key]


def decode_window(item, bbox, signer):
    arrays = {}; transform = None; metadata = []
    for band in BANDS:
        asset = item['assets'][band]
        try:
            with rasterio.open(signer.sign(asset['href'])) as src:
                w = from_bounds(*bbox, src.transform)
                x0=max(0,int(np.floor(w.col_off))-1);y0=max(0,int(np.floor(w.row_off))-1)
                x1=min(src.width,int(np.ceil(w.col_off+w.width))+1);y1=min(src.height,int(np.ceil(w.row_off+w.height))+1)
                if x1<=x0 or y1<=y0: return None
                window=Window(x0,y0,x1-x0,y1-y0)
                tr=src.window_transform(window)
                if transform is not None and tr != transform: raise ValueError('MODIS band grids differ')
                transform=tr
                expected='uint8' if band.startswith('QC') else 'uint16'
                if src.dtypes != (expected,): raise ValueError('Unexpected native MODIS band type')
                if band.startswith('LST') and (src.scales!=(.02,) or src.offsets!=(0.,)):
                    raise ValueError('Unexpected MODIS physical scaling')
                if src.crs != rasterio.crs.CRS.from_wkt(p.CRS_WKT):
                    # GDAL normalizes equivalent sinusoidal WKT differently.
                    from pyproj import CRS
                    if not CRS(src.crs).equals(CRS(p.CRS_WKT),ignore_axis_order=True):
                        raise ValueError('Unexpected MODIS projection')
                # QC zero is GOOD, even though its GeoTIFF nodata tag is also zero.
                raw=src.read(1,window=window,masked=False)
                arrays[band]=raw
                metadata.append({'band':band,'href':asset['href'],'window':[x0,y0,x1-x0,y1-y0],
                                 'decoded_sha256':p.sha(raw.tobytes())})
        except Exception:
            raise RuntimeError(f"MODIS window read failed: {item['id']} {band}; access signature omitted") from None
    day=np.where((arrays['LST_Day_1km']>=7500)&(arrays['QC_Day']==0),arrays['LST_Day_1km'].astype(float)*.02-273.15,np.nan)
    night=np.where((arrays['LST_Night_1km']>=7500)&(arrays['QC_Night']==0),arrays['LST_Night_1km'].astype(float)*.02-273.15,np.nan)
    return (day,night,(day+night)/2),transform,metadata


def summarize(stack, fractions, dates, days):
    """Only well-supported composite cells contribute; all seasons must occur."""
    usable=np.isfinite(stack)&(fractions>=.5)
    counts=usable.sum(axis=0)
    quarter=np.array([(datetime.fromisoformat(d).month-1)//3 for d in dates])
    seasons=np.stack([np.any(usable[quarter==q],axis=0) if np.any(quarter==q) else np.zeros(stack.shape[1:],bool) for q in range(4)])
    enough=(counts>=12)&seasons.all(axis=0)
    weights=np.where(usable,fractions*np.array(days)[:,None,None],0)
    mean=np.divide(np.nansum(stack*weights,axis=0),weights.sum(axis=0),out=np.full(stack.shape[1:],np.nan),where=weights.sum(axis=0)>0)
    low=np.min(np.where(usable,stack,np.inf),axis=0)
    high=np.max(np.where(usable,stack,-np.inf),axis=0)
    coverage=np.minimum(1,np.sum(np.clip(fractions,0,1)*np.array(days)[:,None,None],axis=0)/365)
    return [np.where(enough,v,np.nan) for v in (mean,low,high)]+[coverage]


def worker(site):
    p.OUT=OUT
    checkpoint=OUT/'checkpoints'/f"{site['key']}.json"
    identity=p.sha(Path(__file__).read_bytes())
    if checkpoint.exists():
        previous=json.loads(checkpoint.read_text())
        if previous['processor_sha256']==identity and all(p.sha((ROOT/l['path']).read_bytes())==l['file_sha256'] for l in previous['layers']):
            return previous
    start=time.monotonic();signer=Signer();items=inventory(site)
    fs=p.cells(site,p.CRS_WKT);bbox=p.bounds_of(fs)
    area=np.array([shape(f['geometry']).area for f in fs]).reshape(100,100)
    groups={}
    for item in items:
        date=item['properties']['start_datetime'][:10]
        groups.setdefault(date,[]).append(item)
    means=[[],[],[]];fractions=[[],[],[]];dates=[];days=[];sources=[]
    logging.getLogger('rasterio._env').setLevel(logging.CRITICAL)
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif',GDAL_HTTP_TIMEOUT='30',GDAL_HTTP_MAX_RETRY='2',GDAL_HTTP_RETRY_DELAY='1',GDAL_CACHEMAX=48*1024*1024):
        for date,group in sorted(groups.items()):
            sums=[np.zeros((100,100)) for _ in range(3)];valid=[np.zeros((100,100)) for _ in range(3)]
            for item in group:
                result=decode_window(item,bbox,signer)
                if result is None: continue
                arrays,tr,meta=result
                for k,array in enumerate(arrays):
                    mean,accepted,_=p.aggregate(array,tr,p.CRS_WKT,fs)
                    sums[k]+=np.nan_to_num(mean)*accepted;valid[k]+=accepted
                sources.append({'id':item['id'],'date':date,'windows':meta})
            for k in range(3):
                mean=np.divide(sums[k],valid[k],out=np.full((100,100),np.nan),where=valid[k]>0)
                fraction=np.clip(valid[k]/area,0,1)
                means[k].append(mean);fractions[k].append(fraction)
            dates.append(date)
            begin=datetime.fromisoformat(date);days.append(min(8,(datetime(YEAR+1,1,1)-begin).days))
    if dates:
        summaries=[summarize(np.stack(means[k]),np.stack(fractions[k]),dates,days) for k in range(3)]
    else:
        summaries=[[np.full((100,100),np.nan) for _ in range(3)]+[np.zeros((100,100))] for _ in range(3)]
    windows_path=OUT/'sources'/f"{site['key']}-windows.json"
    p.save(windows_path,sources)
    source={'dataset':'MOD11A2.061','mirror':'Microsoft Planetary Computer public COGs','platform':'Terra','year':YEAR,'method':METHOD,
            'source_catalog_path':str((OUT/'sources'/f"{site['key']}-catalog.json").relative_to(ROOT)),
            'catalog_sha256':p.sha((OUT/'sources'/f"{site['key']}-catalog.json").read_bytes()),
            'composite_dates':dates,'minimum_valid_composites':12,'required_seasons':4,
            'qa':'Native QC byte exactly zero; temperature raw>=7500, scale .02 K, then Celsius. QC nodata tag is not applied.',
            'spatial_method':'Exact fractional overlap on native sinusoidal pixels; tile pieces combined with accepted area weights; >=50% cell support per usable composite.',
            'temporal_method':'Clear-sky composite means weighted by valid area times duration; low/high are observed 8-day composite extrema, not true daily extrema. Require >=12 composites and all four calendar quarters. Temporal coverage layers include absent dates as zero.',
            'warning':'Approximate cross-body thermal comparison: different overpass times, clear-sky sampling, years and planetary day lengths. Not air temperature or calibrated suitability.',
            'source_windows_path':str(windows_path.relative_to(ROOT)),
            'source_windows_sha256':p.sha(windows_path.read_bytes()),'source_granules':len(sources)}
    features=[('surface_day_c',summaries[0][0]),('surface_night_c',summaries[1][0]),('surface_mean_c',summaries[2][0]),
              ('surface_low_c',summaries[1][1]),('surface_high_c',summaries[0][2])]
    low,high=features[3][1],features[4][1]
    features.append(('surface_range_c',high-low))
    layers=[]
    for key,array in features:
        layers.append(p.write_layer(site,key,array,'degree_Celsius',.02,-.02 if key=='surface_range_c' else -150.02,{**source,'statistic':key},'thermal.surface_temperature'))
    for key,summary in zip(['surface_day_valid_fraction','surface_night_valid_fraction','surface_paired_valid_fraction'],summaries):
        layers.append(p.write_layer(site,key,summary[3],'1',.001,-.001,{**source,'statistic':key},'quality.valid_area_fraction'))
    result={'site_key':site['key'],'processor_sha256':identity,'layers':layers,'composites':len(dates),'seconds':time.monotonic()-start}
    p.save(checkpoint,result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--sites',nargs='*')
    args=parser.parse_args()
    base=json.loads(BASE.read_text());sites=[s for s in base['sites'] if not args.sites or s['key'] in args.sites]
    start=time.monotonic();results=[]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        pending={pool.submit(worker,site):site['key'] for site in sites}
        for future in as_completed(pending):
            result=future.result();results.append(result)
            print(json.dumps({k:v for k,v in result.items() if k!='layers'}),flush=True)
    if args.sites:return
    manifest={'processor_path':'scripts/thermal_pilot.py','script_sha256':p.sha(Path(__file__).read_bytes()),
              'dependency_files':{name:p.sha((ROOT/name).read_bytes()) for name in ['scripts/priority_pilot.py','scripts/process_native_temperature.py','scripts/process_nasa_temperature.py']},
              'base_manifest_sha256':p.sha(BASE.read_bytes()),'generated_at':datetime.now(timezone.utc).isoformat(),
              'status':'research_thermal_pilot','sites':base['sites'],'layers':base['layers']+[l for r in sorted(results,key=lambda r:r['site_key']) for l in r['layers']],
              'source_raster_files_retained':0,'year':YEAR,'wall_seconds':time.monotonic()-start}
    p.save(OUT/'manifest.json',manifest)

if __name__=='__main__':main()
