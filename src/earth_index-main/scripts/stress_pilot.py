# /// script
# requires-python = ">=3.11"
# dependencies = ["rasterio==1.5.2", "numpy==2.5.4", "exactextract==0.3.0", "shapely==2.2.0", "pyproj==3.8.0", "h3==4.5.0", "pyhdf==0.11.7", "h5py==3.16.0"]
# ///
"""Expand the frozen priority pilot with predeclared environmental stress patches.

Independent processes keep GDAL/HDF4 library state isolated. Existing scientific
functions and reference artifacts remain unchanged. Original source rasters are
temporary; only quantized tiles and public provenance persist.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import multiprocessing
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
from pyproj import CRS
import rasterio

import priority_pilot as p

ROOT=p.ROOT
OUT=ROOT/'research/stress-pilot'
DEPENDENCIES=['scripts/priority_pilot.py','scripts/process_native_temperature.py','scripts/prepare_nasa_subsets.py']


def identity():
    return {'processor_path':'scripts/stress_pilot.py','script_sha256':p.sha(Path(__file__).read_bytes()),
            'dependency_files':{f:p.sha((ROOT/f).read_bytes()) for f in DEPENDENCIES},
            'site_plan_sha256':p.sha((OUT/'sites.json').read_bytes()),
            'frozen_priorities_sha256':p.sha((OUT/'priorities.json').read_bytes())}


def retry(function,*args):
    for attempt in range(4):
        try:return function(*args)
        except HTTPError as error:
            if error.code not in (429,500,502,503,504) or attempt==3:
                raise RuntimeError(f'Public source HTTP {error.code}; URL omitted') from None
        except (URLError,TimeoutError):
            if attempt==3:raise RuntimeError('Source connection failed; URL omitted') from None
        time.sleep(2**(attempt+1))


def climate(site,fs):
    # Wider longitude requests are needed near the poles. Cache genuine regional
    # responses, then use the unchanged original decoder/overlap calculation.
    xmin,ymin,xmax,ymax=p.bounds_of(fs);lon,lat=site['center'];urls={}
    for parameter in ('T2M','PRECTOTCORR'):
        url='https://power.larc.nasa.gov/api/temporal/climatology/regional?'+urlencode({
            'parameters':parameter,'community':'RE','longitude-min':round(min(lon-1.1,xmin-.7),5),
            'longitude-max':round(max(lon+1.1,xmax+.7),5),
            'latitude-min':round(max(-90,min(lat-1.1,ymin-.6)),5),
            'latitude-max':round(min(90,max(lat+1.1,ymax+.6)),5),'format':'JSON'})
        path=OUT/'sources'/f"power-{site['key']}-{parameter}.json"
        retry(p.get_json,url,path);urls[parameter]=url
    layers=p.climate(site,fs)
    for layer in layers:
        parameter='PRECTOTCORR' if layer['feature_key']=='annual_precip_mm' else 'T2M'
        layer['source']['url']=urls[parameter]
    return layers


def terrain(site):
    # EarthEnv stops at 56 S. Preserve genuinely absent coverage as NODATA;
    # Antarctica is an intentional source-domain stress case, not a bad site.
    fs=p.cells(site,'EPSG:4326',120);_,ymin,_,ymax=p.bounds_of(fs)
    if ymax<=-56 or ymin>=84:
        source={'dataset':'EarthEnv_topography_GMTED','citation':'https://doi.org/10.1594/PANGAEA.867113',
                'missing_reason':'outside_source_latitude_domain','source_domain_lat':[-56,84],
                'method':'No substitute or imputation outside provider domain.'}
        specs=[('elevation_median_m','m',1,-1001,'terrain.elevation'),
               ('slope_mean_deg','degree',.1,-.1,'terrain.slope'),
               ('terrain_valid_fraction','1',.001,-.001,'quality.valid_area_fraction')]
        for width in (5,21):
            specs.extend([(f'relief_{width}km_m','m',1,-1,'terrain.local_relief'),
                          (f'tpi_{width}km_m','m',.25,-5000.25,'terrain.topographic_position')])
        return [p.write_layer(site,key,np.zeros((100,100)) if key=='terrain_valid_fraction' else np.full((100,100),np.nan),
                              unit,scale,offset,source,var) for key,unit,scale,offset,var in specs]
    return p.terrain(site)


def worker(spec,token,permafrost,glim,version):
    p.OUT=OUT
    checkpoint=OUT/'checkpoints'/f"{spec['key']}.json"
    if checkpoint.exists():
        cached=json.loads(checkpoint.read_text())
        if cached['processor_identity']==version and all(p.sha((ROOT/l['path']).read_bytes())==l['file_sha256'] for l in cached['layers']):
            return cached
    start=time.monotonic();site=dict(spec)
    lon,lat=site['center'];crs=CRS.from_proj4(f'+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m')
    site.update(crs_wkt=crs.to_wkt('WKT1_GDAL'),crs_proj4=crs.to_string())
    layers=[]
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif',
                      GDAL_HTTP_TIMEOUT='45',GDAL_HTTP_MAX_RETRY='2',GDAL_HTTP_RETRY_DELAY='1',GDAL_CACHEMAX=96*1024*1024):
        fs=p.cells(site,'EPSG:4326')
        for label,function,args in [('terrain',terrain,(site,)),('climate',climate,(site,fs)),
                ('land cover',p.landcover,(site,fs,token)),('permafrost',p.permafrost,(site,fs,permafrost)),
                ('lithology',p.lithology,(site,glim))]:
            print(site['key']+': '+label,flush=True)
            layers.extend(retry(function,*args))
    result={'site':site,'layers':layers,'wall_seconds':time.monotonic()-start,'processor_identity':version}
    p.save(checkpoint,result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--credentials-file',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4,choices=range(1,5))
    parser.add_argument('--permafrost-file',type=Path,default=Path('/tmp/priority-permafrost-2023.nc'))
    args=parser.parse_args();start=time.monotonic()
    plan=json.loads((OUT/'sites.json').read_text());frozen=json.loads((OUT/'priorities.json').read_text())
    if p.sha((ROOT/frozen['baseline_file']).read_bytes())!=frozen['baseline_sha256']:
        raise ValueError('Baseline filters changed after stress test declaration')
    base=json.loads((ROOT/'research/priority-pilot/manifest.json').read_text())
    if p.sha((ROOT/'scripts/priority_pilot.py').read_bytes())!=base['script_sha256']:
        raise ValueError('Original processor changed')
    if not args.permafrost_file.exists():
        with urlopen(p.PERMA_URL,timeout=90) as response:args.permafrost_file.write_bytes(response.read())
    with urlopen(p.GLIM_URL,timeout=45) as response:glim=response.read(1024*1024+1)
    if len(glim)>1024*1024:raise ValueError('Unexpected GLiM size')
    token=p.original_token(args.credentials_file);version=identity();results={};failures={}
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        pending={pool.submit(worker,s,token,args.permafrost_file,glim,version):s['key'] for s in plan['sites']}
        for task in as_completed(pending):
            key=pending[task]
            try:results[key]=task.result()
            except Exception as error:
                # Exception strings may include provider URLs. Keep evidence safe.
                failures[key]={'error_type':type(error).__name__}
                print(key+': FAILED ('+type(error).__name__+')',flush=True)
            p.save(OUT/'progress.json',{'completed':sorted(results),'failures':failures,'planned':len(pending)})
    if failures:raise RuntimeError('Some patches failed; see progress.json. Resume to reuse validated checkpoints.')
    sites=[{**s,'role':'reference','plausible_profiles':[],'rationale':'Original development example, not held out.'} for s in base['sites']]
    layers=list(base['layers'])
    for spec in plan['sites']:
        result=results[spec['key']];sites.append(result['site']);layers.extend(result['layers'])
    report={**version,'status':'frozen_filter_stress_pilot','generated_at':datetime.now(timezone.utc).isoformat(),
        'sites':sites,'layers':layers,'source_raster_files_retained':0,'workers':args.workers,
        'reference_manifest_sha256':p.sha((ROOT/'research/priority-pilot/manifest.json').read_bytes()),
        'wall_seconds':time.monotonic()-start,'patch_processing_seconds':{k:r['wall_seconds'] for k,r in results.items()},
        'validation_scope':'Predeclared diverse patches with frozen exploratory filters; not random sampling or certified planetary analog truth.'}
    p.save(OUT/'manifest.json',report)
    print(f"Prepared {len(sites)} patches / {len(layers)} layers in {report['wall_seconds']:.1f}s",flush=True)


if __name__=='__main__':main()
