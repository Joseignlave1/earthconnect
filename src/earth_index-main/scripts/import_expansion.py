# /// script
# requires-python = ">=3.11"
# dependencies = ["rasterio==1.5.2", "numpy==2.5.4", "exactextract==0.3.0", "shapely==2.2.0", "pyproj==3.8.0", "h3==4.5.0", "pyhdf==0.11.7", "h5py==3.16.0"]
# ///
"""Execute a frozen, cold-cache 25-variable expansion; preserve original pilots."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import argparse
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import re
import resource
import shutil
import time
from urllib.parse import urlencode
from urllib.request import Request,urlopen

import numpy as np
import rasterio
from rasterio.transform import from_origin

import priority_pilot as p
import thermal_import as t
import thermal_pilot as thermal
import expansion_features as base
from raster_source_cache import SourceCache,atomic_json

ROOT=p.ROOT
DEPS=['scripts/import_expansion.py','scripts/expansion_features.py',*t.DEPENDENCIES]


def log(message):print(datetime.now(timezone.utc).isoformat(timespec='seconds')+' '+message,flush=True)


def download(job):
    url,path=job;path=Path(path)
    if path.exists():return json.loads(path.with_suffix(path.suffix+'.json').read_text())|dict(cached=True)
    before=time.perf_counter();temporary=path.with_suffix(path.suffix+'.part')
    for attempt in range(3):
        try:
            checksum=hashlib.sha256();size=0
            with urlopen(Request(url,headers={'User-Agent':'EarthIndex/expansion'}),timeout=120) as response,temporary.open('wb') as f:
                while chunk:=response.read(4*1024*1024):
                    f.write(chunk);checksum.update(chunk);size+=len(chunk)
            temporary.replace(path)
            result=dict(uri=url,bytes=size,sha256=checksum.hexdigest(),seconds=time.perf_counter()-before,cached=False)
            atomic_json(path.with_suffix(path.suffix+'.json'),result)
            return result
        except Exception:
            temporary.unlink(missing_ok=True)
            if attempt==2:raise
            time.sleep(2**attempt)


def climate_requests(site):
    fs,_=t.grid_features(site,cell_m=40000,side_km=400,target_crs=base.GEO)
    xmin,ymin,xmax,ymax=p.bounds_of(fs)
    # <=5x5 degree API requests; include one coarse native cell of padding.
    result=[]
    for x in range(math.floor((xmin-.7)/5)*5,math.floor((xmax+.7)/5)*5+1,5):
        for y in range(max(-90,math.floor((ymin-.6)/5)*5),min(85,math.floor((ymax+.6)/5)*5)+1,5):
            for parameter in ('T2M','PRECTOTCORR'):
                bounds=(max(-180,x),max(-90,y),min(180,x+5),min(90,y+5))
                url='https://power.larc.nasa.gov/api/temporal/climatology/regional?'+urlencode(dict(
                    parameters=parameter,community='RE',**{'longitude-min':bounds[0],'latitude-min':bounds[1],
                    'longitude-max':bounds[2],'latitude-max':bounds[3]},format='JSON'))
                result.append((url,f'power-{parameter}-{x}-{y}.json'))
    return result


def land_download(job):
    granule,cache,token=job;path=cache/(granule['title']+'.hdf')
    if path.exists():return json.loads(path.with_suffix('.json').read_text())|dict(cached=True)
    start=time.perf_counter()
    for attempt in range(3):
        try:
            payload,meta=p.nasa_download(granule['title'],token)
            tmp=path.with_suffix('.part');tmp.write_bytes(payload);tmp.replace(path)
            meta.update(seconds=time.perf_counter()-start,cached=False);atomic_json(path.with_suffix('.json'),meta);return meta
        except Exception:
            if attempt==2:raise
            time.sleep(2**attempt)


def process(job):
    site,items,config=job;start=time.perf_counter();cpu=time.process_time()
    output=Path(config['output']);cache=Path(config['cache'])
    fs,_=t.grid_features(site,cell_m=4000,side_km=400,target_crs=base.GEO)
    projected,_=t.grid_features(site,cell_m=4000,side_km=400)
    catalog={re.search(r'h\d{2}v\d{2}',g['title'])[0]:g for g in json.loads((output/'landcover-catalog.json').read_text())}
    values={};provenance={};stages={}
    for name,fn in [
        ('terrain',lambda:base.terrain(site,cache)),
        ('climate',lambda:base.climate(fs,[output/'sources'/name for _,name in climate_requests(site)])),
        ('cover',lambda:base.cover(site,projected,cache,catalog)),
        ('permafrost',lambda:base.permafrost(fs,cache)),
        ('lithology',lambda:base.lithology(site,cache))]:
        before=time.perf_counter();v,s=fn();values.update(v);provenance.update(s);stages[name]=time.perf_counter()-before
    checkpoint=t.write_tile((site,items,cache/'thermal',dict(output=str(output),cell_m=4000,side_km=400,verify=False,identity=config['identity'])))
    specs={l['feature_key']:l for l in json.loads((ROOT/'research/thermal-pilot/manifest.json').read_text())['layers']}
    layers=checkpoint['layers'];grid=layers[0]['grid_definition']
    for key,a in sorted(values.items()):
        spec=specs[key];codes,error=p.quantize(a,spec['scale'],spec['offset'])
        path=output/'tiles'/f"{site['key']}--{key}.tif";temporary=path.with_suffix('.tmp.tif')
        with rasterio.open(temporary,'w',driver='GTiff',height=100,width=100,count=1,dtype='uint16',nodata=0,
                           crs=site['crs_wkt'],transform=from_origin(-200000,200000,4000,4000)) as dst:dst.write(codes,1)
        temporary.replace(path)
        layer={k:spec[k] for k in ('feature_key','variable_key','unit','scale','offset','nodata')}
        layer.update(site_key=site['key'],path=str(path.relative_to(ROOT)),grid_definition=grid,file_sha256=p.sha(path.read_bytes()),
            codes_sha256=p.sha(codes.tobytes()),valid_cells=int(np.isfinite(a).sum()),max_rounding_error=error,
            physical_min=float(np.nanmin(a)) if np.isfinite(a).any() else None,physical_max=float(np.nanmax(a)) if np.isfinite(a).any() else None,
            source=provenance[key]|dict(source_inventory='sources/downloads.json',source_inventory_sha256=config['inventory_hash']))
        layers.append(layer)
    if len(layers)!=25:raise ValueError('Incomplete feature set')
    checkpoint.update(layers=layers,base_stage_seconds=stages,seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu,
                      max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    checkpoint['source_artifacts'][str((output/'sources/downloads.json').relative_to(ROOT))]=config['inventory_hash']
    atomic_json(output/'checkpoints'/f"{site['key']}.json",checkpoint)
    return checkpoint


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'research/expansion-4km')
    parser.add_argument('--token-file',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();out=args.output.resolve()
    if not out.is_relative_to(ROOT/'research'):raise ValueError('Output must be under research/')
    plan=json.loads((out/'plan.json').read_text())
    if plan['cell_m']!=4000 or plan['side_km']!=400:raise ValueError('Unsupported experiment grid')
    if (out/'manifest.json').exists():raise ValueError('Experiment already completed')
    for directory in ('sources','tiles','checkpoints','raw-cache'):(out/directory).mkdir(exist_ok=True)
    cache=out/'raw-cache';(cache/'thermal').mkdir(exist_ok=True)
    dependencies={name:p.sha((ROOT/name).read_bytes()) for name in sorted(set(DEPS))}
    identity=p.sha(json.dumps(dict(plan=plan,dependencies=dependencies),sort_keys=True).encode())
    state=out/'execution.json'
    if state.exists():
        if not args.resume:raise ValueError('Existing run: use --resume to preserve its first-start timing')
        execution=json.loads(state.read_text())
        if execution['identity']!=identity:raise ValueError('Processor/plan changed since run start')
    else:
        execution=dict(identity=identity,started_at=datetime.now(timezone.utc).isoformat(),started_epoch=time.time(),forecast=plan['forecast'],stages={})
        atomic_json(state,execution)
    start=time.perf_counter();stages=execution['stages'];log('Cold experiment started; discovering thermal source catalog')
    items_by_site={};unique={}
    for index,site in enumerate(plan['sites']):
        items=t.catalog(site,out,40000,400)
        items=[i for i in items if re.search(r'h\d{2}v\d{2}',i['id'])[0] in site['native_tiles']]
        items_by_site[site['key']]=items
        for i in items:unique[i['id']]=i
        if index%5==0:log(f'Catalog {index+1}/{len(plan["sites"])}')
    stages['catalog_seconds']=time.perf_counter()-start;atomic_json(state,execution)
    hrefs=sorted({item['assets'][key]['href'] for item in unique.values() for key in ('LST_Day_1km','QC_Day','LST_Night_1km','QC_Night')})
    log(f'{len(unique)} thermal granules, {len(hrefs)} unique COGs; downloading with 8 workers')
    before=time.perf_counter()
    # Separate bounded pools: transfer thermal while preparing shared small/base sources.
    with ThreadPoolExecutor(max_workers=1) as background:
        thermal_future=background.submit(SourceCache(cache/'thermal').fetch_all,hrefs,workers=8)
        globals=[(p.TOPO_BASE+'elevation/elevation_1KMmd_GMTEDmd.tif',cache/'elevation.tif'),
                 (p.TOPO_BASE+'slope/slope_1KMmn_GMTEDmd.tif',cache/'slope.tif'),
                 (p.PERMA_URL,cache/'permafrost.nc'),(p.GLIM_URL,cache/'lithology.zip')]
        with ThreadPoolExecutor(max_workers=4) as pool:global_records=list(pool.map(download,globals))
        log('Shared terrain/permafrost/lithology downloaded')
        token=args.token_file.read_text().strip()
        catalog=json.loads((out/'landcover-catalog.json').read_text())
        with ThreadPoolExecutor(max_workers=8) as pool:land_records=list(pool.map(land_download,[(g,cache,token) for g in catalog]))
        del token
        log(f'{len(land_records)} shared land-cover granules downloaded')
        requests={url:name for site in plan['sites'] for url,name in climate_requests(site)}
        with ThreadPoolExecutor(max_workers=4) as pool:power_records=list(pool.map(download,[(url,out/'sources'/name) for url,name in requests.items()]))
        log(f'{len(power_records)} POWER responses downloaded; waiting for thermal transfers')
        thermal_records=thermal_future.result()
    stages['download_seconds']=time.perf_counter()-before
    inventory=dict(global_sources=global_records,landcover=land_records,power=power_records,thermal=thermal_records)
    atomic_json(out/'sources/downloads.json',inventory)
    execution['download_bytes']=sum(r['bytes'] for group in inventory.values() for r in group)
    atomic_json(state,execution);log(f'Download stage complete: {execution["download_bytes"]/1e9:.3f} GB in {stages["download_seconds"]:.1f}s')
    config=dict(output=str(out),cache=str(cache),identity=identity,inventory_hash=p.sha((out/'sources/downloads.json').read_bytes()))
    before=time.perf_counter();results=[];jobs=[]
    for site in plan['sites']:
        previous=t.valid_checkpoint(out/'checkpoints'/f"{site['key']}.json",identity)
        if previous and len(previous['layers'])==25:results.append(previous)
        else:jobs.append((site,items_by_site[site['key']],config))
    with ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures={pool.submit(process,job):job[0]['key'] for job in jobs}
        for future in as_completed(futures):
            result=future.result();results.append(result)
            log(f'Processed {len(results)}/{len(plan["sites"])}: {result["site_key"]}, {result["seconds"]:.1f}s, CPU {result["cpu_seconds"]:.1f}s')
    stages['processing_seconds']=time.perf_counter()-before
    results.sort(key=lambda r:r['site_key'])
    report=dict(status='research_pilot',generated_at=datetime.now(timezone.utc).isoformat(),processor_path='scripts/import_expansion.py',
        script_sha256=dependencies['scripts/import_expansion.py'],dependency_files=dependencies,sites=plan['sites'],
        layers=[l for r in results for l in r['layers']],cell_m=4000,side_km=400,run_identity=identity,
        wall_seconds=time.time()-execution['started_epoch'],stage_seconds=stages,download_bytes=execution['download_bytes'],
        cpu_seconds=sum(r['cpu_seconds'] for r in results),source_raster_files_retained='temporary pending validation',
        validation_scope='Purposive diverse expansion, source masks and unchanged matching; not calibrated site suitability or full global coverage.')
    atomic_json(out/'manifest.json',report)
    execution.update(completed_at=report['generated_at'],wall_seconds=report['wall_seconds'],status='processed_pending_validation')
    atomic_json(state,execution);log(f'All {len(report["layers"])} layers prepared in {report["wall_seconds"]:.1f}s; caches retained for validation')


if __name__=='__main__':main()
