# /// script
# requires-python = ">=3.11"
# dependencies = ["rasterio==1.5.2", "numpy==2.5.4", "exactextract==0.3.0", "shapely==2.2.0", "pyproj==3.8.0", "h3==4.5.0", "pyhdf==0.11.7", "h5py==3.16.0"]
# ///
"""Optimized thermal import: fetch source assets once, process whole output tiles.

Uses separate download threads and single-threaded CPU worker processes. Keeps
the frozen reference processor and existing datasets unchanged. A new output
directory is required; source downloads are temporary unless explicitly kept.
"""
import os
# Avoid nested BLAS/OpenMP pools in each CPU worker (set before importing NumPy).
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'

import argparse
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import logging
import multiprocessing
from pathlib import Path
import re
import resource
import shutil
import time
from urllib.parse import urlencode

import numpy as np
from pyproj import Transformer
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Polygon, mapping, shape

import priority_pilot as p
import thermal_pilot as reference
from raster_overlap import OverlapPlan
from raster_source_cache import SourceCache, asset_path, atomic_json

ROOT=Path(__file__).resolve().parents[1]
YEAR=2023
DEPENDENCIES=('scripts/thermal_import.py','scripts/raster_overlap.py','scripts/raster_source_cache.py',
              'scripts/thermal_pilot.py','scripts/priority_pilot.py','scripts/process_native_temperature.py',
              'scripts/process_nasa_temperature.py')


def grid_features(site, *, cell_m=4000, side_km=100, target_crs=p.CRS_WKT):
    if cell_m<=0 or side_km<=0 or side_km*1000%cell_m:
        raise ValueError('Region side must contain a whole number of cells')
    n=int(side_km*1000/cell_m);half=side_km*500
    project=Transformer.from_crs(site['crs_wkt'],target_crs,always_xy=True)
    features=[]
    for row in range(n):
        for col in range(n):
            x,y=-half+col*cell_m,half-row*cell_m
            corners=[(x,y),(x+cell_m,y),(x+cell_m,y-cell_m),(x,y-cell_m),(x,y)]
            ring=[]
            for a,b in zip(corners,corners[1:]):
                ring.extend(project.transform(a[0]+(b[0]-a[0])*t/4,a[1]+(b[1]-a[1])*t/4) for t in range(4))
            features.append(dict(type='Feature',properties={'i':row*n+col},geometry=mapping(Polygon(ring))))
    return features,(n,n)


def catalog(site, output, cell_m, side_km):
    path=output/'sources'/f"{site['key']}-catalog.json"
    if path.exists():return json.loads(path.read_text())
    features,_=grid_features(site,cell_m=cell_m,side_km=side_km,target_crs='EPSG:4326')
    bbox=p.bounds_of(features)
    query=dict(collections='modis-11A2-061',bbox=','.join(map(str,bbox)),
               datetime=f'{YEAR}-01-01T00:00:00Z/{YEAR}-12-31T23:59:59Z',
               query=json.dumps({'platform':{'eq':'terra'}}),limit=1000)
    url=reference.STAC+'/search?'+urlencode(query);items=[]
    while url:
        data=reference.get_json(url);items.extend(data['features'])
        url=next((link['href'] for link in data.get('links',[]) if link['rel']=='next'),None)
        if url and not url.startswith(reference.STAC+'/'):raise ValueError('Unexpected catalog pagination host')
    ids=[item['id'] for item in items]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate catalog items')
    for item in items:
        if not re.fullmatch(r'MOD11A2\.A2023\d{3}\.h\d{2}v\d{2}\.061\.\d+',item['id']):
            raise ValueError('Unexpected thermal platform/year/product')
    atomic_json(path,items)
    return items


class CachedSigner:
    def __init__(self, directory):self.directory=directory
    def sign(self, href):return str(asset_path(self.directory,href))


def layers_from_summaries(summaries):
    day,night,paired=summaries
    return [
        ('surface_day_c',day[0],.02,-150.02),('surface_night_c',night[0],.02,-150.02),
        ('surface_mean_c',paired[0],.02,-150.02),('surface_low_c',night[1],.02,-150.02),
        ('surface_high_c',day[2],.02,-150.02),('surface_range_c',day[2]-night[1],.02,-.02),
        ('surface_day_valid_fraction',day[3],.001,-.001),
        ('surface_night_valid_fraction',night[3],.001,-.001),
        ('surface_paired_valid_fraction',paired[3],.001,-.001)]


def compute_tile(site, items, cache, *, cell_m=4000, side_km=100, engine='cached', verify=False, max_plan_bytes=128*1024*1024):
    start=time.perf_counter();cpu=time.process_time()
    features,target_shape=grid_features(site,cell_m=cell_m,side_km=side_km)
    geometry_seconds=time.perf_counter()-start
    bbox=p.bounds_of(features)
    area=np.array([shape(f['geometry']).area for f in features]).reshape(target_shape)
    groups={}
    for item in items:groups.setdefault(item['properties']['start_datetime'][:10],[]).append(item)
    plans=OrderedDict();plan_bytes=0;builds=0;build_seconds=0.;aggregate_seconds=0.;decode_seconds=0.;checks=0
    max_mean_error=0.;max_area_relative_error=0.
    means=[[],[],[]];fractions=[[],[],[]];dates=[];days=[];sources=[]
    signer=CachedSigner(cache)
    logging.getLogger('rasterio._env').setLevel(logging.CRITICAL)
    for date,group in sorted(groups.items()):
        sums=[np.zeros(target_shape) for _ in range(3)];valid=[np.zeros(target_shape) for _ in range(3)]
        for item in sorted(group,key=lambda i:i['id']):
            before=time.perf_counter()
            result=reference.decode_window(item,bbox,signer)
            decode_seconds+=time.perf_counter()-before
            if result is None:continue
            arrays,tr,metadata=result
            plan=None
            if engine=='cached':
                key=(arrays[0].shape,tuple(tr))
                if key in plans:
                    plan=plans.pop(key);plans[key]=plan
                else:
                    before=time.perf_counter()
                    plan=OverlapPlan.build(arrays[0].shape,tr,p.CRS_WKT,features,target_shape=target_shape)
                    build_seconds+=time.perf_counter()-before;builds+=1
                    if plan.nbytes>max_plan_bytes:raise ValueError('Overlap plan exceeds worker memory budget; use smaller output tiles')
                    while plans and plan_bytes+plan.nbytes>max_plan_bytes:
                        _,removed=plans.popitem(last=False);plan_bytes-=removed.nbytes
                    plans[key]=plan;plan_bytes+=plan.nbytes
            for k,array in enumerate(arrays):
                before=time.perf_counter()
                if engine=='cached':mean,accepted,_=plan.aggregate(array)
                elif engine=='reference':mean,accepted,_=p.aggregate(array,tr,p.CRS_WKT,features)
                else:raise ValueError('Unknown aggregation engine')
                aggregate_seconds+=time.perf_counter()-before
                if verify and engine=='cached':
                    expected,expected_area,_=p.aggregate(array,tr,p.CRS_WKT,features)
                    np.testing.assert_allclose(mean,expected,rtol=2e-13,atol=1e-9,equal_nan=True)
                    np.testing.assert_allclose(accepted,expected_area,rtol=2e-13,atol=1e-7)
                    if np.isfinite(mean).any():max_mean_error=max(max_mean_error,float(np.nanmax(np.abs(mean-expected))))
                    max_area_relative_error=max(max_area_relative_error,float(np.max(np.abs(accepted-expected_area)/np.maximum(1.,expected_area))))
                    checks+=1
                sums[k]+=np.nan_to_num(mean)*accepted;valid[k]+=accepted
            sources.append(dict(id=item['id'],date=date,windows=metadata))
        for k in range(3):
            mean=np.divide(sums[k],valid[k],out=np.full(target_shape,np.nan),where=valid[k]>0)
            means[k].append(mean);fractions[k].append(np.clip(valid[k]/area,0,1))
        dates.append(date);days.append(min(8,(datetime(YEAR+1,1,1)-datetime.fromisoformat(date)).days))
    summaries=[reference.summarize(np.stack(means[k]),np.stack(fractions[k]),dates,days) for k in range(3)] if dates else [
        [np.full(target_shape,np.nan) for _ in range(3)]+[np.zeros(target_shape)] for _ in range(3)]
    stats=dict(seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu,geometry_seconds=geometry_seconds,
               decode_seconds=decode_seconds,plan_build_seconds=build_seconds,aggregate_seconds=aggregate_seconds,
               plan_builds=builds,plan_cache_bytes=plan_bytes,source_granules=len(sources),composites=len(dates),
               reference_checks=checks,max_mean_error=max_mean_error,max_area_relative_error=max_area_relative_error,
               max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,pid=os.getpid())
    return dict(site=site,layers=layers_from_summaries(summaries),dates=dates,sources=sources,stats=stats)


def write_tile(job):
    site,items,cache,config=job
    result=compute_tile(site,items,cache,cell_m=config['cell_m'],side_km=config['side_km'],verify=config['verify'])
    output=Path(config['output']);shape=result['layers'][0][1].shape;half=config['side_km']*500
    grid=dict(shape=list(shape),transform=[config['cell_m'],0,-half,0,-config['cell_m'],half],crs_wkt=site['crs_wkt'])
    layers=[]
    provenance=dict(dataset='MOD11A2.061',mirror='Microsoft Planetary Computer public COGs',year=YEAR,
        method='mod11a2-terra-cog-2023-qc0-reusable-exact-overlap-v1',composite_dates=result['dates'],
        qa='QC byte exactly zero; native LST >=7500; .02 K scaling, then Celsius; pair requires both day/night valid.',
        spatial_method='Cached exactextract Cartesian-area overlaps; valid area combined across native source tiles before temporal summaries.',
        temporal_method='Unchanged reference support rules and duration/valid-area weighting; night composite low, day composite high.',
        minimum_valid_composites=12,required_seasons=4,source_granules=len(result['sources']),
        warning='Clear-sky 8-day composite statistics, not daily extremes or all-weather climate.')
    sources_path=output/'sources'/f"{site['key']}-windows.json";atomic_json(sources_path,result['sources'])
    artifacts={str(sources_path.relative_to(ROOT)):p.sha(sources_path.read_bytes())}
    catalog_path=output/'sources'/f"{site['key']}-catalog.json"
    if catalog_path.exists():
        artifacts[str(catalog_path.relative_to(ROOT))]=p.sha(catalog_path.read_bytes())
        provenance['source_catalog_path']=str(catalog_path.relative_to(ROOT))
        provenance['source_catalog_sha256']=artifacts[str(catalog_path.relative_to(ROOT))]
    provenance['source_windows_path']=str(sources_path.relative_to(ROOT))
    provenance['source_windows_sha256']=artifacts[str(sources_path.relative_to(ROOT))]
    for key,array,step,offset in result['layers']:
        codes,error=p.quantize(array,step,offset)
        path=output/'tiles'/f"{site['key']}--{key}.tif"
        temporary=path.with_suffix('.tmp.tif')
        try:
            with rasterio.open(temporary,'w',driver='GTiff',height=shape[0],width=shape[1],count=1,dtype='uint16',
                               nodata=0,crs=site['crs_wkt'],transform=from_origin(-half,half,config['cell_m'],config['cell_m'])) as dst:
                dst.write(codes,1)
            temporary.replace(path)
        finally:temporary.unlink(missing_ok=True)
        support=key.endswith('_valid_fraction')
        layers.append(dict(site_key=site['key'],feature_key=key,variable_key='quality.valid_area_fraction' if support else 'thermal.surface_temperature',
            unit='1' if support else 'degree_Celsius',scale=step,offset=offset,nodata=0,path=str(path.relative_to(ROOT)),
            file_sha256=p.sha(path.read_bytes()),codes_sha256=p.sha(codes.tobytes()),grid_definition=grid,
            valid_cells=int(np.isfinite(array).sum()),max_rounding_error=error,
            physical_min=float(np.nanmin(array)) if np.isfinite(array).any() else None,
            physical_max=float(np.nanmax(array)) if np.isfinite(array).any() else None,source=provenance|dict(statistic=key)))
    checkpoint=dict(site_key=site['key'],run_identity=config['identity'],layers=layers,source_artifacts=artifacts,**result['stats'])
    atomic_json(output/'checkpoints'/f"{site['key']}.json",checkpoint)
    return checkpoint


def valid_checkpoint(path, identity):
    if not path.exists():return None
    try:
        data=json.loads(path.read_text())
        if (data['run_identity']==identity
            and all(p.sha((ROOT/l['path']).read_bytes())==l['file_sha256'] for l in data['layers'])
            and all(p.sha((ROOT/name).read_bytes())==checksum for name,checksum in data['source_artifacts'].items())):
            return data
    except (ValueError,KeyError,OSError):pass
    return None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-manifest',type=Path,default=ROOT/'research/stress-pilot/manifest.json')
    selection=parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--sites',nargs='+');selection.add_argument('--all-sites',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cell-m',type=int,default=4000)
    parser.add_argument('--side-km',type=int,default=100)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--download-workers',type=int,default=8)
    parser.add_argument('--cache-dir',type=Path,help='Explicit shared cache, retained after success')
    parser.add_argument('--keep-source-cache',action='store_true')
    parser.add_argument('--verify',action='store_true',help='Compare every optimized aggregate with reference (slower)')
    args=parser.parse_args()
    if args.workers<1 or args.download_workers<1:parser.error('Worker counts must be positive')
    output=args.output.resolve()
    if not output.is_relative_to(ROOT/'research'):parser.error('Output must be a new directory under research/')
    for protected in ('priority-pilot','stress-pilot','thermal-pilot','pilots','import-profile'):
        directory=ROOT/'research'/protected
        if output==directory or output.is_relative_to(directory) or directory.is_relative_to(output):parser.error('Output overlaps an existing protected dataset')
    base=json.loads(args.base_manifest.read_text())
    requested=set(args.sites or [s['key'] for s in base['sites']])
    sites=[s for s in base['sites'] if s['key'] in requested]
    if len(sites)!=len(requested):parser.error('Unknown or duplicated site identifiers')
    if any(not re.fullmatch(r'[a-z0-9_-]+',s['key']) for s in sites):parser.error('Invalid site identifier')
    grid_features(sites[0],cell_m=args.cell_m,side_km=args.side_km)
    dependencies={name:p.sha((ROOT/name).read_bytes()) for name in DEPENDENCIES}
    identity_data=dict(dependencies=dependencies,base_manifest_sha256=p.sha(args.base_manifest.read_bytes()),
                       sites=sites,cell_m=args.cell_m,side_km=args.side_km,verify=args.verify)
    identity=p.sha(json.dumps(identity_data,sort_keys=True).encode())
    if output.exists() and any(output.iterdir()):
        run=output/'run.json'
        if not run.exists() or json.loads(run.read_text()).get('identity')!=identity:parser.error('Output exists with a different run identity; choose a new directory')
    for child in ('sources','tiles','checkpoints'):(output/child).mkdir(parents=True,exist_ok=True)
    atomic_json(output/'run.json',dict(identity=identity,**identity_data))
    start=time.perf_counter();results=[];remaining=[]
    for site in sites:
        previous=valid_checkpoint(output/'checkpoints'/f"{site['key']}.json",identity)
        if previous:results.append(previous)
        else:remaining.append(site)
    catalogs={s['key']:catalog(s,output,args.cell_m,args.side_km) for s in remaining}
    hrefs=[item['assets'][band]['href'] for items in catalogs.values() for item in items for band in reference.BANDS]
    cache=args.cache_dir.resolve() if args.cache_dir else output/'.source-cache'
    before=time.perf_counter();downloads=SourceCache(cache).fetch_all(hrefs,args.download_workers) if hrefs else []
    download_seconds=time.perf_counter()-before
    print(json.dumps(dict(stage='downloads',unique_assets=len(downloads),bytes=sum(d['bytes'] for d in downloads if not d['cached']),
                          cache_hits=sum(d['cached'] for d in downloads),seconds=download_seconds)),flush=True)
    inventory_path=output/'sources/downloads.json'
    inventory={d['href']:d for d in json.loads(inventory_path.read_text())} if inventory_path.exists() else {}
    for download in downloads:
        stable={k:download[k] for k in ('href','sha256','bytes')}
        previous=inventory.get(stable['href'])
        if previous and previous['sha256']!=stable['sha256']:
            raise ValueError('Source asset content changed during resumed import; choose a new output directory')
        inventory[stable['href']]=stable
    atomic_json(inventory_path,[inventory[key] for key in sorted(inventory)])
    config=dict(output=str(output),cell_m=args.cell_m,side_km=args.side_km,verify=args.verify,identity=identity)
    before=time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as workers:
        pending=[workers.submit(write_tile,(site,catalogs[site['key']],str(cache),config)) for site in remaining]
        for future in as_completed(pending):
            result=future.result();results.append(result)
            print(json.dumps({k:v for k,v in result.items() if k not in ('layers','run_identity')}),flush=True)
    cpu_wall=time.perf_counter()-before
    # Only matching 1km/100km base grids can be reused without a separate base-data migration.
    base_layers=[l for l in base['layers'] if l['site_key'] in requested and not l['feature_key'].startswith('surface_')] if args.cell_m==1000 and args.side_km==100 else []
    manifest=dict(processor_path='scripts/thermal_import.py',script_sha256=dependencies['scripts/thermal_import.py'],
        dependency_files={k:v for k,v in dependencies.items() if k!='scripts/thermal_import.py'},
        base_manifest_sha256=identity_data['base_manifest_sha256'],run_identity=identity,
        generated_at=datetime.now(timezone.utc).isoformat(),status='research_optimized_thermal_pilot',sites=sites,
        layers=base_layers+[l for r in sorted(results,key=lambda r:r['site_key']) for l in r['layers']],
        cell_m=args.cell_m,side_km=args.side_km,year=YEAR,workers=args.workers,download_workers=args.download_workers,
        download_seconds=download_seconds,processing_seconds=cpu_wall,wall_seconds=time.perf_counter()-start,
        source_cache_retained=bool(args.keep_source_cache or args.cache_dir),
        source_inventory_path=str(inventory_path.relative_to(ROOT)),source_inventory_sha256=p.sha(inventory_path.read_bytes()),
        coverage_note='Thermal-only output unless matching existing 1km/100km base layers are included; not a full coverage expansion.')
    atomic_json(output/'manifest.json',manifest)
    if not args.keep_source_cache and not args.cache_dir and cache.exists():shutil.rmtree(cache)
    print(json.dumps(dict(stage='complete',manifest=str(output/'manifest.json'),layers=len(manifest['layers']),seconds=manifest['wall_seconds'])),flush=True)


if __name__=='__main__':main()
