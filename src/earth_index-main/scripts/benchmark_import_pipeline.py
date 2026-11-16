# /// script
# requires-python = ">=3.11"
# dependencies = ["rasterio==1.5.2", "numpy==2.5.4", "exactextract==0.3.0", "shapely==2.2.0", "pyproj==3.8.0", "h3==4.5.0", "pyhdf==0.11.7", "h5py==3.16.0"]
# ///
"""Read-only import profiling; never changes a datasource or frozen processor.

Source files live in a temporary cache. SAS signatures stay in memory. A local
forwarder measures actual HTTP payload bytes with pooled TLS connections; logs
contain only unsigned asset identifiers. Timings are samples, not global ETAs.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import argparse
import copy
import hashlib
import http.client
import json
import logging
import multiprocessing
from pathlib import Path
import queue
import resource
import ssl
import statistics
import threading
import time
from urllib.parse import urlparse
import uuid

import numpy as np
from pyproj import Transformer
import rasterio
from rasterio.windows import Window, from_bounds
from shapely.geometry import Polygon, mapping
import priority_pilot as p
import thermal_pilot as thermal

ROOT = Path(__file__).resolve().parents[1]
BANDS = thermal.BANDS
ENV = dict(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif',
           GDAL_HTTP_TIMEOUT='45', GDAL_HTTP_MAX_RETRY='0', GDAL_CACHEMAX=48*1024*1024)
FEATURES = {}


def digest(data): return hashlib.sha256(data).hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def features(site, cell_m):
    key = site['key'], cell_m
    if key in FEATURES: return FEATURES[key]
    start = time.perf_counter()
    project = Transformer.from_crs(site['crs_wkt'], p.CRS_WKT, always_xy=True)
    half = 50*cell_m; fs = []
    for r in range(100):
        for c in range(100):
            x, y = -half+c*cell_m, half-r*cell_m
            corners = [(x,y),(x+cell_m,y),(x+cell_m,y-cell_m),(x,y-cell_m),(x,y)]
            ring = []
            for a,b in zip(corners,corners[1:]):
                ring.extend(project.transform(a[0]+(b[0]-a[0])*t/4, a[1]+(b[1]-a[1])*t/4) for t in range(4))
            fs.append({'type':'Feature','properties':{'i':r*100+c},'geometry':mapping(Polygon(ring))})
    result = fs, p.bounds_of(fs), time.perf_counter()-start
    FEATURES[key] = result
    return result


class HttpPool:
    def __init__(self):
        self.connections = {}; self.lock = threading.Lock()
    def fetch(self, url, method='GET', headers=None):
        parsed = urlparse(url)
        if parsed.scheme != 'https': raise ValueError('HTTPS required')
        with self.lock: pool = self.connections.setdefault(parsed.hostname, queue.LifoQueue())
        start = time.perf_counter()
        last_code = None
        for attempt in range(2):
            try: connection = pool.get_nowait()
            except queue.Empty: connection = http.client.HTTPSConnection(parsed.hostname, timeout=45, context=ssl.create_default_context())
            try:
                connection.request(method, parsed.path+('?' + parsed.query if parsed.query else ''),
                                   headers={'User-Agent':'EarthIndex/import-profile', 'Accept-Encoding':'identity', **(headers or {})})
                response = connection.getresponse(); first = time.perf_counter()
                body = response.read(); info = dict(response.getheaders()); code = response.status
                if response.will_close: connection.close()
                else: pool.put(connection)
                if code not in (200,206):
                    last_code = code
                    if code in (429,500,502,503,504) and not attempt: time.sleep(1); continue
                    raise RuntimeError(f'Public asset HTTP {code}')
                return body, info, {'seconds':time.perf_counter()-start, 'ttfb_seconds':first-start,
                                     'bytes':len(body), 'status':code, 'retries':attempt}
            except (OSError, http.client.HTTPException):
                connection.close()
                if attempt: raise RuntimeError('Public asset connection failed') from None
        raise RuntimeError(f'Public asset failed ({last_code})')
    def close(self):
        for pool in self.connections.values():
            while not pool.empty(): pool.get_nowait().close()


class Forwarder:
    def __init__(self, assets, pool):
        self.assets = assets; self.pool = pool; self.records = []; self.lock = threading.Lock()
        owner = self
        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'
            def log_message(self, *args): pass
            def do_HEAD(self): self.forward('HEAD')
            def do_GET(self): self.forward('GET')
            def forward(self, method):
                parts = urlparse(self.path).path.strip('/').split('/')
                key = parts[-1].removesuffix('.tif')
                if len(parts)!=2 or key not in owner.assets:
                    self.send_error(404); return
                try:
                    headers = {'Range':self.headers['Range']} if self.headers.get('Range') else {}
                    body, info, record = owner.pool.fetch(owner.assets[key]['signed'], method, headers)
                    record.update(run=parts[0], asset=key, method=method, range=headers.get('Range'))
                    with owner.lock: owner.records.append(record)
                    self.send_response(record['status'])
                    for k,v in info.items():
                        if k.lower() in ('content-length','content-range','content-type','etag','last-modified','accept-ranges'):
                            self.send_header(k,v)
                    self.end_headers()
                    if method=='GET': self.wfile.write(body)
                except Exception as error:
                    with owner.lock: owner.records.append({'run':parts[0], 'asset':key, 'method':method, 'error':type(error).__name__})
                    self.send_error(502)
        self.server = ThreadingHTTPServer(('127.0.0.1',0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True); self.thread.start()
    def url(self, run, key): return f'http://127.0.0.1:{self.server.server_port}/{run}/{key}.tif'
    def stop(self): self.server.shutdown(); self.server.server_close(); self.thread.join()


class IdentitySigner:
    def sign(self, url): return url


def decode_local(group, assets, cache, cell_m):
    fs, bbox, build_seconds = features(group['site'],cell_m)
    start = time.perf_counter(); cpu = time.process_time(); arrays = {}; transform = None; decoded = {}
    for band in BANDS:
        key = group['keys'][band]
        with rasterio.open(cache/(key+'.tif')) as src:
            window = from_bounds(*bbox, src.transform)
            x0=max(0,int(np.floor(window.col_off))-1); y0=max(0,int(np.floor(window.row_off))-1)
            x1=min(src.width,int(np.ceil(window.col_off+window.width))+1); y1=min(src.height,int(np.ceil(window.row_off+window.height))+1)
            if x1<=x0 or y1<=y0: raise RuntimeError('Sample does not intersect tile')
            window = Window(x0,y0,x1-x0,y1-y0)
            raw = src.read(1,window=window,masked=False)
            arrays[band] = raw; transform = src.window_transform(window); decoded[band] = digest(raw.tobytes())
    day=np.where((arrays['LST_Day_1km']>=7500)&(arrays['QC_Day']==0),arrays['LST_Day_1km'].astype(float)*.02-273.15,np.nan)
    night=np.where((arrays['LST_Night_1km']>=7500)&(arrays['QC_Night']==0),arrays['LST_Night_1km'].astype(float)*.02-273.15,np.nan)
    return (day,night,(day+night)/2), transform, decoded, time.perf_counter()-start, time.process_time()-cpu, build_seconds


def cpu_task(job):
    group, cache, cell_m, repeat = job
    start = time.perf_counter(); cpu = time.process_time()
    arrays,tr,decoded,decode_seconds,decode_cpu,geometry_seconds = decode_local(group,None,Path(cache),cell_m)
    fs,_,_ = features(group['site'],cell_m)
    agg_start=time.perf_counter(); agg_cpu=time.process_time(); hashes=[]
    for array in arrays:
        mean,valid,total = p.aggregate(array,tr,p.CRS_WKT,fs)
        # Same quantization as the frozen thermal layers; ensure equal science at every worker count.
        codes,error = p.quantize(mean,.02,-150.02)
        hashes.append(digest(codes.tobytes()+np.asarray(valid,dtype='<f8').tobytes()))
    return dict(site=group['site']['key'], item=group['item']['id'], repeat=repeat,
                seconds=time.perf_counter()-start, cpu_seconds=time.process_time()-cpu,
                decode_seconds=decode_seconds, decode_cpu_seconds=decode_cpu,
                aggregate_seconds=time.perf_counter()-agg_start, aggregate_cpu_seconds=time.process_time()-agg_cpu,
                geometry_seconds=geometry_seconds, hashes=hashes, native_hashes=decoded,
                max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, pid=os.getpid())


def metadata_sources():
    # Only paths/unsigned metadata are recorded, never credentials.
    manifest=json.loads((ROOT/'research/thermal-pilot/manifest.json').read_text())
    result={}
    for layer in manifest['layers']:
        key=layer['feature_key']
        if key not in result: result[key] = layer['source'].get('dataset')
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--items',type=Path,default=ROOT/'research/import-profile/sample-items.json')
    ap.add_argument('--cache',type=Path,default=Path('/tmp/earth-import-profile-cache'))
    ap.add_argument('--report',type=Path,default=ROOT/'research/import-profile/benchmark.json')
    ap.add_argument('--network-repeats',type=int,default=2)
    ap.add_argument('--cpu-repeats',type=int,default=6)
    ap.add_argument('--skip-network',action='store_true')
    args=ap.parse_args(); args.cache.mkdir(parents=True,exist_ok=True)
    groups=json.loads(args.items.read_text()); assets={}; signer=thermal.Signer(); pool=HttpPool()
    for group in groups:
        group['keys']={}
        for band in BANDS:
            href=group['item']['assets'][band]['href']; key=digest(href.encode())[:20]
            group['keys'][band]=key
            assets[key]={'href':href,'signed':signer.sign(href),'band':band,'item':group['item']['id']}
    report={'generated_at':datetime.now(timezone.utc).isoformat(),'status':'running',
            'processor_sha256':digest(Path(__file__).read_bytes()),
            'machine':{'logical_cpus':os.cpu_count(),'mem_total_kib':next(int(l.split()[1]) for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemTotal:'))},
            'variables':metadata_sources(), 'sample':[{ 'site':g['site']['key'],'item':g['item']['id']} for g in groups],
            'assets':{k:{a:v for a,v in asset.items() if a!='signed'} for k,asset in assets.items()},
            'windows':[],'network':[],'cpu':[]}
    logging.getLogger('rasterio._env').setLevel(logging.CRITICAL)
    def checkpoint(label):
        save(args.report,report); print(json.dumps({'stage':label,'windows':len(report['windows']),'network':len(report['network']),'cpu':len(report['cpu'])}),flush=True)
    if not args.skip_network:
        proxy=Forwarder(assets,pool)
        try:
            for cell_m in (1000,4000):
                run=uuid.uuid4().hex; start=time.perf_counter(); cpu=time.process_time(); outcomes=[]
                for group in groups:
                    _,bbox,geometry_seconds=features(group['site'],cell_m)
                    item=copy.deepcopy(group['item'])
                    for band,key in group['keys'].items(): item['assets'][band]['href']=proxy.url(run,key)
                    before=time.perf_counter()
                    with rasterio.Env(**ENV): arrays,tr,metadata=thermal.decode_window(item,bbox,IdentitySigner())
                    outcomes.append({'site':group['site']['key'],'item':item['id'],'read_seconds':time.perf_counter()-before,
                                     'geometry_seconds':geometry_seconds,'hashes':{m['band']:m['decoded_sha256'] for m in metadata}})
                requests=[r for r in proxy.records if r['run']==run]
                report['windows'].append({'cell_m':cell_m,'cells_per_region':10000,'region_side_km':cell_m/10,
                    'wall_seconds':time.perf_counter()-start,'cpu_seconds':time.process_time()-cpu,
                    'http_seconds_serial':sum(r.get('seconds',0) for r in requests),'http_bytes':sum(r.get('bytes',0) for r in requests),
                    'requests':requests,'samples':outcomes})
                checkpoint(f'remote_windows_{cell_m}')
        finally: proxy.stop()
        # Two orders reduce the first-pass versus warmed-provider-cache bias.
        for repeat in range(args.network_repeats):
            for workers in ([1,4,8,16] if repeat%2==0 else [16,8,4,1]):
                def download(pair):
                    key,asset=pair
                    body,headers,stats=pool.fetch(asset['signed'])
                    file=args.cache/(key+'.tif'); checksum=digest(body)
                    if file.exists():
                        if digest(file.read_bytes())!=checksum: raise RuntimeError('Source file changed during benchmark')
                    else: file.write_bytes(body)
                    return {'asset':key,'sha256':checksum,**stats}
                start=time.perf_counter(); cpu=time.process_time()
                with ThreadPoolExecutor(max_workers=workers) as threads: results=list(threads.map(download,assets.items()))
                report['network'].append({'workers':workers,'repeat':repeat,'wall_seconds':time.perf_counter()-start,
                    'cpu_seconds':time.process_time()-cpu,'bytes':sum(r['bytes'] for r in results),'files':results})
                checkpoint(f'full_COG_downloads_{workers}_{repeat}')
    pool.close()
    reference={}
    for workers in (1,4,8,16):
        jobs=[(g,str(args.cache),4000,i) for i in range(args.cpu_repeats) for g in groups]
        start=time.perf_counter()
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as processes:
            results=list(processes.map(cpu_task,jobs,chunksize=1))
        for r in results:
            key=r['site'],r['item']; previous=reference.setdefault(key,(r['hashes'],r['native_hashes']))
            if previous!=(r['hashes'],r['native_hashes']): raise RuntimeError('Parallelism changed decoded or aggregated values')
        report['cpu'].append({'workers':workers,'jobs':len(jobs),'wall_seconds':time.perf_counter()-start,
            'task_cpu_seconds':sum(r['cpu_seconds'] for r in results),'samples':results})
        checkpoint(f'local_decode_aggregate_{workers}')
    # Remote and local pixel reads must be byte-identical before interpreting timings.
    if report['windows']:
        observed={(s['site'],s['item']):s['hashes'] for s in report['windows'][-1]['samples']}
        if any(observed[key]!=raw for key,(_,raw) in reference.items()): raise RuntimeError('Remote/local window decoding differs')
    report['status']='complete';report['validation']='Remote/local decoded pixels and quantized aggregate/support hashes agree across worker counts.'
    checkpoint('complete')


if __name__=='__main__': main()
