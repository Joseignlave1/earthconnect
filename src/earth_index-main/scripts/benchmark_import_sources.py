# /// script
# requires-python = ">=3.11"
# dependencies = ["rasterio==1.5.2", "numpy==2.5.4", "exactextract==0.3.0", "shapely==2.2.0", "pyproj==3.8.0", "h3==4.5.0", "pyhdf==0.11.7", "h5py==3.16.0"]
# ///
"""Profile unchanged terrain calculations and small ancillary source downloads.

No scientific outputs, source manifests, or databases are modified. Downloaded
rasters are temporary. Optional Earthdata credentials are read from a private
file and never included in the report or exception output.
"""
import argparse
import json
import time
import uuid
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import rasterio
from pyhdf.SD import SD, SDC
from rasterio.transform import Affine
from rasterio.windows import from_bounds

import priority_pilot as p
from benchmark_import_pipeline import ENV, Forwarder, HttpPool, save, digest

ROOT = Path(__file__).resolve().parents[1]


def download(url):
    start = time.perf_counter(); cpu = time.process_time()
    with urlopen(Request(url, headers={'User-Agent': 'EarthIndex/import-profile'}), timeout=90) as response:
        first = time.perf_counter(); payload = response.read()
    return payload, dict(seconds=time.perf_counter()-start, ttfb_seconds=first-start,
                         cpu_seconds=time.process_time()-cpu, bytes=len(payload), sha256=digest(payload))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--credentials-file', type=Path)
    ap.add_argument('--landcover-only', action='store_true')
    args = ap.parse_args()
    output = ROOT/'research/import-profile/other-sources.json'
    report = json.loads(output.read_text()) if args.landcover_only and output.exists() else {}
    if not args.landcover_only:
        assets = {key: {'signed': p.TOPO_BASE+file} for key, file in [
            ('elevation', 'elevation/elevation_1KMmd_GMTEDmd.tif'),
            ('slope', 'slope/slope_1KMmn_GMTEDmd.tif')]}
        pool = HttpPool(); proxy = Forwarder(assets, pool)
        report['terrain'] = []
        try:
            for site in p.sites():
                start = time.perf_counter(); cpu = time.process_time()
                fs = p.cells(site, 'EPSG:4326', 120); bbox = p.bounds_of(fs)
                record = dict(site=site['key'], geometry_seconds=time.perf_counter()-start, sources=[])
                run = uuid.uuid4().hex; height = None
                for key in assets:
                    before = time.perf_counter()
                    with rasterio.Env(**ENV), rasterio.open(proxy.url(run, key)) as src:
                        a, tr, window = p.read_window(src, bbox)
                        info = dict(shape=list(src.shape), block_shapes=src.block_shapes,
                                    compression=str(src.compression), decoded_sha256=digest(a.tobytes()), window=window)
                        read = time.perf_counter()-before; before = time.perf_counter()
                        mean, valid, total = p.aggregate(a, tr, src.crs.to_wkt(), fs, True)
                        info.update(key=key, read_seconds=read, aggregate_seconds=time.perf_counter()-before,
                                    mean_sha256=digest(mean.tobytes()))
                        if key == 'elevation': height = mean
                    record['sources'].append(info)
                before = time.perf_counter()
                for width in (5, 21):
                    radius = width//2
                    windows = np.lib.stride_tricks.sliding_window_view(height, (width,width))[10-radius:110-radius,10-radius:110-radius]
                    sufficient = np.isfinite(windows).sum(axis=(-1,-2)) >= width*width*.9
                    relief = np.nanmax(windows,axis=(-1,-2))-np.nanmin(windows,axis=(-1,-2))
                    tpi = height[10:110,10:110]-np.nanmean(windows,axis=(-1,-2))
                    # Exercise original validity and quantization without writing a layer.
                    p.quantize(np.where(sufficient, relief, np.nan), 1, -1)
                    p.quantize(np.where(sufficient, tpi, np.nan), .25, -5000.25)
                requests = [r for r in proxy.records if r['run'] == run]
                record.update(context_seconds=time.perf_counter()-before, wall_seconds=time.perf_counter()-start,
                              cpu_seconds=time.process_time()-cpu, http_bytes=sum(r.get('bytes',0) for r in requests),
                              requests=requests)
                report['terrain'].append(record); save(output, report)
                print(json.dumps({'terrain':site['key'],'wall_seconds':record['wall_seconds'],'bytes':record['http_bytes']}), flush=True)
        finally:
            proxy.stop(); pool.close()
        report['ancillary'] = []
        # Whole reusable globals, plus one real regional request for each climate parameter.
        urls = [('permafrost_global',p.PERMA_URL),('lithology_global',p.GLIM_URL)]
        manifest = json.loads((ROOT/'research/priority-pilot/manifest.json').read_text())
        for parameter, key in [('T2M','coldest_month_air_c'),('PRECTOTCORR','annual_precip_mm')]:
            layer = next(l for l in manifest['layers'] if l['site_key']=='atacama' and l['feature_key']==key)
            urls.append((parameter,layer['source']['url']))
        for name, url in urls:
            try:
                payload, stats = download(url); report['ancillary'].append({'name':name,'url':url,**stats})
                print(json.dumps({'ancillary':name,**stats}), flush=True)
            except Exception as error:
                report['ancillary'].append({'name':name,'error':type(error).__name__})
            save(output, report)
    if args.credentials_file:
        token = args.credentials_file.read_text().strip()
        site = p.sites()[0]
        catalog = json.loads((ROOT/'research/priority-pilot/sources/landcover-cmr-atacama.json').read_text())
        title = catalog['feed']['entry'][0]['title']
        start = time.perf_counter(); cpu = time.process_time()
        payload, metadata = p.nasa_download(title, token)
        result = dict(metadata, download_seconds=time.perf_counter()-start,
                      download_cpu_seconds=time.process_time()-cpu)
        import tempfile
        with tempfile.TemporaryDirectory(prefix='earth-import-bench-') as tmp:
            file = Path(tmp)/'source.hdf'; file.write_bytes(payload); del payload
            before = time.perf_counter(); fs = p.cells(site,p.CRS_WKT)
            result['geometry_seconds'] = time.perf_counter()-before
            before = time.perf_counter(); h = SD(str(file),SDC.READ)
            try:
                grid = p.grid_transform(h.attributes()['StructMetadata.0'],(2400,2400))
                w = from_bounds(*p.bounds_of(fs),grid)
                x0=max(0,int(np.floor(w.col_off))); y0=max(0,int(np.floor(w.row_off)))
                x1=min(2400,int(np.ceil(w.col_off+w.width))); y1=min(2400,int(np.ceil(w.row_off+w.height)))
                arrays=[]
                for name in ('LC_Type1','QC'):
                    ds=h.select(name)
                    try: arrays.append(ds.get(start=(y0,x0),count=(y1-y0,x1-x0)))
                    finally: ds.endaccess()
                lc,qc=arrays; good=np.isin(qc,[0,2])&(lc>=1)&(lc<=17)
                result['decode_seconds']=time.perf_counter()-before
                result['decoded_sha256']=digest(lc.tobytes()+qc.tobytes())
                before=time.perf_counter()
                for category in (16,15):
                    p.aggregate(np.where(good,(lc==category).astype(float),np.nan),
                                grid*Affine.translation(x0,y0),p.CRS_WKT,fs)
                result['aggregate_seconds']=time.perf_counter()-before
            finally: h.end()
        report['landcover']=result; save(output, report)
        print(json.dumps({'landcover':title,**result}),flush=True)


if __name__ == '__main__': main()
