# /// script
# requires-python = ">=3.11"
# dependencies = ["rasterio==1.5.2", "numpy==2.5.4", "exactextract==0.3.0", "shapely==2.2.0", "pyproj==3.8.0", "h3==4.5.0", "pyhdf==0.11.7", "h5py==3.16.0"]
# ///
"""Count source tiles and measure two seasonal dates' asset sizes via HEAD.

Corridors are planning geometries, not an import. Only metadata is downloaded.
"""
import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlencode
from pyproj import Transformer
from shapely.geometry import shape, box
from shapely.ops import transform, unary_union
import thermal_pilot as t
from benchmark_import_pipeline import HttpPool, save

ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan',type=Path,default=Path('/tmp/earth-corridor-plan.json'))
    args=ap.parse_args()
    plans=json.loads(args.plan.read_text())['plans']; output=[]
    forward=Transformer.from_crs(4326,t.p.CRS_WKT,always_xy=True).transform
    span=1111950.5196667
    for plan in plans:
        geometry=unary_union([transform(forward,shape(r['geometry'])) for r in plan['routes']])
        tiles=[]
        for h in range(36):
            for v in range(18):
                x=-20015109.354+h*span;y=10007554.677-v*span
                intersection=geometry.intersection(box(x,y-span,x+span,y))
                if intersection.area>1e6: tiles.append(f'h{h:02}v{v:02}')
        output.append(dict(width_km=plan['width_km'],land_area_km2=plan['land_area_km2'],
                           land_cells_4km=plan['land_cells_4km'],source_tiles=len(tiles),tiles=tiles))
    needed=set(output[-1]['tiles']);signer=t.Signer();pool=HttpPool();samples=[]
    for date in ('2023-01-25','2023-07-12'):
        url=t.STAC+'/search?'+urlencode(dict(collections='modis-11A2-061',datetime=f'{date}T00:00:00Z/{date}T23:59:59Z',
            query=json.dumps({'platform':{'eq':'terra'}}),limit=1000))
        items=[]
        while url:
            data=t.get_json(url);items.extend(data['features'])
            url=next((l['href'] for l in data.get('links',[]) if l['rel']=='next'),None)
            if url and not url.startswith(t.STAC+'/'):raise RuntimeError('Unexpected pagination host')
        assets=[]
        for item in items:
            match=re.search(r'\.((h\d{2}v\d{2}))\.',item['id'])
            if not match or match[1] not in needed:continue
            for band in t.BANDS:
                href=item['assets'][band]['href']
                assets.append(dict(tile=match[1],item=item['id'],band=band,href=href,signed=signer.sign(href)))
        def head(asset):
            _,headers,stats=pool.fetch(asset['signed'],method='HEAD')
            headers={k.lower():v for k,v in headers.items()}
            return {k:v for k,v in asset.items() if k!='signed'}|dict(bytes=int(headers['content-length']))
        with ThreadPoolExecutor(max_workers=8) as threads: measured=list(threads.map(head,assets))
        samples.append(dict(date=date,assets=measured))
        print(json.dumps(dict(date=date,assets=len(measured),bytes=sum(a['bytes'] for a in measured))),flush=True)
    pool.close()
    for plan in output:
        seasonal=[];missing=[]
        for sample in samples:
            assets=[a for a in sample['assets'] if a['tile'] in plan['tiles']]
            seasonal.append(dict(date=sample['date'],tiles=len(set(a['tile'] for a in assets)),bytes=sum(a['bytes'] for a in assets)))
            missing.extend(sorted(set(plan['tiles'])-set(a['tile'] for a in assets)))
        plan['seasonal_samples']=seasonal;plan['missing_sample_tiles']=sorted(set(missing))
        plan['annual_bytes_at_44_dates_estimate']=sum(s['bytes'] for s in seasonal)/len(seasonal)*44
    save(ROOT/'research/import-profile/volume.json',dict(plans=output,samples=samples,
         caveat='Estimate scales two seasonal dates to 44 available composites; not an annual inventory. Planned geometries are simplified; intersections under 1 km2 omitted. Missing tiles require explicit missing-data handling.'))


if __name__=='__main__':main()
