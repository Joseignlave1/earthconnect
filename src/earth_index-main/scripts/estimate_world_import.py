# /// script
# requires-python = ">=3.11"
# dependencies = ["rasterio==1.5.2", "numpy==2.5.4", "exactextract==0.3.0", "shapely==2.2.0", "pyproj==3.8.0", "h3==4.5.0", "pyhdf==0.11.7", "h5py==3.16.0"]
# ///
"""Metadata-only global source sizing; never downloads source rasters/imports data."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
from urllib.parse import urlencode
import thermal_pilot as t
from benchmark_import_pipeline import HttpPool, save

ROOT=Path(__file__).resolve().parents[1]


def main():
    old=json.loads((ROOT/'research/import-profile/volume.json').read_text())
    known={a['href']:a['bytes'] for s in old['samples'] for a in s['assets']}
    pool=HttpPool();signer=t.Signer();samples=[]
    for date in ('2023-01-25','2023-07-12'):
        url=t.STAC+'/search?'+urlencode(dict(collections='modis-11A2-061',
            datetime=f'{date}T00:00:00Z/{date}T23:59:59Z',query=json.dumps({'platform':{'eq':'terra'}}),limit=1000))
        items=[]
        while url:
            data=t.get_json(url);items.extend(data['features'])
            url=next((link['href'] for link in data.get('links',[]) if link['rel']=='next'),None)
            if url and not url.startswith(t.STAC+'/'):raise ValueError('Unexpected catalog pagination host')
        assets=[]
        for item in items:
            for band in t.BANDS:
                href=item['assets'][band]['href']
                assets.append(dict(item=item['id'],band=band,href=href,signed=None if href in known else signer.sign(href)))
        def head(asset):
            href=asset['href']
            if href in known:size=known[href]
            else:
                _,headers,_=pool.fetch(asset['signed'],method='HEAD')
                size=int(next(v for k,v in headers.items() if k.lower()=='content-length'))
            return {k:v for k,v in asset.items() if k!='signed'}|dict(bytes=size)
        with ThreadPoolExecutor(max_workers=8) as workers:measured=list(workers.map(head,assets))
        samples.append(dict(date=date,items=len(items),files=len(measured),bytes=sum(a['bytes'] for a in measured),assets=measured))
        print(json.dumps({k:v for k,v in samples[-1].items() if k!='assets'}),flush=True)
    pool.close()
    url='https://cmr.earthdata.nasa.gov/search/granules.json?'+urlencode(dict(short_name='MCD12Q1',version='061',
        temporal='2024-01-01T00:00:00Z,2024-01-02T00:00:00Z',page_size=1000))
    cover=t.get_json(url)['feed']['entry']
    if len(cover)>=1000:raise RuntimeError('Land-cover pagination required')
    report=dict(generated_at=datetime.now(timezone.utc).isoformat(),thermal_samples=samples,
        thermal_annual_bytes_at_44_dates_estimate=sum(s['bytes'] for s in samples)/len(samples)*44,
        thermal_annual_files_at_44_dates_estimate=sum(s['files'] for s in samples)/len(samples)*44,
        landcover=dict(year=2024,granules=len(cover),bytes_from_cmr_sizes=sum(float(g.get('granule_size',0))*1024**2 for g in cover),
                       missing_size_count=sum('granule_size' not in g for g in cover),metadata_url=url),
        caveat='Thermal estimate extrapolates two global seasonal dates to 44 composites, not an all-year inventory. CMR land-cover sizes are MiB, checked against the previously downloaded Atacama file. Global source footprints do not imply valid data everywhere.')
    save(ROOT/'research/import-profile/world-volume.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='thermal_samples'}),flush=True)


if __name__=='__main__':main()
