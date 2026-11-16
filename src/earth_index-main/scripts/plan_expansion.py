# /// script
# requires-python = ">=3.11"
# dependencies = ["rasterio==1.5.2", "numpy==2.5.4", "exactextract==0.3.0", "shapely==2.2.0", "pyproj==3.8.0", "h3==4.5.0", "pyhdf==0.11.7", "h5py==3.16.0"]
# ///
"""Declare a diverse 4km expansion and forecast before downloading its rasters."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import re
import statistics
from pyproj import CRS,Transformer
from shapely.geometry import Polygon,box
import thermal_pilot as t
from raster_source_cache import atomic_json

ROOT=Path(__file__).resolve().parents[1]
EXTRAS=[('algeria_coast',3,36.5),('mali',-3,20),('sahel',10,12),('ethiopia',40,10),
 ('kalahari',23,-24),('karoo',23,-32),('madagascar',47,-19),('oman',56,21),('iran',58,31),
 ('thar',71,27),('tibetan_plateau',92,34),('taklamakan',83,39),('mongolia_steppe',111,46),
 ('siberia',125,60),('kamchatka',158,56),('chukotka',170,65),('central_canada',-105,60),
 ('alaska',-148,65),('eastern_canada',-65,55),('great_basin',-117,40),('arizona',-110,33),
 ('mexico',-104,25),('yucatan',-89,20),('cuba',-78,22),('andes_peru',-75,-12),
 ('atacama_south',-70,-29),('patagonia',-69,-46),('brazil_cerrado',-48,-14),
 ('brazil_pantanal',-57,-18),('southeast_brazil',-44,-23),('western_australia',120,-24),
 ('queensland',145,-22),('tropical_australia',133,-14),('new_zealand',170,-44),
 ('south_china',110,26),('central_india',78,20),('philippines',123,12),('sumatra',101,0),
 ('spain',-4,42),('scotland',-4,57),('anatolia',34,39),('ukraine',32,49),('kazakhstan',67,46)]


def boundary(site,target_crs,side_km=400):
    project=Transformer.from_crs(site['crs_wkt'],target_crs,always_xy=True)
    half=side_km*500;corners=[(-half,half),(half,half),(half,-half),(-half,-half),(-half,half)]
    points=[]
    for a,b in zip(corners,corners[1:]):
        points.extend(project.transform(a[0]+(b[0]-a[0])*i/400,a[1]+(b[1]-a[1])*i/400) for i in range(400))
    return Polygon(points)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,default=ROOT/'research/expansion-4km')
    ap.add_argument('--target-minutes',type=float,default=30)
    args=ap.parse_args();output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    if (output/'plan.json').exists():raise ValueError('Plan already frozen; choose a new directory')
    base=json.loads((ROOT/'research/thermal-pilot/manifest.json').read_text())
    candidates=list(base['sites'])
    for index,(key,lon,lat) in enumerate(EXTRAS):
        crs=CRS.from_proj4(f'+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m')
        candidates.append(dict(key=key,name=key.replace('_',' '),center=[lon,lat],srid=920000+index,
            crs_wkt=crs.to_wkt('WKT1_GDAL'),crs_proj4=crs.to_string(),role='expansion_control',plausible_profiles=[]))
    world=json.loads((ROOT/'research/import-profile/world-volume.json').read_text())
    sizes={}
    for sample in world['thermal_samples']:
        per_tile={}
        for asset in sample['assets']:
            tile=re.search(r'\.(h\d{2}v\d{2})\.',asset['item'])[1]
            per_tile[tile]=per_tile.get(tile,0)+asset['bytes']
        for tile,size in per_tile.items():sizes.setdefault(tile,[]).append(size)
    cmr=t.get_json('https://cmr.earthdata.nasa.gov/search/granules.json?short_name=MCD12Q1&version=061&temporal=2024-01-01T00%3A00%3A00Z,2024-01-02T00%3A00%3A00Z&page_size=1000')
    cover={re.search(r'\.(h\d{2}v\d{2})\.',g['title'])[1]:g for g in cmr['feed']['entry']}
    annual={tile:statistics.mean(values)*44 for tile,values in sizes.items()}
    bytes_per_second=8456401.34988292;span=1111950.5196667
    selected=[];native=set();history=[]
    for site in candidates:
        geometry=boundary(site,t.p.CRS_WKT);tiles=[]
        for h in range(36):
            for v in range(18):
                tile=f'h{h:02}v{v:02}';x=-20015109.354+h*span;y=10007554.677-v*span
                if geometry.intersection(box(x,y-span,x+span,y)).area>1:
                    tiles.append(tile)
        selected.append(site|dict(native_tiles=tiles));native.update(tiles)
        thermal=sum(annual.get(tile,0) for tile in native)
        land=sum(float(cover[tile]['granule_size'])*1024**2 for tile in native if tile in cover)
        payload=thermal+land+1198050360+12448396+38670
        seconds=payload/bytes_per_second*1.15+len(selected)*(1.644*16/6.86+.667+.5)+120
        history.append(dict(regions=len(selected),native_tiles=len(native),estimated_bytes=payload,predicted_seconds=seconds,
                            thermal_estimated_bytes=thermal,landcover_estimated_bytes=land))
        if len(selected)>=len(base['sites']) and seconds>=args.target_minutes*60:break
    # Keep the closest complete predeclared prefix, retaining all original controls.
    index=min(range(len(base['sites'])-1,len(history)),key=lambda i:abs(history[i]['predicted_seconds']-args.target_minutes*60))
    selected=selected[:index+1];forecast=history[index];native=set(t for s in selected for t in s['native_tiles'])
    plan=dict(generated_at=datetime.now(timezone.utc).isoformat(),status='frozen_before_raster_downloads',
        cell_m=4000,side_km=400,shape=[100,100],variables=25,workers=8,download_workers=8,
        sites=selected,forecast=forecast,nominal_cells=len(selected)*10000,nominal_area_km2=len(selected)*160000,
        forecast_basis='Two global seasonal asset-size samples extrapolated to 44 composites; measured 8.46 MB/s; 15% transfer allowance; area-scaled base CPU plus thermal and 120s fixed load/check allowance.',
        scope='Purposive diverse enlarged patches and extra controls, not statistically random or complete global coverage. Nominal areas can overlap; report union separately.',
        validation=['All 25 U16 bands, valid quantization and support/missingness semantics.',
                    'Preserve physical 5/21km terrain contexts using a 1km intermediate grid.',
                    'Compare old/new shared coverage and unchanged live frontend target/query logic.',
                    'Check reference/plausible/negative regions and same-cell co-location adversarial fixtures.',
                    'Load as separate datasource and keep current demo default.'])
    atomic_json(output/'plan.json',plan)
    atomic_json(output/'landcover-catalog.json',[cover[tile] for tile in sorted(native) if tile in cover])
    atomic_json(output/'selection-forecast.json',history)
    print(json.dumps({k:v for k,v in plan.items() if k not in ('sites','validation')},indent=2))


if __name__=='__main__':main()
