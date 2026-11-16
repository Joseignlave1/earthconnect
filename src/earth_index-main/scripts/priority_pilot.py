# /// script
# requires-python = ">=3.11"
# dependencies = ["rasterio==1.5.2", "numpy==2.5.4", "exactextract==0.3.0", "shapely==2.2.0", "pyproj==3.8.0", "h3==4.5.0", "pyhdf==0.11.7", "h5py==3.16.0"]
# ///
"""Build a bounded, source-tested 16-bit raster pilot around three analog examples.

uv run scripts/priority_pilot.py --credentials-file /private/earthdata-file
Source rasters are window-read or temporarily downloaded. Derived tiles persist.
"""
import argparse
import calendar
import csv
import hashlib
import io
import json
import math
import re
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, build_opener, urlopen

import h5py
import numpy as np
import rasterio
from exactextract import exact_extract
from exactextract.raster import NumPyRasterSource
from pyhdf.SD import SD, SDC
from pyproj import CRS, Transformer
from rasterio.transform import Affine, array_bounds, from_origin
from rasterio.windows import Window, from_bounds
from shapely.geometry import Polygon, mapping, shape

from prepare_nasa_subsets import NoRedirect
from process_native_temperature import CRS_WKT, grid_transform, original_token

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research/priority-pilot'
MONTHS='JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC'.split()
SITES=[('atacama','Atacama / northern Chile',-68.5,-23.5,910001),
       ('haughton','Haughton Crater',-89.82,75.42,910002),
       ('deathvalley','Death Valley / Badwater vicinity',-116.82,36.24,910003)]
PERMA_URL=('https://dap.ceda.ac.uk/neodc/esacci/permafrost/data/permafrost_extent/L4/area4/pp/v05.0/'
 'northern_hemisphere/2023/ESACCI-PERMAFROST-L4-PFR-MODISLST_CRYOGRID-AREA4_PP-2023-fv05.0.nc')
TOPO_BASE='https://hs.pangaea.de/model/GMTED/'
GLIM_URL='https://epic.awi.de/id/eprint/31092/1/hartmann-moosdorf_2012.zip'


def sha(data): return hashlib.sha256(data).hexdigest()


def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def get_json(url,path):
    if path.exists(): return json.loads(path.read_text())
    with urlopen(Request(url,headers={'User-Agent':'earth-index-priority-pilot/1'}),timeout=60) as r:
        obj=json.load(r)
    save(path,obj)
    return obj


def sites():
    result=[]
    for key,name,lon,lat,srid in SITES:
        crs=CRS.from_proj4(f'+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m')
        result.append({'key':key,'name':name,'center':[lon,lat],'srid':srid,
                       'crs_wkt':crs.to_wkt('WKT1_GDAL'),'crs_proj4':crs.to_string()})
    return result


def cells(site,target_crs,n=100):
    project=Transformer.from_crs(site['crs_wkt'],target_crs,always_xy=True)
    half=n*500
    fs=[]
    for r in range(n):
        for c in range(n):
            x,y=-half+c*1000,half-r*1000
            corners=[(x,y),(x+1000,y),(x+1000,y-1000),(x,y-1000),(x,y)]
            ring=[]
            for a,b in zip(corners,corners[1:]):
                ring.extend(project.transform(a[0]+(b[0]-a[0])*t/4,a[1]+(b[1]-a[1])*t/4) for t in range(4))
            fs.append({'type':'Feature','properties':{'i':r*n+c},'geometry':mapping(Polygon(ring))})
    return fs


def bounds_of(fs):
    bs=np.array([shape(f['geometry']).bounds for f in fs])
    return (bs[:,0].min(),bs[:,1].min(),bs[:,2].max(),bs[:,3].max())


def read_window(src,bounds,pad=1):
    f=from_bounds(*bounds,src.transform)
    x0=max(0,math.floor(f.col_off)-pad);y0=max(0,math.floor(f.row_off)-pad)
    x1=min(src.width,math.ceil(f.col_off+f.width)+pad);y1=min(src.height,math.ceil(f.row_off+f.height)+pad)
    if x1<=x0 or y1<=y0: return None,None,None
    w=Window(x0,y0,x1-x0,y1-y0)
    a=src.read(1,window=w,masked=True).astype('float64').filled(np.nan)
    a[~np.isfinite(a)]=np.nan
    return a,src.window_transform(w),[x0,y0,x1-x0,y1-y0]


def aggregate(a,tr,crs,fs,geographic=False):
    n=math.isqrt(len(fs));weight='area_spherical_m2' if geographic else 'area_cartesian'
    source=NumPyRasterSource(a,*array_bounds(*a.shape,tr),nodata=np.nan,srs_wkt=crs)
    rows=exact_extract(source,fs,[f'mean(coverage_weight={weight})',f'valid=count(coverage_weight={weight})',
                                  f'total=count(default_value=0,coverage_weight={weight})'],include_cols=['i'])
    mean=np.full(n*n,np.nan);valid=np.zeros(n*n);total=np.zeros(n*n)
    for r in rows:
        p=r['properties'];i=p['i'];valid[i]=p['valid'];total[i]=p['total']
        if p['valid']>0:mean[i]=p['mean']
    return mean.reshape(n,n),valid.reshape(n,n),total.reshape(n,n)


def quantize(a,step,offset):
    valid=np.isfinite(a);codes=np.zeros(a.shape,dtype='<u2')
    q=np.rint((a[valid]-offset)/step)
    if np.any(q<1) or np.any(q>65535): raise ValueError('Quantization overflow or NODATA collision')
    codes[valid]=q.astype('<u2')
    error=float(np.max(np.abs(codes[valid].astype(float)*step+offset-a[valid]))) if valid.any() else 0
    if error>step/2+1e-8:raise ValueError('Quantization error bound exceeded')
    return codes,error


def write_layer(site,key,a,unit,step,offset,source,variable_key):
    if a.shape!=(100,100):raise ValueError('Expected 100x100 tile')
    codes,error=quantize(a,step,offset)
    path=OUT/'tiles'/f"{site['key']}--{key}.tif";path.parent.mkdir(parents=True,exist_ok=True)
    with rasterio.open(path,'w',driver='GTiff',height=100,width=100,count=1,dtype='uint16',
                       nodata=0,crs=site['crs_wkt'],transform=from_origin(-50000,50000,1000,1000)) as dst:
        dst.write(codes,1)
    return {'site_key':site['key'],'feature_key':key,'variable_key':variable_key,'unit':unit,
            'scale':step,'offset':offset,'nodata':0,'path':str(path.relative_to(ROOT)),
            'file_sha256':sha(path.read_bytes()),'codes_sha256':sha(codes.tobytes()),
            'valid_cells':int(np.isfinite(a).sum()),'max_rounding_error':error,
            'physical_min':float(np.nanmin(a)) if np.isfinite(a).any() else None,
            'physical_max':float(np.nanmax(a)) if np.isfinite(a).any() else None,
            'source':source}


def climate(site,fs):
    lon,lat=site['center'];dx=2.2 if site['key']=='haughton' else 1.1
    layers=[]
    for parameter in ('T2M','PRECTOTCORR'):
        url='https://power.larc.nasa.gov/api/temporal/climatology/regional?'+urlencode({
            'parameters':parameter,'community':'RE','longitude-min':round(lon-dx,4),'longitude-max':round(lon+dx,4),
            'latitude-min':round(lat-1.1,4),'latitude-max':round(lat+1.1,4),'format':'JSON'})
        path=OUT/'sources'/f"power-{site['key']}-{parameter}.json"
        obj=get_json(url,path)
        unit=obj['parameters'][parameter]['units']
        if unit!=('C' if parameter=='T2M' else 'mm/day'):raise ValueError('Unexpected POWER unit')
        xs=sorted(set(f['geometry']['coordinates'][0] for f in obj['features']))
        ys=sorted(set(f['geometry']['coordinates'][1] for f in obj['features']),reverse=True)
        if not np.allclose(np.diff(xs),.625) or not np.allclose(np.diff(ys),-.5):raise ValueError('Unexpected POWER grid')
        tr=from_origin(xs[0]-.3125,ys[0]+.25,.625,.5)
        fill=obj['header']['fill_value']
        start=int(obj['header']['start'][:4]);end=int(obj['header']['end'][:4])
        year_days=sum(366 if calendar.isleap(y) else 365 for y in range(start,end+1))/(end-start+1)
        outputs={'coldest_month_air_c':None,'warmest_month_air_c':None} if parameter=='T2M' else {'annual_precip_mm':None}
        for key in outputs:
            a=np.full((len(ys),len(xs)),np.nan)
            for f in obj['features']:
                x,y,_=f['geometry']['coordinates'];p=f['properties']['parameter'][parameter]
                monthly=np.array([p[m] for m in MONTHS],dtype=float)
                if np.any(monthly==fill) or not np.all(np.isfinite(monthly)):continue
                v=min(monthly) if key.startswith('coldest') else max(monthly)
                if parameter=='PRECTOTCORR':
                    if p['ANN']==fill or p['ANN']<0:continue
                    v=p['ANN']*year_days
                a[ys.index(y),xs.index(x)]=v
            mean,valid,total=aggregate(a,tr,rasterio.crs.CRS.from_epsg(4326).to_wkt(),fs,True)
            coverage=np.divide(valid,total,out=np.zeros_like(valid),where=total>0)
            if coverage.min()<.999:raise ValueError('POWER grid does not fully cover pilot')
            if key=='annual_precip_mm':unit2,step,offset,var='mm/year',.1,-.1,'weather.precipitation'
            else:unit2,step,offset,var='degree_Celsius',.01,-100.01,'weather.air_temperature_2m'
            source={'dataset':'NASA_POWER_MERRA2','url':url,'response_path':str(path.relative_to(ROOT)),
                'response_sha256':sha(path.read_bytes()),'header':obj['header'],'native_grid_degrees':[.625,.5],
                'method':'Source-grid cell supports, fractional spherical-area overlap onto target cells; no fine-scale climate downscaling.',
                'statistic':key,'annual_precip_days_per_climatological_year':year_days if parameter=='PRECTOTCORR' else None,
                'note':'Coldest/warmest values are extrema of monthly MEANS, not daily extreme temperatures. Annual rain = ANN mean daily rate times mean year length.'}
            layers.append(write_layer(site,key,mean,unit2,step,offset,source,var))
    return layers


def nasa_download(title,token):
    if not re.fullmatch(r'MCD12Q1\.A2024001\.h\d{2}v\d{2}\.061\.\d+',title):raise ValueError('Unexpected land-cover granule')
    canonical=f'https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MCD12Q1.061/{title}/{title}.hdf'
    url=canonical;auth=token;opener=build_opener(NoRedirect)
    for _ in range(5):
        try:
            req=Request(url,headers={'User-Agent':'earth-index-priority-pilot/1',**({'Authorization':'Bearer '+auth} if auth else {})})
            with opener.open(req,timeout=90) as r:
                b=r.read(100*1024*1024+1)
            if len(b)>100*1024*1024 or b[:4]!=b'\x0e\x03\x13\x01':raise ValueError('Unexpected HDF4 payload')
            return b,{'title':title,'uri':canonical,'sha256':sha(b),'bytes':len(b)}
        except HTTPError as e:
            if e.code not in (301,302,303,307,308):raise RuntimeError(f'Land-cover download failed (HTTP {e.code}); credentials/redirect omitted') from None
            target=e.headers['Location'];host=urlparse(target).hostname or ''
            if urlparse(target).scheme!='https' or not (host.endswith('.amazonaws.com') or host=='d1nklfio7vscoe.cloudfront.net'):
                raise RuntimeError('Unexpected NASA download redirect host') from None
            url=target;auth=''
    raise RuntimeError('Too many NASA redirects')


def landcover(site,fs,token):
    bbox=bounds_of(fs)
    url='https://cmr.earthdata.nasa.gov/search/granules.json?'+urlencode({'short_name':'MCD12Q1','version':'061',
        'temporal':'2024-01-01T00:00:00Z,2024-01-02T00:00:00Z','bounding_box':','.join(map(str,bbox)),'page_size':100})
    path=OUT/'sources'/f"landcover-cmr-{site['key']}.json";obj=get_json(url,path)
    projected=cells(site,CRS_WKT);pb=bounds_of(projected)
    target_area=np.array([shape(f['geometry']).area for f in projected]).reshape(100,100)
    sums={k:np.zeros((100,100)) for k in ('bare_fraction','permanent_snow_ice_fraction')}
    valid_area=np.zeros((100,100));sources=[]
    for g in sorted(obj['feed']['entry'],key=lambda x:x['title']):
        payload,meta=nasa_download(g['title'],token)
        with tempfile.TemporaryDirectory(prefix='earth-index-landcover-') as tmp:
            p=Path(tmp)/'source.hdf';p.write_bytes(payload);del payload
            h=SD(str(p),SDC.READ)
            try:
                grid=grid_transform(h.attributes()['StructMetadata.0'],(2400,2400))
                f=from_bounds(*pb,grid)
                x0=max(0,math.floor(f.col_off));y0=max(0,math.floor(f.row_off))
                x1=min(2400,math.ceil(f.col_off+f.width));y1=min(2400,math.ceil(f.row_off+f.height))
                if x1<=x0 or y1<=y0:continue
                decoded=[]
                for name in ('LC_Type1','QC'):
                    ds=h.select(name)
                    try:decoded.append(ds.get(start=(y0,x0),count=(y1-y0,x1-x0)))
                    finally:ds.endaccess()
                lc,qc=decoded;tr=grid*Affine.translation(x0,y0)
                good=np.isin(qc,[0,2])&(lc>=1)&(lc<=17)
                meta.update(read_window=[x0,y0,x1-x0,y1-y0],qc_counts={str(int(v)):int((qc==v).sum()) for v in np.unique(qc)},
                    decoded_sha256=sha(lc.tobytes()+qc.tobytes()))
                for key,category in [('bare_fraction',16),('permanent_snow_ice_fraction',15)]:
                    a=np.where(good,(lc==category).astype(float),np.nan)
                    mean,valid,_=aggregate(a,tr,CRS_WKT,projected)
                    sums[key]+=np.nan_to_num(mean)*valid
                    if key=='bare_fraction':valid_area+=valid
                sources.append(meta)
            finally:h.end()
    fraction=valid_area/target_area
    if fraction.max()>1.0001:raise ValueError('Duplicate or overlapping land-cover input')
    source={'dataset':'MCD12Q1.061','year':2024,'files':sources,'native_pixel_side_m':463.3127165,
            'qa':'Only QC values 0 (classified land) and 2 (classified water); exclude unclassified land even when labelled barren.',
            'method':'Fractional overlap with source sinusoidal pixels; fractions of accepted classified support, coverage supplied separately.',
            'note':'Permanent snow/ice CLASS fraction is not seasonal snow persistence.'}
    layers=[]
    for key,total in sums.items():
        a=np.divide(total,valid_area,out=np.full_like(total,np.nan),where=valid_area>0)
        layers.append(write_layer(site,key,a,'1',.001,-.001,source,'land_cover.igbp_area_fraction'))
    layers.append(write_layer(site,'landcover_valid_fraction',np.clip(fraction,0,1),'1',.001,-.001,source,'quality.valid_area_fraction'))
    return layers


def terrain(site):
    fs=cells(site,'EPSG:4326',120);out=[];height=None;source_records={}
    sources=[('elevation_median_m','elevation/elevation_1KMmd_GMTEDmd.tif','m',1,-1001,'terrain.elevation'),
             ('slope_mean_deg','slope/slope_1KMmn_GMTEDmd.tif','degree',.1,-.1,'terrain.slope')]
    for key,file,unit,step,offset,var in sources:
        url=TOPO_BASE+file
        with rasterio.open(url) as src:
            a,tr,w=read_window(src,bounds_of(fs))
            if src.crs.to_epsg()!=4326:raise ValueError('Unexpected terrain CRS')
            mean,valid,total=aggregate(a,tr,src.crs.to_wkt(),fs,True)
            coverage=np.divide(valid,total,out=np.zeros_like(valid),where=total>0)
            source={'dataset':'EarthEnv_topography_GMTED','uri':url,'citation':'https://doi.org/10.1594/PANGAEA.867113',
                'native_grid_degrees':[abs(src.transform.a),abs(src.transform.e)],'window':w,
                'decoded_window_sha256':sha(a.tobytes()),'tags':src.tags(),
                'method':'Publication-defined statistic, fractional spherical-area-weighted overlap onto target grid.',
                'warnings':['Elevation TIFF description says Maximum despite published median filename; preserved as provider metadata conflict.'] if key.startswith('elevation') else []}
            out.append(write_layer(site,key,mean[10:110,10:110],unit,step,offset,source,var))
            source_records[key]=source
            if key.startswith('elevation'):height=mean
            else:out.append(write_layer(site,'terrain_valid_fraction',coverage[10:110,10:110],'1',.001,-.001,source,'quality.valid_area_fraction'))
    for width in (5,21):
        radius=width//2
        windows=np.lib.stride_tricks.sliding_window_view(height,(width,width))[10-radius:110-radius,10-radius:110-radius]
        sufficient=np.isfinite(windows).sum(axis=(-1,-2))>=width*width*.9
        with np.errstate(invalid='ignore'):
            relief=np.nanmax(windows,axis=(-1,-2))-np.nanmin(windows,axis=(-1,-2))
            tpi=height[10:110,10:110]-np.nanmean(windows,axis=(-1,-2))
        source={**source_records['elevation_median_m'],'method':f'{width}x{width} neighbourhood on 1km derived elevation; halo read before clipping; require >=90% usable neighbours.',
                'note':'These are coarse-grid contextual descriptors, not native-pixel roughness or proof of an impact crater.'}
        out.append(write_layer(site,f'relief_{width}km_m',np.where(sufficient,relief,np.nan),'m',1,-1,source,'terrain.local_relief'))
        out.append(write_layer(site,f'tpi_{width}km_m',np.where(sufficient,tpi,np.nan),'m',.25,-5000.25,source,'terrain.topographic_position'))
    return out


def permafrost(site,fs,path):
    with h5py.File(path,'r') as f:
        ds=f['PFR'];lats=f['lat'][:];lons=f['lon'][:]
        if ds.shape!=(1,len(lats),len(lons)) or not np.all(np.diff(lats)>0) or not np.all(np.diff(lons)>0):raise ValueError('Unexpected CCI dimensions')
        fill=int(np.asarray(ds.attrs['_FillValue']).item())
        if fill!=255 or ds.attrs['units']!=b'percent':raise ValueError('Unexpected CCI encoding')
        xmin,ymin,xmax,ymax=bounds_of(fs)
        rows=np.flatnonzero((lats>=ymin-.02)&(lats<=ymax+.02));cols=np.flatnonzero((lons>=xmin-.02)&(lons<=xmax+.02))
        source={'dataset':'ESA_CCI_PERMAFROST_v5','url':PERMA_URL,'sha256':sha(path.read_bytes()),
            'year':2023,'native_grid_degrees':[.01,.01],'source_variable':'PFR','source_fill':255,
            'coordinate_domain_lat':[float(lats[0]),float(lats[-1])],
            'warnings':['File GeoTransform conflicts with CF coordinates; coordinate vectors define placement and row order.'],
            'method':'CF latitude/longitude coordinates; flip ascending latitude to north-up; percent/100, then area-weighted overlap. Modeled fraction, not an ice observation.'}
        if not len(rows) or not len(cols):
            a=np.full((100,100),np.nan);coverage=np.zeros_like(a);source['missing_reason']='outside_source_coordinate_domain'
        else:
            raw=ds[0,rows[0]:rows[-1]+1,cols[0]:cols[-1]+1][::-1,:]
            values=np.where((raw!=fill)&(raw<=100),raw.astype(float)/100,np.nan)
            # Nominal spacing is .01 degrees; axis floats are rounded float32.
            tr=from_origin(-180+int(cols[0])*.01,25+(int(rows[-1])+1)*.01,.01,.01)
            a,valid,total=aggregate(values,tr,rasterio.crs.CRS.from_epsg(4326).to_wkt(),fs,True)
            coverage=np.divide(valid,total,out=np.zeros_like(valid),where=total>0)
            source.update(window=[int(cols[0]),int(rows[0]),len(cols),len(rows)],decoded_window_sha256=sha(raw.tobytes()))
            if not np.isfinite(a).any():source['missing_reason']='no_valid_source_PFR_values_in_region'
    return [write_layer(site,'permafrost_fraction',a,'1',.001,-.001,source,'ground.permafrost_fraction'),
            write_layer(site,'permafrost_valid_fraction',coverage,'1',.001,-.001,source,'quality.valid_area_fraction')]


def lithology(site,payload):
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        text=archive.read('glim_wgs84_0point5deg.txt.asc').decode()
        dictionary={int(r['Value_']):r['xx'] for r in csv.DictReader(
            io.StringIO(archive.read('Classnames.txt').decode()),delimiter=';')}
    lines=text.splitlines()
    header={k.lower():float(v) for k,v in (line.split() for line in lines[:6])}
    if header!={'ncols':720,'nrows':360,'xllcorner':-180,'yllcorner':-90,'cellsize':.5,'nodata_value':-9999}:
        raise ValueError('Unexpected GLiM grid')
    grid=np.loadtxt(io.StringIO('\n'.join(lines[6:])),dtype=int)
    if grid.shape!=(360,720) or not set(np.unique(grid)).issubset({-9999,*range(1,17)}):
        raise ValueError('Unexpected GLiM classes')
    yy,xx=np.indices((100,100))
    project=Transformer.from_crs(site['crs_wkt'],'EPSG:4326',always_xy=True)
    lon,lat=project.transform(-50000+(xx+.5)*1000,50000-(yy+.5)*1000)
    values=grid[np.floor((90-lat)/.5).astype(int),np.floor((lon+180)/.5).astype(int)]
    a=np.where(np.isin(values,[-9999,15]),np.nan,values.astype(float))
    source={'dataset':'GLiM_v1.0_0.5_degree','url':GLIM_URL,'sha256':sha(payload),
        'citation':'https://doi.org/10.1594/PANGAEA.788537','license':'CC BY 3.0',
        'native_grid_degrees':[.5,.5],'class_dictionary':dictionary,'missing_classes':[-9999,15],
        'method':'Containing source cell at target pixel center; categorical codes are never averaged.',
        'note':'Coarse dominant lithology context, not 1km geology, local basalt detection or detailed mineralogy. Codes 11 and 16 denote water and ice/glaciers.'}
    return [write_layer(site,'dominant_lithology_code',a,'category_code',1,0,source,'geology.dominant_lithology')]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--credentials-file',type=Path,required=True)
    parser.add_argument('--permafrost-file',type=Path,default=Path('/tmp/priority-permafrost-2023.nc'))
    args=parser.parse_args();start=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
    if not args.permafrost_file.exists():
        with urlopen(PERMA_URL,timeout=90) as r:args.permafrost_file.write_bytes(r.read())
    token=original_token(args.credentials_file)
    with urlopen(GLIM_URL,timeout=45) as response:glim=response.read(1024*1024+1)
    if len(glim)>1024*1024:raise ValueError('Unexpected GLiM archive size')
    report={'status':'research_pilot','generated_at':datetime.now(timezone.utc).isoformat(),'script_sha256':sha(Path(__file__).read_bytes()),
            'sites':sites(),'layers':[],'source_raster_files_retained':0,
            'pending':['Seasonal snow persistence (annual permanent snow/ice class is available instead).',
                       'Detailed geology/mineralogy, dune morphology and validated crater recognition.'],
            'validation_scope':'Source access, quantized raster roundtrip and exploratory environmental profiles; not a global analog-discovery validation.'}
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif',
                      GDAL_HTTP_TIMEOUT='45',GDAL_HTTP_MAX_RETRY='2',GDAL_HTTP_RETRY_DELAY='1',GDAL_CACHEMAX=128*1024*1024):
        for site in report['sites']:
            print(site['key']+': terrain and broad landform context',flush=True)
            report['layers']+=terrain(site)
            fs=cells(site,'EPSG:4326')
            print(site['key']+': source-native climate',flush=True)
            report['layers']+=climate(site,fs)
            print(site['key']+': classified bare ground and snow/ice',flush=True)
            report['layers']+=landcover(site,fs,token)
            print(site['key']+': modeled permafrost',flush=True)
            report['layers']+=permafrost(site,fs,args.permafrost_file)
            print(site['key']+': coarse lithology context',flush=True)
            report['layers']+=lithology(site,glim)
            save(OUT/'progress.json',{'completed_site':site['key'],'layers':len(report['layers'])})
    report['wall_seconds']=time.monotonic()-start
    save(OUT/'manifest.json',report)
    print(f"Prepared {len(report['layers'])} aligned layers in {report['wall_seconds']:.1f}s",flush=True)


if __name__=='__main__':main()
