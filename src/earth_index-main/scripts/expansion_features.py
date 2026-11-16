"""Source-native base features for the timed 4 km experiment.

Terrain context is computed at 1 km first, then averaged over each 4 km cell.
Source validity remains separate from physical values. No missing region is filled.
"""
import calendar
import csv
import io
import json
import math
from pathlib import Path
import warnings
import zipfile

import h5py
import numpy as np
from pyhdf.SD import SD, SDC
from pyproj import Transformer
import rasterio
from rasterio.transform import Affine, from_origin
from rasterio.windows import Window, from_bounds
from shapely.geometry import shape

import priority_pilot as p
import thermal_import as t
from raster_overlap import OverlapPlan

GEO=rasterio.crs.CRS.from_epsg(4326).to_wkt()


def reduce4(values, weights=None, minimum=0.):
    """Conservative aligned 4x4 reduction; optional accepted-source area weights."""
    n,m=values.shape
    if n%4 or m%4:raise ValueError('Unaligned fine grid')
    v=values.reshape(n//4,4,m//4,4)
    good=np.isfinite(v)
    w=np.ones_like(v) if weights is None else weights.reshape(v.shape)
    w=np.where(good,w,0.)
    total=w.sum(axis=(1,3));sums=(np.where(good,v,0)*w).sum(axis=(1,3))
    result=np.divide(sums,total,out=np.full(total.shape,np.nan),where=total>0)
    result[good.sum(axis=(1,3))<16*minimum]=np.nan
    return result


def contexts(height, side):
    """Block rolling reductions keep 21x21 temporary memory bounded per worker."""
    output={}
    for width in (5,21):
        radius=width//2
        relief=np.full((side,side),np.nan);tpi=np.full_like(relief,np.nan)
        windows=np.lib.stride_tricks.sliding_window_view(height,(width,width))
        for row in range(0,side,80):
            for col in range(0,side,80):
                nr=min(80,side-row);nc=min(80,side-col)
                w=windows[10-radius+row:10-radius+row+nr,10-radius+col:10-radius+col+nc]
                good=np.isfinite(w).sum(axis=(-1,-2))>=.9*width*width
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore',RuntimeWarning)
                    r=np.nanmax(w,axis=(-1,-2))-np.nanmin(w,axis=(-1,-2))
                    v=height[10+row:10+row+nr,10+col:10+col+nc]-np.nanmean(w,axis=(-1,-2))
                relief[row:row+nr,col:col+nc]=np.where(good,r,np.nan)
                tpi[row:row+nr,col:col+nc]=np.where(good,v,np.nan)
        output[f'relief_{width}km_m']=reduce4(relief,minimum=.5)
        output[f'tpi_{width}km_m']=reduce4(tpi,minimum=.5)
    return output


def padded_window(src,bounds):
    f=from_bounds(*bounds,src.transform)
    x0=math.floor(f.col_off)-1;y0=math.floor(f.row_off)-1
    x1=math.ceil(f.col_off+f.width)+1;y1=math.ceil(f.row_off+f.height)+1
    w=Window(x0,y0,x1-x0,y1-y0)
    a=src.read(1,window=w,boundless=True,masked=True).astype(float).filled(np.nan)
    return a,src.window_transform(w),[x0,y0,x1-x0,y1-y0]


def terrain(site,cache,side_km=400):
    features,_=t.grid_features(site,cell_m=1000,side_km=side_km+20,target_crs=GEO)
    bounds=p.bounds_of(features);result={};sources={};height=None
    for key,file in [('elevation_median_m','elevation'),('slope_mean_deg','slope')]:
        with rasterio.open(cache/f'{file}.tif') as src:
            if src.crs.to_epsg()!=4326:raise ValueError('Unexpected terrain CRS')
            a,tr,window=padded_window(src,bounds)
            plan=OverlapPlan.build(a.shape,tr,GEO,features,geographic=True)
            mean,valid,total=plan.aggregate(a)
            source=dict(dataset='EarthEnv_topography_GMTED',window=window,decoded_window_sha256=p.sha(a.tobytes()),
                tags=src.tags(),native_grid_degrees=[abs(tr.a),abs(tr.e)],
                method='Spherical source-area overlap onto 1 km grid; accepted-area weighted 4x4 reduction.',
                warnings=['Published median filename conflicts with Maximum TIFF description.'] if file=='elevation' else [])
            crop=np.s_[10:-10,10:-10]
            result[key]=reduce4(mean[crop],valid[crop])
            sources[key]=source
            if file=='elevation':height=mean
            else:
                quality=np.divide(valid,total,out=np.zeros_like(valid),where=total>0)
                result['terrain_valid_fraction']=reduce4(quality[crop],total[crop]);sources['terrain_valid_fraction']=source
    for key,value in contexts(height,side_km).items():
        result[key]=value;sources[key]=sources['elevation_median_m']|dict(method='Original 5/21 km neighborhoods on 1 km elevation with 10 km halo, >=90% valid neighbors; mean over 16 subcells, >=50% valid subcells.')
    return result,sources


def cover(site,features,cache,catalog):
    n=math.isqrt(len(features));pb=p.bounds_of(features)
    area=np.array([shape(f['geometry']).area for f in features]).reshape(n,n)
    sums={k:np.zeros((n,n)) for k in ('bare_fraction','permanent_snow_ice_fraction')}
    valid_area=np.zeros((n,n));records=[]
    for tile in site['native_tiles']:
        if tile not in catalog:continue
        title=catalog[tile]['title'];path=cache/f'{title}.hdf'
        h=SD(str(path),SDC.READ)
        try:
            grid=p.grid_transform(h.attributes()['StructMetadata.0'],(2400,2400))
            f=from_bounds(*pb,grid)
            x0=max(0,math.floor(f.col_off));y0=max(0,math.floor(f.row_off))
            x1=min(2400,math.ceil(f.col_off+f.width));y1=min(2400,math.ceil(f.row_off+f.height))
            if x1<=x0 or y1<=y0:continue
            decoded=[]
            for name in ('LC_Type1','QC'):
                ds=h.select(name)
                try:decoded.append(ds.get(start=(y0,x0),count=(y1-y0,x1-x0)))
                finally:ds.endaccess()
            lc,qc=decoded;good=np.isin(qc,[0,2])&(lc>=1)&(lc<=17)
            tr=grid*Affine.translation(x0,y0)
            plan=OverlapPlan.build(lc.shape,tr,p.CRS_WKT,features)
            for key,category in [('bare_fraction',16),('permanent_snow_ice_fraction',15)]:
                mean,valid,_=plan.aggregate(np.where(good,(lc==category).astype(float),np.nan))
                sums[key]+=np.nan_to_num(mean)*valid
                if key=='bare_fraction':valid_area+=valid
            records.append(dict(title=title,window=[x0,y0,x1-x0,y1-y0],decoded_sha256=p.sha(lc.tobytes()+qc.tobytes())))
        finally:h.end()
    if np.max(valid_area/area)>1.0001:raise ValueError('Overlapping land-cover input')
    result={k:np.divide(v,valid_area,out=np.full_like(v,np.nan),where=valid_area>0) for k,v in sums.items()}
    result['landcover_valid_fraction']=np.clip(valid_area/area,0,1)
    source=dict(dataset='MCD12Q1.061',year=2024,files=records,qa='QC 0 or 2; IGBP classes 1..17.',
                method='Exact source sinusoidal area overlap; bare=16 and permanent ice=15 among classified support; quality relative to full target area.')
    return result,{k:source for k in result}


def permafrost(features,cache):
    n=math.isqrt(len(features));bounds=p.bounds_of(features)
    # Allocate the complete requested support, including outside the product domain.
    x0=math.floor((bounds[0]+180)/.01)-1;x1=math.ceil((bounds[2]+180)/.01)+1
    y0=math.floor((85-bounds[3])/.01)-1;y1=math.ceil((85-bounds[1])/.01)+1
    values=np.full((y1-y0,x1-x0),np.nan)
    with h5py.File(cache/'permafrost.nc','r') as f:
        ds=f['PFR'];lats=f['lat'][:];lons=f['lon'][:]
        if ds.shape!=(1,len(lats),len(lons)) or not np.all(np.diff(lats)>0) or not np.all(np.diff(lons)>0):raise ValueError('Unexpected CCI coordinates')
        if int(np.asarray(ds.attrs['_FillValue']).item())!=255 or ds.attrs['units']!=b'percent':raise ValueError('Unexpected CCI encoding')
        a=max(0,x0);b=min(len(lons),x1);c=max(0,y0);d=min(len(lats),y1)
        if a<b and c<d:
            raw=ds[0,len(lats)-d:len(lats)-c,a:b][::-1,:]
            values[c-y0:d-y0,a-x0:b-x0]=np.where(raw<=100,raw.astype(float)/100,np.nan)
    tr=from_origin(-180+x0*.01,85-y0*.01,.01,.01)
    plan=OverlapPlan.build(values.shape,tr,GEO,features,geographic=True)
    mean,valid,total=plan.aggregate(values)
    quality=np.divide(valid,total,out=np.zeros_like(valid),where=total>0)
    result={'permafrost_fraction':mean,'permafrost_valid_fraction':np.clip(quality,0,1)}
    source=dict(dataset='ESA_CCI_PERMAFROST_v5',year=2023,source_variable='PFR',
        method='Ascending CF latitude flipped north-up; percent/100; exact spherical area; outside-domain support remains missing.',
        warnings=['File GeoTransform conflicts with CF coordinate vectors; CF vectors authoritative.'])
    return result,{k:source for k in result}


def lithology(site,cache,side_km=400):
    payload=(cache/'lithology.zip').read_bytes()
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        lines=z.read('glim_wgs84_0point5deg.txt.asc').decode().splitlines()
        dictionary={int(r['Value_']):r['xx'] for r in csv.DictReader(io.StringIO(z.read('Classnames.txt').decode()),delimiter=';')}
    header={k.lower():float(v) for k,v in (line.split() for line in lines[:6])}
    if header!={'ncols':720,'nrows':360,'xllcorner':-180,'yllcorner':-90,'cellsize':.5,'nodata_value':-9999}:raise ValueError('Unexpected GLiM grid')
    grid=np.loadtxt(io.StringIO('\n'.join(lines[6:])),dtype=int)
    n=side_km//4;yy,xx=np.indices((n,n));half=side_km*500
    project=Transformer.from_crs(site['crs_wkt'],GEO,always_xy=True)
    lon,lat=project.transform(-half+(xx+.5)*4000,half-(yy+.5)*4000)
    codes=grid[np.floor((90-lat)/.5).astype(int),np.floor((lon+180)/.5).astype(int)]
    result={'dominant_lithology_code':np.where(np.isin(codes,[-9999,15]),np.nan,codes.astype(float))}
    source=dict(dataset='GLiM_v1.0_0.5_degree',class_dictionary=dictionary,native_grid_degrees=[.5,.5],
        method='Containing source cell at 4 km target center; categorical codes never averaged; source classes -9999/15 missing.')
    return result,{k:source for k in result}


def climate(features,paths):
    records={};headers={}
    for path in paths:
        obj=json.loads(Path(path).read_text())
        for parameter in obj['parameters']:
            if parameter not in ('T2M','PRECTOTCORR'):continue
            if obj['parameters'][parameter]['units']!=('C' if parameter=='T2M' else 'mm/day'):raise ValueError('Unexpected POWER unit')
            headers[parameter]=obj['header']
            for f in obj['features']:
                x,y,*_=f['geometry']['coordinates'];key=(parameter,float(x),float(y))
                data=f['properties']['parameter'][parameter]
                if key in records and records[key]!=data:raise ValueError('Conflicting overlapping POWER cells')
                records[key]=data
    bounds=p.bounds_of(features)
    xs=sorted(set(x for _,x,y in records));ys=sorted(set(y for _,x,y in records),reverse=True)
    if not xs or not ys:raise ValueError('Empty POWER response')
    # Expand to full regular native grid, so missing responses cannot shrink denominator.
    xs=np.arange(min(xs),max(xs)+.3125,.625);ys=np.arange(max(ys),min(ys)-.25,-.5)
    tr=from_origin(xs[0]-.3125,ys[0]+.25,.625,.5)
    if tr.c>bounds[0] or tr.f<bounds[3] or tr.c+len(xs)*.625<bounds[2] or tr.f-len(ys)*.5>bounds[1]:raise ValueError('POWER source support does not cover target')
    plan=OverlapPlan.build((len(ys),len(xs)),tr,GEO,features,geographic=True)
    result={};sources={}
    for key,parameter in [('coldest_month_air_c','T2M'),('warmest_month_air_c','T2M'),('annual_precip_mm','PRECTOTCORR')]:
        header=headers[parameter];fill=header['fill_value']
        start=int(header['start'][:4]);end=int(header['end'][:4]);days=sum(366 if calendar.isleap(y) else 365 for y in range(start,end+1))/(end-start+1)
        a=np.full((len(ys),len(xs)),np.nan)
        for (param,x,y),v in records.items():
            if param!=parameter:continue
            monthly=np.array([v[m] for m in p.MONTHS],dtype=float)
            if np.any(monthly==fill) or not np.all(np.isfinite(monthly)):continue
            value=float(min(monthly) if key.startswith('coldest') else max(monthly))
            if parameter=='PRECTOTCORR':
                if v['ANN']==fill or v['ANN']<0:continue
                value=v['ANN']*days
            row=round((ys[0]-y)/.5);col=round((x-xs[0])/.625);a[row,col]=value
        mean,valid,total=plan.aggregate(a)
        # Missing POWER land/ocean/ice evidence remains missing, never zero.
        coverage=np.divide(valid,total,out=np.zeros_like(valid),where=total>0)
        mean[coverage<.999]=np.nan
        result[key]=mean;sources[key]=dict(dataset='NASA_POWER_MERRA2',header=header,native_grid_degrees=[.625,.5],
            method='Deduplicated native grid supports; exact spherical-area average; require >=99.9% support. No fine-scale downscaling.',
            annual_precip_days_per_climatological_year=days if parameter=='PRECTOTCORR' else None,
            note='Air monthly-mean extrema; annual precipitation from ANN mean daily rate.')
    return result,sources
