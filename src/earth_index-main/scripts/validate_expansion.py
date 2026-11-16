# /// script
# requires-python = ">=3.11"
# dependencies = ["psycopg[binary]==3.3.9", "rasterio==1.5.2", "numpy==2.5.4", "shapely==2.2.0", "pyproj==3.8.0", "h3==4.5.0"]
# ///
"""Independent checks of codes, coarse/fine agreement and frontend SQL scores."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import psycopg
from psycopg import sql
from pyproj import Geod,Transformer
import rasterio
from shapely.geometry import Polygon,mapping
from shapely.ops import unary_union

from load_raster_pilot import ROOT,connection,manifest


def arrays(data):
    result={}
    for l in data['layers']:
        with rasterio.open(ROOT/l['path']) as src:a=src.read(1)
        result.setdefault(l['site_key'],{})[l['feature_key']]=np.where(a==0,np.nan,a.astype(float)*l['scale']+l['offset'])
    return result


def score(values,frozen):
    fields={f['key']:f for f in frozen['fields']};groups={};known=np.ones(next(iter(values.values())).shape,dtype=bool)
    for c in frozen['criteria']:
        if not c['enabled']:continue
        f=fields[c['key']];v=values[c['key']];known &= np.isfinite(v)
        if f['qualityKey']:known &= values[f['qualityKey']]>=.9
        loss=np.where(v==c['target'],0.,25.) if f['categorical'] else ((v-c['target'])/c['tolerance'])**2
        groups.setdefault(f['group'],[]).append((loss,c['weight']))
    total=np.zeros_like(known,dtype=float);weights=0.
    for members in groups.values():
        vote=max(w for _,w in members);weights+=vote
        total+=vote*sum(loss*w for loss,w in members)/sum(w for _,w in members)
    value=np.exp(-.5*np.minimum(1000,total/weights))
    # PostgreSQL round(float8) follows ties-to-even as NumPy does.
    codes=np.where(known,np.rint(value*65534),65535).astype('<u2')
    return codes


def stats(a):
    v=a[np.isfinite(a)]
    return dict(valid=int(v.size),missing=int(a.size-v.size),min=float(v.min()) if v.size else None,
                max=float(v.max()) if v.size else None,mean=float(v.mean()) if v.size else None)


def geometry(site):
    project=Transformer.from_crs(site['crs_wkt'],4326,always_xy=True)
    corners=[(-200000,200000),(200000,200000),(200000,-200000),(-200000,-200000),(-200000,200000)]
    ring=[]
    for a,b in zip(corners,corners[1:]):ring.extend(project.transform(a[0]+(b[0]-a[0])*i/160,a[1]+(b[1]-a[1])*i/160) for i in range(160))
    return Polygon(ring)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'research/expansion-4km');parser.add_argument('--database',action='store_true')
    args=parser.parse_args();out=args.output;start=time.perf_counter();data,bundle=manifest(out)
    new=arrays(data);old=arrays(json.loads((ROOT/'research/thermal-pilot/manifest.json').read_text()))
    report=dict(bundle_sha256=bundle,datasource_id='earth-'+bundle[:12],schema='raster_'+bundle[:32],checks={},quality={},shared_area={},matching={})
    keys=set(next(iter(new.values())))
    assert len(keys)==25 and len(new)==38
    for site,v in new.items():
        assert set(v)==keys
        for key,a in v.items():
            assert a.shape==(100,100) and not np.isinf(a).any()
            if key.endswith('_fraction'):assert not np.any((a<-.000001)|(a>1.000001))
            if key.startswith('relief_') or key=='annual_precip_mm':assert not np.any(a<0)
            if key=='dominant_lithology_code':assert set(a[np.isfinite(a)]).issubset({*range(1,15),16})
        assert not np.any(v['surface_low_c']>v['surface_high_c']+.02)
        assert np.nanmax(np.abs(v['surface_high_c']-v['surface_low_c']-v['surface_range_c']))<=.0300001 if np.isfinite(v['surface_range_c']).any() else True
        report['quality'][site]={k:stats(a) for k,a in v.items()}
    assert np.isnan(new['atacama']['permafrost_fraction']).all()
    assert np.all(new['atacama']['permafrost_valid_fraction']==0)
    report['checks']['all25_code_ranges_and_missingness']='passed'
    # Compare precisely aligned shared inner 96x96 km. Thermal has nonlinear
    # time-support rules and is evaluated as a score, not a commuting average.
    for key in sorted(keys-{'dominant_lithology_code'}-{k for k in keys if k.startswith('surface_')}):
        differences=[];validity_agreement=[];high_support_differences=[]
        support_key={'bare_fraction':'landcover_valid_fraction','permanent_snow_ice_fraction':'landcover_valid_fraction','permafrost_fraction':'permafrost_valid_fraction','elevation_median_m':'terrain_valid_fraction','slope_mean_deg':'terrain_valid_fraction'}.get(key)
        for site,values in old.items():
            fine=values[key][2:98,2:98].reshape(24,4,24,4)
            valid=np.isfinite(fine);counts=valid.sum(axis=(1,3))
            weights=values[support_key][2:98,2:98].reshape(24,4,24,4) if support_key else np.ones_like(fine)
            weights=np.where(valid,weights,0.);accepted=weights.sum(axis=(1,3))
            expected=np.divide((np.where(valid,fine,0)*weights).sum(axis=(1,3)),accepted,out=np.full((24,24),np.nan),where=accepted>0)
            actual=new[site][key][38:62,38:62]
            shared=np.isfinite(expected)&np.isfinite(actual)
            differences.extend(np.abs(actual[shared]-expected[shared]).tolist())
            supported=shared&(new[site][support_key][38:62,38:62]>=.9) if support_key else shared
            high_support_differences.extend(np.abs(actual[supported]-expected[supported]).tolist())
            validity_agreement.extend((np.isfinite(expected)==np.isfinite(actual)).ravel().tolist())
        d=np.array(differences)
        report['shared_area'][key]=dict(cells=len(differences),mean_abs_difference=float(d.mean()) if d.size else None,
            p95_abs_difference=float(np.quantile(d,.95)) if d.size else None,max_abs_difference=float(d.max()) if d.size else None,
            validity_agreement=float(np.mean(validity_agreement)),
            method='Accepted-support weighted mean of fine cells' if support_key else 'Mean of fine cells',
            high_support_p95_difference=float(np.quantile(high_support_differences,.95)) if high_support_differences else None)
    validation=out/'validation'
    targets=json.loads((validation/'targets.json').read_text())
    for frozen in targets:
        target=frozen['target']['key'];compared=0;disagreement=0;supported=0;coarse_high=0;unsupported_high=0;severe_high=0;correlations=[];negative_cells=0;negative_high=0;negative_max=0.;shared_deltas=[];new_known_fine_unknown=0
        for site,values in new.items():
            expected=score(values,frozen)
            binary=validation/f'expanded-{target}-{site}.bin'
            if not binary.exists():continue
            actual=np.fromfile(binary,dtype='<u2').reshape(100,100)
            compared+=actual.size;disagreement+=int((expected!=actual).sum())
            negative=(values['bare_fraction']<.05)&(actual!=65535)
            negative_cells+=int(negative.sum());negative_high+=int((negative&(actual/65534>=.75)).sum())
            if negative.any():negative_max=max(negative_max,float(actual[negative].max()/65534))
            if site in old:
                fine=np.fromfile(validation/f'baseline-{target}-{site}.bin',dtype='<u2').reshape(100,100)[2:98,2:98].reshape(24,4,24,4)
                valid=fine!=65535;count=valid.sum(axis=(1,3));sums=np.where(valid,fine.astype(float)/65534,0).sum(axis=(1,3))
                mean=np.divide(sums,count,out=np.full((24,24),np.nan),where=count>0)
                maximum=np.where(valid,fine.astype(float)/65534,np.nan)
                # All-missing windows are recorded separately, not evidence against match.
                maximum=np.max(np.where(valid,maximum,-1),axis=(1,3))
                coarse=actual[38:62,38:62];known=(coarse!=65535)&(count>0)
                new_known_fine_unknown+=int(((coarse!=65535)&(count==0)).sum())
                high=known&(coarse/65534>=.75)
                shared_deltas.extend((coarse[known]/65534-mean[known]).tolist())
                coarse_high+=int(high.sum());unsupported_high+=int((high&(maximum<.75)).sum());severe_high+=int((known&(coarse/65534>=.9)&(maximum<.75)).sum());supported+=int(known.sum())
                if known.sum()>2 and np.std(mean[known])>0 and np.std(coarse[known])>0:
                    correlations.append(float(np.corrcoef(mean[known],coarse[known]/65534)[0,1]))
        if compared:
            assert disagreement==0,(target,disagreement)
            assert negative_high==0,(target,negative_high)
            report['matching'][target]=dict(sql_numpy_cells=compared,code_disagreements=disagreement,
                shared_supported_coarse_cells=supported,shared_coarse_score_ge75=coarse_high,
                coarse_known_where_all_fine_unknown=new_known_fine_unknown,
                nonbare_control_cells=negative_cells,nonbare_control_ge75=negative_high,nonbare_control_max=negative_max,
                shared_mean_score_delta=float(np.mean(shared_deltas)) if shared_deltas else None,
                shared_p95_abs_score_delta=float(np.quantile(np.abs(shared_deltas),.95)) if shared_deltas else None,
                high_coarse_without_any_fine_ge75=unsupported_high,
                coarse_ge90_without_any_fine_ge75=severe_high,
                mean_patch_pearson_vs_mean_fine=float(np.mean(correlations)) if correlations else None)
    geod=Geod(ellps='WGS84');footprints=[geometry(site) for site in data['sites']]
    union=unary_union(footprints)
    # Orient rings for signed ellipsoidal area.
    from shapely.geometry.polygon import orient
    parts=list(union.geoms) if union.geom_type=='MultiPolygon' else [union]
    area=sum(abs(geod.geometry_area_perimeter(orient(part))[0]) for part in parts)/1e6
    report['coverage']=dict(union_area_km2=area,percent_earth_surface=area/510065621.724*100,nominal_area_km2=len(new)*160000,
                           footprint_note='Product footprints, not all-variable usable land; overlapping regions counted once in union.')
    (out/'coverage.geojson').write_text(json.dumps(dict(type='FeatureCollection',features=[dict(type='Feature',properties={'site_key':site['key']},geometry=mapping(g)) for site,g in zip(data['sites'],footprints)])))
    if args.database:
        with connection() as conn:
            conn.execute("SET LOCAL postgis.gdal_enabled_drivers='GTiff'")
            checked=0
            for layer in data['layers']:
                row=conn.execute('''SELECT ST_DumpValues(t.rast,1,false),ST_Width(t.rast),ST_Height(t.rast),ST_ScaleX(t.rast),ST_ScaleY(t.rast),ST_UpperLeftX(t.rast),ST_UpperLeftY(t.rast)
                    FROM data.raster_tiles t JOIN catalog.raster_layers l USING(layer_id) WHERE bundle_sha256=%s AND site_key=%s AND feature_key=%s''',
                    (bundle,layer['site_key'],layer['feature_key'])).fetchone()
                with rasterio.open(ROOT/layer['path']) as src:expected=src.read(1)
                assert np.array_equal(np.asarray(row[0],dtype='<u2'),expected)
                assert row[1:]==(100,100,4000.,-4000.,-200000.,200000.)
                checked+=expected.size
            profiles=json.loads((ROOT/'research/priority-pilot/priorities.json').read_text())['profiles']
            profile_results={}
            for name,profile in profiles.items():
                clauses=[];parameters=[];expected_counts={}
                for key,limits in profile['conditions'].items():
                    for direction,value in limits.items():
                        operator='>=' if direction=='min' else '<='
                        clauses.append(sql.SQL('{} '+operator+' %s').format(sql.Identifier(key)))
                        parameters.append(value-1e-9 if direction=='min' else value+1e-9)
                query=sql.SQL('SELECT site_key,count(*) FILTER (WHERE {}) FROM {}.cells GROUP BY site_key').format(sql.SQL(' AND ').join(clauses),sql.Identifier(report['schema']))
                actual_counts=dict(conn.execute(query,parameters).fetchall())
                for site,values in new.items():
                    mask=np.ones((100,100),dtype=bool)
                    for key,limits in profile['conditions'].items():
                        a=values[key];mask &= np.isfinite(a)
                        if 'min' in limits:mask &= a>=limits['min']-1e-9
                        if 'max' in limits:mask &= a<=limits['max']+1e-9
                    expected_counts[site]=int(mask.sum())
                assert actual_counts==expected_counts,(name,actual_counts,expected_counts)
                profile_results[name]=actual_counts
            assert profile_results['dry_bare']['atacama']>0
            assert profile_results['hot_dry_bare']['deathvalley']>0
            assert profile_results['cold_bare_frozen']['haughton']>0
            report['reference_profiles']=profile_results
            # No original bundle, value count or layer hash may change.
            before=json.loads((out/'database-before.json').read_text())
            for identity,count,valid,checksum in before['bundles']:
                actual=conn.execute("SELECT count(*),sum(valid_cells),md5(string_agg(file_sha256,'' ORDER BY site_key,feature_key)) FROM catalog.raster_layers WHERE bundle_sha256=%s",(identity,)).fetchone()
                assert actual==(count,valid,checksum)
            original_codes=0
            old_manifest=json.loads((ROOT/'research/thermal-pilot/manifest.json').read_text())
            old_bundle='c994825c6379ec01fbb007fa550218fad9f94464b83de270aa8b97e5881aebf7'
            for layer in old_manifest['layers']:
                row=conn.execute('SELECT ST_DumpValues(t.rast,1,false) FROM data.raster_tiles t JOIN catalog.raster_layers l USING(layer_id) WHERE bundle_sha256=%s AND site_key=%s AND feature_key=%s',
                    (old_bundle,layer['site_key'],layer['feature_key'])).fetchone()
                with rasterio.open(ROOT/layer['path']) as src:expected=src.read(1)
                assert np.array_equal(np.asarray(row[0],dtype='<u2'),expected)
                original_codes+=expected.size
            report['checks']['original_active_codes_verified']=original_codes
            rejected=0
            layer_id=conn.execute('SELECT layer_id FROM catalog.raster_layers WHERE bundle_sha256=%s LIMIT 1',(bundle,)).fetchone()[0]
            for expression in ['ST_SetScale(rast,1000,-1000)','ST_SetUpperLeft(rast,0,0)','ST_SetSRID(rast,4326)']:
                try:
                    with conn.transaction():conn.execute(sql.SQL('UPDATE data.raster_tiles SET rast={} WHERE layer_id=%s').format(sql.SQL(expression)),(layer_id,))
                except psycopg.Error:rejected+=1
                else:raise AssertionError('Invalid raster grid accepted')
            report['checks'].update(database_codes_verified=checked,original_bundles_unchanged=len(before['bundles']),invalid_grid_updates_rejected=rejected)
    report['wall_seconds']=time.perf_counter()-start
    report['limitations']=['Purposive sites, not ground-truth analog validation.','4 km metadata averages can hide subcell variation; fine/coarse disagreement is measured.',
        'Earth MODIS clear-sky composite temperatures and planetary models are approximate comparisons.','Footprint union includes missing evidence and water; score areas are nominal and regions can overlap.']
    (out/'validation-report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('quality','shared_area')},indent=2))


if __name__=='__main__':main()
