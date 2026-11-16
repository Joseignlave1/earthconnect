# /// script
# requires-python = ">=3.11"
# dependencies = ["psycopg[binary]==3.3.9", "h3==4.5.0", "rasterio==1.5.2", "numpy==2.5.4"]
# ///
"""Measure generous retrieval and continuous ranking on the existing raster pilot.

No schema/index changes, materialized caches or additional source downloads.
SQL computes the actual scores; NumPy independently checks every window.
"""
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np
import psycopg
from psycopg import sql
import rasterio
from rasterio.warp import transform

from load_raster_pilot import ROOT,connection,manifest

OUT=ROOT/'research/ranking-pilot'
TOL=1e-9


def sha(data):return hashlib.sha256(data).hexdigest()


def component_sql(spec):
    field=sql.Identifier(spec['field']);kind=spec['transform']
    if kind=='fraction':expr=sql.SQL('greatest(0.0,least(1.0,{}))').format(field)
    elif kind=='exponential_decay':expr=sql.SQL('exp(-{} / {})').format(field,sql.Literal(float(spec['scale'])))
    elif kind=='logistic_increasing':
        expr=sql.SQL('1.0/(1.0+exp(-({}-{})/{}))').format(field,sql.Literal(float(spec['midpoint'])),sql.Literal(float(spec['width'])))
    elif kind=='logistic_decreasing':
        expr=sql.SQL('1.0/(1.0+exp(({}-{})/{}))').format(field,sql.Literal(float(spec['midpoint'])),sql.Literal(float(spec['width'])))
    else:raise ValueError('Unknown score transform')
    unknown=sql.SQL('{} IS NULL').format(field)
    if 'coverage_field' in spec:
        coverage=sql.Identifier(spec['coverage_field'])
        unknown=sql.SQL('({} OR {} IS NULL OR {} < {})').format(unknown,coverage,coverage,sql.Literal(spec['minimum_coverage']-TOL))
    # PostgreSQL LEAST/GREATEST ignore NULL: preserve missingness explicitly.
    return sql.SQL('CASE WHEN {} THEN NULL ELSE {} END').format(unknown,expr)


def query(spec,weights):
    gate=spec['candidate_gate'];parts=[];logs=[];known=[];zeros=[]
    for key,weight in weights.items():
        name=sql.Identifier('c_'+key)
        parts.append(sql.SQL('{} AS {}').format(component_sql(spec['components'][key]),name))
        logs.append(sql.SQL('coalesce({}*ln(nullif({},0)),0)').format(sql.Literal(weight),name))
        known.append(sql.SQL('CASE WHEN {} IS NOT NULL THEN {} ELSE 0 END').format(name,sql.Literal(weight)))
        zeros.append(sql.SQL('{}=0').format(name))
    classification=sql.SQL('''CASE
        WHEN (landcover_valid_fraction >= {lc} AND bare_fraction < {bare})
          OR (terrain_valid_fraction >= {tc} AND slope_mean_deg > {slope}) THEN 'excluded'
        WHEN landcover_valid_fraction IS NULL OR landcover_valid_fraction < {lc}
          OR terrain_valid_fraction IS NULL OR terrain_valid_fraction < {tc}
          OR bare_fraction IS NULL OR slope_mean_deg IS NULL THEN 'unknown'
        ELSE 'candidate' END''').format(lc=sql.Literal(gate['landcover_valid_fraction_min']-TOL),
            tc=sql.Literal(gate['terrain_valid_fraction_min']-TOL),
            bare=sql.Literal(gate['bare_fraction_min']-TOL),slope=sql.Literal(gate['mean_slope_max_deg']+TOL))
    return sql.SQL('''WITH cells AS MATERIALIZED (
        SELECT * FROM research.priority_pilot_cells WHERE bundle_sha256=%s
    ), components AS MATERIALIZED (
        SELECT site_key,x,y,{classification} AS gate,{parts} FROM cells
    ), bounded AS (
        SELECT *,({evidence})::float8 AS evidence,
            CASE WHEN ({zeros}) THEN 0 ELSE exp({logs}) END AS score_upper
        FROM components
    ), scored AS (
        SELECT *,CASE WHEN evidence >= 0.999999999 THEN score_upper ELSE 0 END AS score_lower
        FROM bounded
    )
    SELECT site_key,((x-1)/{width})+1 AS wx,((y-1)/{width})+1 AS wy,
        count(*) AS pixels,
        count(*) FILTER (WHERE gate='candidate') AS candidate_cells,
        count(*) FILTER (WHERE gate='unknown') AS unknown_cells,
        count(*) FILTER (WHERE gate='excluded') AS excluded_cells,
        count(*) FILTER (WHERE gate='candidate' AND evidence>=0.999999999) AS fully_scored_candidate_cells,
        sum(CASE WHEN gate='candidate' THEN score_lower ELSE 0 END)/{area} AS score_lower,
        sum(CASE WHEN gate<>'excluded' THEN score_upper ELSE 0 END)/{area} AS score_upper,
        sum(CASE WHEN gate<>'excluded' THEN evidence ELSE 0 END)/{area} AS component_evidence
    FROM scored GROUP BY site_key,wx,wy ORDER BY site_key,wy,wx''').format(
        classification=classification,parts=sql.SQL(',').join(parts),evidence=sql.SQL('+').join(known),
        zeros=sql.SQL(' OR ').join(zeros),logs=sql.SQL('+').join(logs),
        width=sql.Literal(spec['window_pixels']),area=sql.Literal(float(spec['window_score_denominator'])))


def arrays(data):
    result={}
    for layer in data['layers']:
        with rasterio.open(ROOT/layer['path']) as src:codes=src.read(1)
        result.setdefault(layer['site_key'],{})[layer['feature_key']]=np.where(codes==0,np.nan,codes.astype(float)*layer['scale']+layer['offset'])
    return result


def components_numpy(values,spec):
    result={}
    for name,c in spec['components'].items():
        a=values[c['field']];kind=c['transform']
        if kind=='fraction':v=np.clip(a,0,1)
        elif kind=='exponential_decay':v=np.exp(-a/c['scale'])
        elif kind=='logistic_increasing':v=1/(1+np.exp(-(a-c['midpoint'])/c['width']))
        elif kind=='logistic_decreasing':v=1/(1+np.exp((a-c['midpoint'])/c['width']))
        else:raise ValueError('Unknown transform')
        if 'coverage_field' in c:
            coverage=values[c['coverage_field']]
            v=np.where(np.isfinite(coverage)&(coverage>=c['minimum_coverage']-TOL),v,np.nan)
        result[name]=v
    return result


def gate_numpy(a,spec):
    g=spec['candidate_gate'];lc=a['landcover_valid_fraction'];tc=a['terrain_valid_fraction']
    reliable_land=np.isfinite(lc)&(lc>=g['landcover_valid_fraction_min']-TOL)
    reliable_terrain=np.isfinite(tc)&(tc>=g['terrain_valid_fraction_min']-TOL)
    incompatible=(reliable_land&(a['bare_fraction']<g['bare_fraction_min']-TOL))|(reliable_terrain&(a['slope_mean_deg']>g['mean_slope_max_deg']+TOL))
    observed=reliable_land&reliable_terrain&np.isfinite(a['bare_fraction'])&np.isfinite(a['slope_mean_deg'])
    # 0 = observed incompatibility; 1 = eligible; 2 = unknown eligibility.
    return np.where(incompatible,0,np.where(observed,1,2))


def scores_numpy(components,weights):
    shape=next(iter(components.values())).shape;logs=np.zeros(shape);known=np.zeros(shape);zero=np.zeros(shape,bool)
    for key,weight in weights.items():
        a=components[key];finite=np.isfinite(a)
        known+=weight*finite;zero|=finite&(a==0)
        logs+=weight*np.log(np.where(finite&(a>0),a,1))
    upper=np.where(zero,0,np.exp(logs));lower=np.where(known>=1-TOL,upper,0)
    return lower,upper,known


def windows_numpy(all_arrays,spec,weights):
    rows=[];pixels={};width=spec['window_pixels']
    for key,a in all_arrays.items():
        gate=gate_numpy(a,spec);low,high,evidence=scores_numpy(components_numpy(a,spec),weights)
        pixels[key]={'gate':gate,'lower':low,'upper':high,'evidence':evidence}
        for wy in range(100//width):
            for wx in range(100//width):
                region=(slice(wy*width,(wy+1)*width),slice(wx*width,(wx+1)*width))
                g=gate[region];lo=low[region];hi=high[region];e=evidence[region]
                rows.append({'site_key':key,'wx':wx+1,'wy':wy+1,'pixels':width*width,
                    'candidate_cells':int((g==1).sum()),'unknown_cells':int((g==2).sum()),'excluded_cells':int((g==0).sum()),
                    'fully_scored_candidate_cells':int(((g==1)&(e>=1-TOL)).sum()),
                    'score_lower':float(np.where(g==1,lo,0).mean()),'score_upper':float(np.where(g!=0,hi,0).mean()),
                    'component_evidence':float(np.where(g!=0,e,0).mean())})
    return sorted(rows,key=lambda r:(r['site_key'],r['wy'],r['wx'])),pixels


def verify(actual,expected):
    if len(actual)!=len(expected):raise AssertionError('Wrong window count')
    max_error=0
    for got,want in zip(actual,expected):
        for key,value in want.items():
            if isinstance(value,float):
                error=abs(got[key]-value);max_error=max(max_error,error)
                if error>TOL:raise AssertionError(f'SQL/NumPy score mismatch: {key}')
            elif got[key]!=value:raise AssertionError(f'SQL/NumPy window mismatch: {key}')
    return max_error


def order(rows,field='score_lower'):
    return sorted(rows,key=lambda r:(-round(r[field],12),-round(r['component_evidence'],12),r['site_key'],r['wy'],r['wx']))


def diverse(rows,field='score_lower'):
    best={}
    for row in order(rows,field):best.setdefault(row['site_key'],row)
    return order(list(best.values()),field)


def index_fingerprint():
    with connection() as conn:
        rows=conn.execute("SELECT schemaname,indexname,indexdef FROM pg_indexes WHERE schemaname IN ('catalog','data','research','spatial') ORDER BY 1,2").fetchall()
    return {'sha256':sha(json.dumps(rows).encode()),'count':len(rows)}


def run_sql(statement,bundle):
    started=time.monotonic()
    try:
        with connection() as conn:
            conn.execute('SET LOCAL ROLE earth_index_reader');conn.execute("SET LOCAL statement_timeout='10s'")
            cursor=conn.execute(statement,(bundle,));names=[d.name for d in cursor.description]
            rows=[dict(zip(names,row)) for row in cursor.fetchall()]
        return rows,{'status':'passed','wall_seconds':time.monotonic()-started}
    except psycopg.errors.QueryCanceled:
        return None,{'status':'timeout','timeout_seconds':10,'wall_seconds':time.monotonic()-started}


def summarize(rows,pixels,sites,all_arrays,spec,weights):
    ranked=diverse(rows);ranked_positive=[r for r in ranked if r['score_lower']>TOL]
    for rank,row in enumerate(ranked_positive,1):row['rank']=rank
    raw=order([r for r in rows if r['score_lower']>TOL])[:20]
    uncertain=order([r for r in rows if r['score_upper']>r['score_lower']+TOL],field='score_upper')[:20]
    roles={s['key']:s['role'] for s in sites};summary={}
    for s in sites:
        key=s['key'];p=pixels[key];g=p['gate'];best=next(r for r in ranked if r['site_key']==key)
        width=spec['window_pixels'];region=(slice((best['wy']-1)*width,best['wy']*width),slice((best['wx']-1)*width,best['wx']*width))
        projected_x=-50000+(best['wx']-.5)*width*1000;projected_y=50000-(best['wy']-.5)*width*1000
        lon,lat=transform(s['crs_wkt'],'EPSG:4326',[projected_x],[projected_y])
        best['center_lon_lat']=[lon[0],lat[0]]
        means={}
        for field in {spec['components'][k]['field'] for k in weights}:
            values=all_arrays[key][field][region];valid=(g[region]==1)&np.isfinite(values)
            means[field]=float(values[valid].mean()) if valid.any() else None
        candidates=g==1
        summary[key]={'role':s['role'],'name':s['name'],'best_window':best,
            'best_window_candidate_pixel_physical_means':means,
            'candidate_cells':int(candidates.sum()),'excluded_cells':int((g==0).sum()),'unknown_gate_cells':int((g==2).sum()),
            'fully_scored_candidate_cells':int((candidates&(p['evidence']>=1-TOL)).sum()),
            'best_possible_window':diverse([r for r in rows if r['site_key']==key],'score_upper')[0]}
    # Compare ungated fully observed pixel scores to the broad gate. This proxy
    # measures whether the gate throws away the scoring model's own best results.
    ungated=[]
    for key,p in pixels.items():
        yy,xx=np.where(p['evidence']>=1-TOL)
        ungated.extend((float(p['lower'][y,x]),key,int(y),int(x),int(p['gate'][y,x])) for y,x in zip(yy,xx))
    ungated.sort(key=lambda v:(-round(v[0],12),v[1],v[2],v[3]));n=max(1,int(np.ceil(len(ungated)*.01)))
    top=ungated[:n];retained=sum(v[4]==1 for v in top)
    return {'positive_ranked_patches':len(ranked_positive),'ranked_patches':ranked_positive,'sites':summary,
            'raw_top20_windows':raw,'raw_top20_unique_patches':len({r['site_key'] for r in raw}),
            'uncertainty_top20_windows':uncertain,
            'top5_role_counts':dict(Counter(roles[r['site_key']] for r in ranked_positive[:5])),
            'gate_totals':{name:sum(s[field] for s in summary.values()) for name,field in [('candidates','candidate_cells'),('excluded','excluded_cells'),('unknown','unknown_gate_cells')]},
            'gate_retention_of_ungated_best_1percent':{'fully_observed_pixels':len(ungated),'top_pixels':len(top),'retained':retained,
                'fraction':retained/len(top) if top else None,'caveat':'Self-consistency of this score and gate, not scientific recall.'}}


def sensitivity(all_arrays,spec,base):
    result={}
    for name,profile in spec['profiles'].items():
        baseline={r['site_key'] for r in base[name]['ranked_patches'][:5]};variants=[]
        for component in profile['weights']:
            for factor in (.8,1.2):
                weights={k:v*(factor if k==component else 1) for k,v in profile['weights'].items()}
                total=sum(weights.values());weights={k:v/total for k,v in weights.items()}
                rows,_=windows_numpy(all_arrays,spec,weights)
                ordered=[r['site_key'] for r in diverse(rows) if r['score_lower']>TOL][:5];top=set(ordered)
                union=baseline|top;jaccard=len(baseline&top)/len(union) if union else 1
                variants.append({'component':component,'factor':factor,'top5':ordered,'top5_set_jaccard':jaccard})
        result[name]={'variants':variants,'minimum_top5_set_jaccard':min(v['top5_set_jaccard'] for v in variants)}
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--repetitions',type=int,default=3)
    parser.add_argument('--report',type=Path,default=ROOT/'research/pilots/database/ranking-evaluation.json')
    parser.add_argument('--artifacts-dir',type=Path,default=OUT)
    args=parser.parse_args();start=time.monotonic();raw=(OUT/'spec.json').read_bytes();spec=json.loads(raw)
    args.artifacts_dir.mkdir(parents=True,exist_ok=True)
    if args.repetitions<1 or args.repetitions>3:parser.error('Use 1..3 timing samples')
    if any(abs(sum(p['weights'].values())-1)>TOL for p in spec['profiles'].values()):raise ValueError('Weights must sum to one')
    data,bundle=manifest(ROOT/'research/stress-pilot')
    if bundle!=spec['source_bundle_sha256']:raise ValueError('Ranking source bundle changed')
    before=index_fingerprint();all_arrays=arrays(data);profiles={};timeouts={}
    for name,profile in spec['profiles'].items():
        expected,pixels=windows_numpy(all_arrays,spec,profile['weights']);statement=query(spec,profile['weights']);samples=[];rows=None;errors=[]
        with connection() as conn:(args.artifacts_dir/f'{name}.sql').write_text(statement.as_string(conn)+';\n')
        for _ in range(args.repetitions):
            got,timing=run_sql(statement,bundle);samples.append(timing)
            if got is not None:errors.append(verify(got,expected));rows=got
            print(name+': '+timing['status']+f" {timing['wall_seconds']:.2f}s",flush=True)
        if rows is None:timeouts[name]=samples;continue
        summary=summarize(rows,pixels,data['sites'],all_arrays,spec,profile['weights']);summary.update(label=profile['label'],weights=profile['weights'],timings=samples,
            median_successful_wall_seconds=statistics.median(t['wall_seconds'] for t in samples if t['status']=='passed'),
            sql_windows_verified=len(rows),max_sql_numpy_absolute_error=max(errors))
        profiles[name]=summary
        (args.artifacts_dir/f'{name}-windows.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
    after=index_fingerprint()
    if before!=after:raise AssertionError('Database indexes changed during evaluation')
    if sha((OUT/'spec.json').read_bytes())!=sha(raw):raise AssertionError('Ranking rules changed during evaluation')
    report={'status':'passed' if not timeouts else 'incomplete_due_to_query_timeout','bundle_sha256':bundle,'spec_sha256':sha(raw),
        'processor_sha256':sha(Path(__file__).read_bytes()),
        'generated_at':datetime.now(timezone.utc).isoformat(),'wall_seconds':time.monotonic()-start,'patches':len(data['sites']),
        'windows_per_profile':len(data['sites'])*100,'profiles':profiles,'timeouts':timeouts,
        'index_fingerprint_before':before,'index_fingerprint_after':after,
        'sensitivity':sensitivity(all_arrays,spec,profiles) if len(profiles)==len(spec['profiles']) else None,
        'limits':['Known purposive sample, exploratory preference curves; no calibrated analog probability or global accuracy.',
                  'SQL score/evidence bounds and all window counts independently verified; scientific uncertainty is much larger than arithmetic error.',
                  '1km mean slope and coarse climate cannot certify a fine-scale usable landing/trafficability footprint.',
                  'Geographic patch diversity is a diagnostic grouping, not a modification of the raw global window ranking.']}
    report['wall_seconds']=time.monotonic()-start
    args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':report['status'],'profiles':{k:{'top5':[r['site_key'] for r in v['ranked_patches'][:5]],
         'ranked_patches':v['positive_ranked_patches'],'median_seconds':v['median_successful_wall_seconds']} for k,v in profiles.items()},'timeouts':timeouts},indent=2))


if __name__=='__main__':main()
