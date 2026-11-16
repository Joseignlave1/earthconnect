# /// script
# requires-python = ">=3.11"
# dependencies = ["psycopg[binary]==3.3.9", "h3==4.5.0", "rasterio==1.5.2", "numpy==2.5.4"]
# ///
"""Inspect source support and sensitivity to stricter data coverage, without SQL changes."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

from evaluate_ranking_pilot import OUT,TOL,arrays,diverse,windows_numpy
from load_raster_pilot import ROOT,manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ranking-report',type=Path,default=ROOT/'research/pilots/database/ranking-evaluation.json')
    parser.add_argument('--report',type=Path,default=ROOT/'research/pilots/database/ranking-coverage-audit.json')
    args=parser.parse_args();ranking=json.loads(args.ranking_report.read_text());raw=(OUT/'spec.json').read_bytes();spec=json.loads(raw)
    if hashlib.sha256(raw).hexdigest()!=ranking['spec_sha256']:raise ValueError('Ranking specification changed')
    data,bundle=manifest(ROOT/'research/stress-pilot')
    if bundle!=ranking['bundle_sha256']:raise ValueError('Ranking source bundle changed')
    values=arrays(data);profiles={}
    for name,profile in spec['profiles'].items():
        baseline=ranking['profiles'][name];support={};width=spec['window_pixels']
        original,_=windows_numpy(values,spec,profile['weights'])
        for key,site in baseline['sites'].items():
            best=site['best_window'];region=(slice((best['wy']-1)*width,best['wy']*width),slice((best['wx']-1)*width,best['wx']*width))
            fields=['landcover_valid_fraction','terrain_valid_fraction']
            if 'frozen' in profile['weights']:fields.append('permafrost_valid_fraction')
            support[key]={}
            for field in fields:
                a=values[key][field][region];finite=np.isfinite(a)
                support[key][field]={'mean_fraction':float(a[finite].mean()) if finite.any() else None,
                    'min_fraction':float(a[finite].min()) if finite.any() else None,
                    'pixels_with_coverage_at_least_0_9':int((finite&(a>=.9-TOL)).sum())}
        variants=[];baseline_top={r['site_key'] for r in baseline['ranked_patches'][:5]}
        for minimum in (.9,.99):
            strict=deepcopy(spec)
            strict['candidate_gate']['landcover_valid_fraction_min']=minimum
            strict['candidate_gate']['terrain_valid_fraction_min']=minimum
            for component in strict['components'].values():
                if 'coverage_field' in component:component['minimum_coverage']=minimum
            rows,_=windows_numpy(values,strict,profile['weights'])
            if any(b['score_lower']>a['score_lower']+TOL for a,b in zip(original,rows)):
                raise AssertionError('Requiring more support increased a supported window score')
            best=[r for r in diverse(rows) if r['score_lower']>TOL];top={r['site_key'] for r in best[:5]}
            variants.append({'minimum_source_coverage':minimum,'positive_windows':sum(r['score_lower']>TOL for r in rows),
                'positive_patches':len(best),'top5':[{'site_key':r['site_key'],'score_lower':r['score_lower']} for r in best[:5]],
                'baseline_top5_members_retained':len(top&baseline_top)})
        profiles[name]={'baseline_best_window_source_support':support,'stricter_coverage_variants':variants}
    report={'status':'passed','bundle_sha256':bundle,'spec_sha256':ranking['spec_sha256'],'profiles':profiles,
        'scope':'Offline quality-sensitivity calculation on the verified source grids; base SQL scores remain unchanged. Component evidence measures availability, not statistical reliability or within-pixel joint valid-area coverage.',
        'limits':['High source-valid-area fractions do not establish measurement accuracy or resolve coarse climate.',
                  'Strict coverage variants test fragility, not a recommendation to discard incomplete candidate areas.']}
    args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v['stricter_coverage_variants'] for k,v in profiles.items()},indent=2))


if __name__=='__main__':main()
