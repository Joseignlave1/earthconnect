# /// script
# requires-python = ">=3.11"
# dependencies = ["psycopg[binary]==3.3.9", "h3==4.5.0", "rasterio==1.5.2", "numpy==2.5.4"]
# ///
"""Isolate current demo SQL stages using disposable temp relations; no production changes."""
import json
from pathlib import Path
import re
import statistics
from psycopg import sql
from load_raster_pilot import ROOT,connection

out=ROOT/'research/expansion-4km'
target=next(t for t in json.loads((out/'validation/targets.json').read_text()) if t['target']['key']=='valles')
ns=json.loads((out/'database.json').read_text())['schema_name']
query=re.sub(r'(?<!:):([A-Za-z_]\w*)',r'%(\1)s',target['query']['query'])
params={k:float(v) for k,v in target['query']['params'].items()}
marker='\nSELECT site_key,\n'
assert query.count(marker)==1
prefix,suffix=query.split(marker)
source=f'"{ns}".cells'
full=query.replace('FROM cells','FROM '+source)
scored=prefix.replace('FROM cells','FROM '+source)
fields=sorted({c['key'] for c in target['criteria'] if c['enabled']}|{f['qualityKey'] for f in target['fields'] if f['qualityKey'] and any(c['enabled'] and c['key']==f['key'] for c in target['criteria'])})
select=','.join('"'+f+'"' for f in fields)
queries={
 'decode_checksum':f'SELECT '+','.join(f'sum(coalesce("{f}",0))' for f in fields)+f' FROM {source}',
 'decode_and_score_checksum':scored+' SELECT count(*),sum(score),count(score) FROM scored',
 'full_packed':full,
 'score_from_temp_decoded':prefix.replace('FROM cells','FROM pg_temp.query_decoded')+' SELECT count(*),sum(score),count(score) FROM scored',
 'pack_from_temp_scores':'SELECT site_key,\n'+suffix.replace('FROM scored','FROM pg_temp.query_scores'),
}
# A query-local control reads only required U16 bands; it stores no persistent
# decoded values, scores, user queries or preferred targets.
ids=','.join(f"max(layer_id) FILTER (WHERE feature_key='{f}') AS l{i}" for i,f in enumerate(fields))
joins=' '.join(f'LEFT JOIN research.raster_tiles t{i} ON t{i}.layer_id=p.l{i}' for i in range(len(fields)))
columns=','.join(f'ST_DumpValues(t{i}.rast,1,true) a{i},t{i}.scale s{i},t{i}.offset_value o{i}' for i in range(len(fields)))
values=','.join(f'u.c{i}*d.s{i}+d.o{i} AS "{f}"' for i,f in enumerate(fields))
expansions=','.join(['generate_series(1,d.width*d.height)']+[f'unnest(d.a{i})' for i in range(len(fields))])
aliases=','.join(['pixel']+[f'c{i}' for i in range(len(fields))])
bundle=json.loads((out/'database.json').read_text())['bundle_sha256']
source_control=f"""WITH packed AS MATERIALIZED (
 SELECT site_key,max((grid_definition->'shape'->>1)::int) width,
 max((grid_definition->'shape'->>0)::int) height,{ids}
 FROM research.raster_layers WHERE bundle_sha256='{bundle}' GROUP BY site_key
), decoded AS MATERIALIZED (
 SELECT p.site_key,p.width,p.height,{columns} FROM packed p {joins}
)
SELECT d.site_key,mod(u.pixel-1,d.width)+1 AS x,((u.pixel-1)/d.width)+1 AS y,{values}
FROM decoded d CROSS JOIN LATERAL ROWS FROM ({expansions}) AS u({aliases})"""
queries['full_required_raster_bands']=query.replace('FROM cells','FROM ('+source_control+') selected_bands')
queries['full_from_temp_decoded']=query.replace('FROM cells','FROM pg_temp.query_decoded')
cte=source_control.split('\nSELECT d.site_key,',1)[0]
per_patch_source=f'SELECT d.site_key,mod(u.pixel-1,d.width)+1 AS x,((u.pixel-1)/d.width)+1 AS y,{values} FROM ROWS FROM ({expansions}) AS u({aliases})'
per_patch_query=query.replace('FROM cells','FROM ('+per_patch_source+') one_patch')
queries['full_per_patch']=cte+' SELECT r.site_key,r.scores FROM decoded d CROSS JOIN LATERAL ('+per_patch_query.rstrip(';')+') r ORDER BY r.site_key'

report=dict(target=target['target'],fields=fields,cells=380000,paths={})
with connection() as conn:
 conn.execute("SET LOCAL statement_timeout='30s'")
 conn.execute('CREATE TEMP TABLE query_decoded ON COMMIT DROP AS SELECT site_key,x,y,'+select+' FROM '+source)
 conn.execute('CREATE TEMP TABLE query_scores ON COMMIT DROP AS '+scored+' SELECT * FROM scored',params)
 for name in ('query_decoded','query_scores'):
  conn.execute(f'ANALYZE pg_temp.{name}')
  conn.execute(f'GRANT SELECT ON pg_temp.{name} TO earth_index_reader')
 conn.execute('SET LOCAL ROLE earth_index_reader')
 baseline=conn.execute(full,params).fetchall()
 assert conn.execute(queries['pack_from_temp_scores']).fetchall()==baseline
 for name in ('full_required_raster_bands','full_from_temp_decoded','full_per_patch'):
  assert conn.execute(queries[name],params).fetchall()==baseline,name
 sums=conn.execute(queries['decode_and_score_checksum'],params).fetchone()
 assert conn.execute(queries['score_from_temp_decoded'],params).fetchone()==sums
 queries={k:v for k,v in queries.items() if k in ('full_packed','full_required_raster_bands','full_per_patch')}
 samples={k:[] for k in queries}
 for iteration in range(3):
  names=list(queries);names=names[iteration:]+names[:iteration]
  for name in names:
   plan=conn.execute('EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON, SETTINGS) '+queries[name],params if '%(' in queries[name] else None).fetchone()[0][0]
   samples[name].append(plan['Execution Time']);report['paths'][name]={'samples_ms':samples[name],'median_ms':statistics.median(samples[name]),'last_plan':plan}
 report['score_checksum']=[str(v) for v in sums]
 report['result_bytes']=sum(len(row[1]) for row in baseline)
report['method']='Warm server EXPLAIN ANALYZE, 3 rotated-order samples; exact same target and SQL; disposable temp materializations isolate stages. Stage timings are controls, not additive components. No production data/schema changes.'
(out/'query-profile-patch.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:{'median_ms':v['median_ms'],'samples_ms':v['samples_ms']} for k,v in report['paths'].items()},indent=2))
