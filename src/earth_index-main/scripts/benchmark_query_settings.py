# /// script
# requires-python = ">=3.11"
# dependencies = ["psycopg[binary]==3.3.9", "h3==4.5.0", "rasterio==1.5.2", "numpy==2.5.4"]
# ///
"""Session-local setting and projection controls; persistent schema/data unchanged."""
import json
import re
import statistics
from datetime import datetime,timezone
from pathlib import Path
import time
from psycopg import sql
from load_raster_pilot import ROOT,connection

OUT=ROOT/'research/expansion-4km'
TARGET=next(t for t in json.loads((OUT/'validation/targets.json').read_text()) if t['target']['key']=='valles')
BUNDLE=json.loads((OUT/'database.json').read_text())
NS=BUNDLE['schema_name']
Q=re.sub(r'(?<!:):([A-Za-z_]\w*)',r'%(\1)s',TARGET['query']['query'])
PARAMS={k:float(v) for k,v in TARGET['query']['params'].items()}


def per_patch_query(site_filter=False,target=None):
    target=target or TARGET
    query_text=re.sub(r'(?<!:):([A-Za-z_]\w*)',r'%(\1)s',target['query']['query'])
    fields=sorted({c['key'] for c in target['criteria'] if c['enabled']}|{
        f['qualityKey'] for f in target['fields'] if f['qualityKey'] and any(c['enabled'] and c['key']==f['key'] for c in target['criteria'])})
    ids=','.join(f"max(layer_id) FILTER (WHERE feature_key='{f}') AS l{i}" for i,f in enumerate(fields))
    joins=' '.join(f'LEFT JOIN research.raster_tiles t{i} ON t{i}.layer_id=p.l{i}' for i in range(len(fields)))
    columns=','.join(f'ST_DumpValues(t{i}.rast,1,true) a{i},t{i}.scale s{i},t{i}.offset_value o{i}' for i in range(len(fields)))
    values=','.join(f'u.c{i}*d.s{i}+d.o{i} AS "{f}"' for i,f in enumerate(fields))
    expansions=','.join(['generate_series(1,d.width*d.height)']+[f'unnest(d.a{i})' for i in range(len(fields))])
    aliases=','.join(['pixel']+[f'c{i}' for i in range(len(fields))])
    predicate=' AND site_key=ANY(%(chunk_sites)s)' if site_filter else ''
    cte=f"""WITH packed AS MATERIALIZED (
      SELECT site_key,max((grid_definition->'shape'->>1)::int) width,
      max((grid_definition->'shape'->>0)::int) height,{ids}
      FROM research.raster_layers WHERE bundle_sha256='{BUNDLE['bundle_sha256']}'{predicate} GROUP BY site_key
    ), decoded AS MATERIALIZED (
      SELECT p.site_key,p.width,p.height,{columns} FROM packed p {joins}
    )"""
    source=f'SELECT d.site_key,mod(u.pixel-1,d.width)+1 AS x,((u.pixel-1)/d.width)+1 AS y,{values} FROM ROWS FROM ({expansions}) AS u({aliases})'
    query=query_text.replace('FROM cells','FROM ('+source+') one_patch')
    return cte+' SELECT r.site_key,r.scores FROM decoded d CROSS JOIN LATERAL ('+query.rstrip(';')+') r ORDER BY r.site_key'


def nodes(plan):
    yield plan
    for child in plan.get('Plans',[]):yield from nodes(child)


def main():
    configs={
      'defaults':dict(jit='on',work_mem='4MB',max_parallel_workers_per_gather='2'),
      'jit_off':dict(jit='off',work_mem='4MB',max_parallel_workers_per_gather='2'),
      'memory_64mb':dict(jit='on',work_mem='64MB',max_parallel_workers_per_gather='2'),
      'memory_64mb_jit_off':dict(jit='off',work_mem='64MB',max_parallel_workers_per_gather='2'),
      'parallel_4':dict(jit='off',work_mem='64MB',max_parallel_workers_per_gather='4',parallel_setup_cost='0',parallel_tuple_cost='0',min_parallel_table_scan_size='0',min_parallel_index_scan_size='0'),
    }
    report=dict(generated_at=datetime.now(timezone.utc).isoformat(),target=TARGET['target'],cells=380000,result_bytes=760000,paths={},settings=configs)
    with connection() as conn:
        names=['shared_buffers','effective_cache_size','work_mem','jit','max_parallel_workers_per_gather','max_parallel_workers','max_worker_processes']
        report['actual_settings']={name:conn.execute('SELECT current_setting(%s)',(name,)).fetchone()[0] for name in names}
        conn.prepare_threshold=None
        conn.execute("SELECT set_config('application_name','earth_query_settings_benchmark',true)")
        current=Q.replace('FROM cells',f'FROM "{NS}".cells')
        targeted=per_patch_query()
        conn.execute('SET LOCAL ROLE earth_index_reader');conn.execute("SET LOCAL statement_timeout='30s'")
        baseline=conn.execute(current,PARAMS).fetchall()
        assert conn.execute(targeted,PARAMS).fetchall()==baseline,'Chunked query changed score bytes'
        queries={'current':current,'required_bands_per_patch':targeted}
        # Rotate paired query/config ordering; 3 samples each. LOCAL settings reset
        # after transaction. Aggressive parallel costs are diagnostic only.
        for iteration in range(3):
            settings=list(configs);settings=settings[iteration:]+settings[:iteration]
            for config in settings:
                reset=dict(jit='on',work_mem='4MB',max_parallel_workers_per_gather='2',parallel_setup_cost='1000',parallel_tuple_cost='.1',min_parallel_table_scan_size='8MB',min_parallel_index_scan_size='512kB')
                for name,value in (reset|configs[config]).items():conn.execute('SELECT set_config(%s,%s,true)',(name,value))
                order=list(queries) if iteration%2==0 else list(reversed(queries))
                for name in order:
                    key=name+'/'+config;before=time.perf_counter()
                    plan=conn.execute('EXPLAIN (ANALYZE, BUFFERS, VERBOSE, FORMAT JSON, SETTINGS) '+queries[name],PARAMS).fetchone()[0][0]
                    row=report['paths'].setdefault(key,dict(samples_ms=[]))
                    row['samples_ms'].append(plan['Execution Time']);row['median_ms']=statistics.median(row['samples_ms'])
                    row['last_plan']=plan
                    row['workers_launched']=sum(n.get('Workers Launched',0) for n in nodes(plan['Plan']))
                    row['temp_written_blocks']=plan['Plan'].get('Temp Written Blocks',0)
                    row['jit_ms']=plan.get('JIT',{}).get('Timing',{}).get('Total',0)
                    print(iteration+1,key,round(plan['Execution Time'],1),'ms',flush=True)
        # Exact score bytes also checked under changed settings, not just baseline.
        assert conn.execute(targeted,PARAMS).fetchall()==baseline
        assert conn.execute(current,PARAMS).fetchall()==baseline
    report['method']='380k cells, one frozen target, 3 rotated paired samples, warm server EXPLAIN ANALYZE. SET LOCAL only; no persistent settings/schema/data changes. Exact packed score equality checked.'
    (OUT/'query-settings.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:{f:v[f] for f in ('median_ms','workers_launched','temp_written_blocks','jit_ms')} for k,v in report['paths'].items()},indent=2))


if __name__=='__main__':main()
