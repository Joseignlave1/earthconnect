# /// script
# requires-python = ">=3.11"
# dependencies = ["psycopg[binary]==3.3.9", "h3==4.5.0", "rasterio==1.5.2", "numpy==2.5.4"]
# ///
"""Read-only chunk scheduling experiment; no production query or server changes."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from datetime import datetime,timezone
import json
import statistics
import time
from benchmark_query_settings import OUT,NS,TARGET,PARAMS,per_patch_query,connection


def main():
    report=dict(generated_at=datetime.now(timezone.utc).isoformat(),target=TARGET['target'],cells=380000,paths={})
    with ExitStack() as stack:
        connections=[stack.enter_context(connection()) for _ in range(8)]
        for conn in connections:
            conn.prepare_threshold=None;conn.read_only=True
            conn.execute('SET LOCAL ROLE earth_index_reader')
            conn.execute("SET LOCAL jit=off; SET LOCAL work_mem='4MB'; SET LOCAL max_parallel_workers_per_gather=0; SET LOCAL statement_timeout='15s'")
        sites=[r[0] for r in connections[0].execute(f'SELECT DISTINCT site_key FROM "{NS}".layers ORDER BY site_key')]
        expected={s:(OUT/'validation'/f'expanded-valles-{s}.bin').read_bytes() for s in sites}
        query=per_patch_query(site_filter=True)
        def job(job):
            conn,keys=job
            return conn.execute(query,PARAMS|dict(chunk_sites=keys)).fetchall()
        def run(workers):
            batches=[sites[i::workers] for i in range(workers)]
            start=time.perf_counter()
            with ThreadPoolExecutor(max_workers=workers) as pool:
                groups=list(pool.map(job,zip(connections,batches)))
            elapsed=(time.perf_counter()-start)*1000
            rows={s:bytes(b) for group in groups for s,b in group}
            assert rows==expected,'Chunking changed scores/addresses'
            return elapsed
        for workers in (1,2,4,8):run(workers)
        samples={workers:[] for workers in (1,2,4,8)}
        for iteration in range(3):
            order=[1,2,4,8];order=order[iteration:]+order[:iteration]
            for workers in order:
                milliseconds=run(workers);samples[workers].append(milliseconds)
                print(iteration+1,workers,'workers',round(milliseconds,1),'ms',flush=True)
        for workers,values in samples.items():report['paths'][str(workers)]={'samples_ms':values,'median_ms':statistics.median(values)}
        validation={}
        for case in json.loads((OUT/'validation/targets.json').read_text()):
            target_query=per_patch_query(site_filter=True,target=case)
            target_params={k:float(v) for k,v in case['query']['params'].items()}
            batches=[sites[i::8] for i in range(8)]
            def case_job(job):
                conn,keys=job
                return conn.execute(target_query,target_params|dict(chunk_sites=keys)).fetchall()
            start=time.perf_counter()
            with ThreadPoolExecutor(max_workers=8) as pool:groups=list(pool.map(case_job,zip(connections,batches)))
            elapsed=(time.perf_counter()-start)*1000
            rows={s:bytes(b) for group in groups for s,b in group}
            expected_case={s:(OUT/'validation'/f"expanded-{case['target']['key']}-{s}.bin").read_bytes() for s in sites}
            assert rows==expected_case,case['target']['key']
            validation[case['target']['key']]={'cells':380000,'bytes':760000,'wall_ms':elapsed,'exact':True}
        report['all_targets_verified']=validation

    report['method']='Warm pooled connections, 3 rotated-order client wall-time samples; one frozen 380k-cell target, 38 existing patches distributed across 1/2/4/8 read-only connections. JIT off, work_mem 4MB, internal query parallelism off. Exact 760kB output compared with original API scores on every run. Connection setup and gRPC excluded. Not a generic arbitrary-SQL splitter.'
    (OUT/'query-chunks.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['paths'],indent=2))


if __name__=='__main__':main()
