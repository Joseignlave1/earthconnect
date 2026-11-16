# /// script
# requires-python = ">=3.11"
# dependencies = ["duckdb==2.0.0", "pyarrow==26.0.1", "numpy==2.5.4", "rasterio==1.5.2", "psycopg[binary]==3.3.9", "h3==4.5.0"]
# ///
"""Isolated-process comparison of warm file-backed and actual in-memory DuckDB.

uv run scripts/benchmark_duckdb_memory.py
Source file is read-only; each RAM copy is detached from the source before tests.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import time

import duckdb

import benchmark_duckdb_scoring as base


OUT = base.WORLD / 'duckdb-memory'
MODES = ['file', 'ram', 'ram_compressed']


def rss():
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmRSS:'):
            return int(line.split()[1]) * 1024
    raise RuntimeError('Missing process RSS')


def memory(conn):
    rows = conn.execute('SELECT tag,memory_usage_bytes,temporary_storage_bytes FROM duckdb_memory()').fetchall()
    return dict(process_rss_bytes=rss(), duckdb_tracked_bytes=sum(row[1] for row in rows),
                temporary_storage_bytes=sum(row[2] for row in rows),
                allocations=[dict(tag=row[0], bytes=row[1]) for row in rows if row[1]])


def worker(mode):
    report = dict(mode=mode, threads=8, duckdb_version=duckdb.__version__, initial_rss_bytes=rss())
    start = time.perf_counter()
    if mode == 'file':
        conn = duckdb.connect(str(base.DB), read_only=True)
        base.settings(conn, 8)
        conn.execute("SET temp_directory=''")
        features = [row[0] for row in conn.execute("DESCRIBE codes").fetchall() if row[0] not in ('region_id', 'pixel')]
        # Warm all 25 source columns, not just the score query's working set.
        conn.execute('SELECT '+','.join('sum('+base.ident(f)+'::UBIGINT)' for f in features)+' FROM codes').fetchall()
    else:
        conn = duckdb.connect(':memory:')
        base.settings(conn, 8)
        conn.execute("SET temp_directory=''")
        conn.execute("ATTACH '"+str(base.DB).replace("'", "''")+"' AS source_db (READ_ONLY)")
        option = ' (COMPRESS)' if mode == 'ram_compressed' else ''
        conn.execute("ATTACH ':memory:' AS ram_db"+option)
        conn.execute('COPY FROM DATABASE source_db TO ram_db')
        conn.execute('USE ram_db')
        if mode == 'ram_compressed':
            conn.execute('CHECKPOINT ram_db')
        conn.execute('DETACH source_db')
        assert all(row[1] is None for row in conn.execute('SELECT database_name,path FROM duckdb_databases()').fetchall())
    report['startup_or_prewarm_seconds'] = time.perf_counter()-start
    report['after_load'] = memory(conn)
    sites = [row[0] for row in conn.execute('SELECT site_key FROM regions ORDER BY region_id').fetchall()]
    assert len(sites) == 460
    assert conn.execute('SELECT count(*) FROM codes').fetchall()[0][0] == base.ROWS
    query = base.query_for(base.TARGET)
    payload, measurement = base.output(conn, query, base.params_for(base.TARGET))
    base.verify(payload, sites)
    report['warmup'] = measurement
    del payload
    report['after_warmup'] = memory(conn)
    report['samples'] = []
    for iteration in range(3):
        payload, measurement = base.output(conn, query, base.params_for(base.TARGET))
        base.verify(payload, sites)
        measurement['exact'] = True
        report['samples'].append(measurement)
        del payload
        print(mode, 'run', iteration+1, round(measurement['total_seconds'], 4), 'seconds; exact', flush=True)
    report['median_seconds'] = statistics.median(row['total_seconds'] for row in report['samples'])
    report['after_queries'] = memory(conn)
    report['peak_process_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    report['source_detached'] = mode != 'file'
    report['exact_all_scores'] = True
    assert report['after_queries']['temporary_storage_bytes'] == 0
    conn.close()
    base.atomic_json(OUT / f'{mode}.json', report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=MODES)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.mode:
        return worker(args.mode)
    before = base.DB.stat()
    for mode in MODES:
        print('Starting isolated benchmark:', mode, flush=True)
        subprocess.run([sys.executable, str(Path(__file__).resolve()), '--mode', mode], check=True)
    after = base.DB.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    result = dict(generated_at=datetime.now(timezone.utc).isoformat(), database_file_bytes=after.st_size,
                  rows=base.ROWS, fields=25, paths={mode: json.loads((OUT/f'{mode}.json').read_text()) for mode in MODES},
                  method='Separate Python process per mode; 8 DuckDB threads; 8 GB engine limit; '
                         'all 25 U16 columns copied to actual in-memory databases, then source DB detached. '
                         'COMPRESS plus an explicit CHECKPOINT uses DuckDB in-memory compression. File-backed control warms all columns. '
                         'No temp directory / spill permitted. Three warm full-output runs per mode; '
                         'includes scoring, Arrow fetch, address scatter and 71.1 MB U16 packing. '
                         'Every score code checked against original PostgreSQL hashes. '
                         'RSS includes Python, Arrow, engine, caches and transient output buffers; '
                         'DuckDB tracked bytes are reported separately. Source file unchanged.')
    base.atomic_json(OUT/'benchmark.json', result)
    print('Completed:', OUT/'benchmark.json', flush=True)


if __name__ == '__main__':
    main()
