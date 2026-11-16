# /// script
# requires-python = ">=3.11"
# dependencies = ["duckdb==2.0.0", "pyarrow==26.0.1", "numpy==2.5.4", "rasterio==1.5.2", "psycopg[binary]==3.3.9", "h3==4.5.0"]
# ///
"""Compare persisted physical DOUBLE columns with the quantized U16 layout.

uv run scripts/benchmark_duckdb_double.py
Creates a separate benchmark file. No source or production changes.
"""
import argparse
from datetime import datetime, timezone
import json
import statistics
import time

import duckdb
import numpy as np

import benchmark_duckdb_scoring as base


OUT = base.WORLD / 'duckdb-double'
DB = OUT / 'world-double.duckdb'


def configure(conn):
    base.settings(conn, 8)
    conn.execute("SET temp_directory=''")


def build():
    started = time.perf_counter()
    identity = json.dumps(dict(layout='physical-double-v1',
                               source=base.sha((base.OUT / 'schema.sql').read_bytes()),
                               bundle=json.loads((base.OUT / 'load.json').read_text())['bundle_sha256']),
                          sort_keys=True)
    conn = duckdb.connect(str(DB))
    configure(conn)
    exists = conn.execute("SELECT count(*) FROM information_schema.tables WHERE table_name='layout_identity'").fetchall()[0][0]
    if exists:
        assert conn.execute('SELECT identity FROM layout_identity').fetchall()[0][0] == identity
        conn.close()
        return
    conn.execute("ATTACH '"+str(base.DB).replace("'", "''")+"' AS source_db (READ_ONLY)")
    print('Materializing all 25 physical-value columns as DOUBLE with NULL missing values', flush=True)
    conn.execute('BEGIN')
    try:
        conn.execute('CREATE TABLE cells AS SELECT * FROM source_db.cells')
        conn.execute('CREATE TABLE regions AS SELECT * FROM source_db.regions')
        conn.execute('CREATE TABLE layout_identity (identity VARCHAR)')
        conn.execute('INSERT INTO layout_identity VALUES (?)', [identity])
        conn.execute('COMMIT')
    except BaseException:
        conn.execute('ROLLBACK')
        raise
    copy_seconds = time.perf_counter()-started
    types = dict((row[0], row[1]) for row in conn.execute('DESCRIBE cells').fetchall())
    assert types.pop('region_id') == 'USMALLINT'
    assert types.pop('pixel') == 'UINTEGER'
    assert len(types) == 25 and set(types.values()) == {'DOUBLE'}
    # Checks every decoded column and NULL status, independently of the score.
    args = ','.join(base.ident(key) for key in ['region_id', 'pixel'] + sorted(types))
    before = time.perf_counter()
    reference = conn.execute(f'SELECT count(*),bit_xor(hash({args})) FROM source_db.cells').fetchall()[0]
    actual = conn.execute(f'SELECT count(*),bit_xor(hash({args})) FROM cells').fetchall()[0]
    assert actual == reference and actual[0] == base.ROWS
    check_seconds = time.perf_counter()-before
    conn.execute('DETACH source_db')
    conn.execute('CHECKPOINT')
    conn.close()
    base.atomic_json(OUT/'load.json', dict(status='completed', rows=base.ROWS, fields=25,
                     source_bytes=base.DB.stat().st_size, double_bytes=DB.stat().st_size,
                     build_and_checkpoint_seconds=time.perf_counter()-started,
                     materialize_seconds=copy_seconds, all_column_checksum_seconds=check_seconds,
                     all_column_checksum=str(actual[1]), all_column_checksums_match=True,
                     note='Decoded copy of already quantized measurements; no original source precision is recovered.'))
    print('DOUBLE file ready:', DB.stat().st_size, 'bytes', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=5)
    parser.add_argument('--build-only', action='store_true')
    args = parser.parse_args()
    assert args.samples > 0
    OUT.mkdir(parents=True, exist_ok=True)
    source_stat = base.DB.stat()
    build()
    if args.build_only:
        return
    query = base.query_for(base.TARGET)
    params = base.params_for(base.TARGET)
    (OUT/'query.sql').write_text(query+';\n')
    (OUT/'params.json').write_text(json.dumps(params, indent=2)+'\n')
    report = dict(generated_at=datetime.now(timezone.utc).isoformat(), duckdb_version=duckdb.__version__,
                  rows=base.ROWS, fields=25, threads=8, samples_per_path=args.samples,
                  source_file_bytes=base.DB.stat().st_size, double_file_bytes=DB.stat().st_size,
                  paths={}, paired_runs=[], other_targets=[],
                  method='File-backed, warm-cache comparison. Eight threads and 8 GB engine limit per layout, '
                         'disk spilling disabled. Identical score SQL and DOUBLE intermediate arithmetic. '
                         'U16 view performs decode at query time; DOUBLE table stores physical values and NULL. '
                         'Alternate layout order between paired runs. Full-output timings include scoring, Arrow fetch, '
                         'address scatter and all 71.1 MB U16 packing. Separate compute-plus-checksum timings '
                         'omit full result export and are not full-output latency. No Rust/API/browser/network timing. '
                         'Every full-world score code checked against original PostgreSQL references. '
                         'Data remain at the original import quantization precision.')
    conns = {key: duckdb.connect(str(path), read_only=True) for key, path in [('uint16', base.DB), ('double', DB)]}
    sites = [row[0] for row in conns['uint16'].execute('SELECT site_key FROM regions ORDER BY region_id').fetchall()]
    for key, conn in conns.items():
        configure(conn)
        print('Warming', key, 'and checking the complete score output', flush=True)
        payload, timing = base.output(conn, query, params)
        base.verify(payload, sites)
        expected_sum = int(np.frombuffer(payload, dtype='<u2').astype(np.uint64).sum())
        report['paths'][key] = dict(warmup=timing, samples=[], compute_samples_seconds=[], exact=True)
        del payload
    for iteration in range(args.samples):
        order = ['uint16', 'double'] if iteration % 2 == 0 else ['double', 'uint16']
        pair = {}
        for key in order:
            payload, timing = base.output(conns[key], query, params)
            base.verify(payload, sites)
            assert timing['rows'] == base.ROWS
            timing['exact'] = True
            report['paths'][key]['samples'].append(timing)
            pair[key] = timing['total_seconds']
            del payload
            print(key, 'full output', iteration+1, round(timing['total_seconds'], 4), 'seconds; exact', flush=True)
        report['paired_runs'].append(pair)
        base.atomic_json(OUT/'benchmark.json', report)
    compute = 'SELECT count(*),sum(code::UBIGINT) FROM ('+query+') scored_output'
    for iteration in range(args.samples):
        order = ['double', 'uint16'] if iteration % 2 == 0 else ['uint16', 'double']
        for key in order:
            started = time.perf_counter()
            result = conns[key].execute(compute, params).fetchall()[0]
            elapsed = time.perf_counter()-started
            assert result == (base.ROWS, expected_sum)
            report['paths'][key]['compute_samples_seconds'].append(elapsed)
            print(key, 'compute + checksum', iteration+1, round(elapsed, 4), 'seconds', flush=True)
    ids = [sites.index(site) for site in ('h08v05', 'h11v11', 'h15v01')]
    for target in base.TARGETS:
        table = conns['double'].sql(base.query_for(target, regions=True),
            params=base.params_for(target) | {'region_ids': ids}).to_arrow_table()
        addresses, values = table['address'].to_numpy(), table['code'].to_numpy()
        for region_id in ids:
            selected = addresses // base.PIXELS == region_id
            output = np.empty(base.PIXELS, dtype='<u2')
            output[addresses[selected] % base.PIXELS] = values[selected]
            expected = (base.WORLD/'validation'/f'{target["target"]["key"]}--{sites[region_id]}.bin').read_bytes()
            assert output.tobytes() == expected
        report['other_targets'].append(dict(target=target['target']['key'], score_codes=3*base.PIXELS, exact=True))
    for key, conn in conns.items():
        row = report['paths'][key]
        row['median_seconds'] = statistics.median(value['total_seconds'] for value in row['samples'])
        row['compute_median_seconds'] = statistics.median(row['compute_samples_seconds'])
        row['engine_tracked_bytes'] = conn.execute('SELECT sum(memory_usage_bytes) FROM duckdb_memory()').fetchall()[0][0]
        row['plan'] = conn.execute('EXPLAIN '+query, params).fetchall()
        (OUT/f'plan-{key}.txt').write_text('\n'.join(item[1] for item in row['plan'])+'\n')
        conn.close()
    assert (base.DB.stat().st_size, base.DB.stat().st_mtime_ns) == (source_stat.st_size, source_stat.st_mtime_ns)
    report['status'] = 'completed'
    base.atomic_json(OUT/'benchmark.json', report)
    print('Completed:', OUT/'benchmark.json', flush=True)


if __name__ == '__main__':
    main()
