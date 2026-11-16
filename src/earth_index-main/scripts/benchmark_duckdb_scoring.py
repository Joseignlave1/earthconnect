# /// script
# requires-python = ">=3.11"
# dependencies = ["duckdb==2.0.0", "pyarrow==26.0.1", "numpy==2.5.4", "rasterio==1.5.2", "psycopg[binary]==3.3.9", "h3==4.5.0"]
# ///
"""Separate, resumable DuckDB experiment over the immutable global U16 data.

uv run scripts/benchmark_duckdb_scoring.py
No PostgreSQL schema, API, demo, or scientific source files are modified.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import resource
import statistics
import time

import duckdb
import numpy as np
import pyarrow as pa
import rasterio

import benchmark_query_settings as pg


WORLD = pg.ROOT / 'research/world-4km'
OUT = WORLD / 'duckdb-scoring'
DB = OUT / 'world.duckdb'
PIXELS = 278 * 278
ROWS = 460 * PIXELS
TARGETS = json.loads((WORLD.parent / 'expansion-4km/validation/targets.json').read_text())
TARGET = next(t for t in TARGETS if t['target']['key'] == 'valles')
REFERENCE = json.loads((WORLD / 'scoring-benchmark.json').read_text())['paths']['production_sql']['all_regions_sha256']


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.part')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def ident(name):
    return '"' + name.replace('"', '""') + '"'


def query_for(target, regions=False):
    # Retain the builder's complete score expression and parameter arithmetic.
    # PostgreSQL float8 round uses ties-to-even; DuckDB round() differs at ties.
    score_cte = target['query']['query'].split('\nSELECT site_key,')[0]
    score_cte = score_cte.replace('SELECT site_key, x, y,', 'SELECT region_id, pixel,')
    if regions:
        score_cte = score_cte.replace('FROM cells', 'FROM cells WHERE region_id=ANY($region_ids::USMALLINT[])')
    score_cte = re.sub(r'(?<!:):([A-Za-z_]\w*)', r'$\1', score_cte)
    return score_cte + f"""
SELECT region_id::UINTEGER * {PIXELS} + pixel AS address,
       CASE WHEN score IS NULL THEN 65535::USMALLINT
            ELSE round_even(65534 * greatest(0.0, least(1.0, score)), 0)::USMALLINT
       END AS code
FROM scored"""


def settings(conn, threads=8):
    conn.execute(f'SET threads={threads}')
    conn.execute("SET memory_limit='8GB'")
    conn.execute('SET enable_progress_bar=false')


def source_table(item, features, groups):
    index, key = item
    arrays = [pa.array(np.full(PIXELS, index, dtype=np.uint16)),
              pa.array(np.arange(PIXELS, dtype=np.uint32))]
    for feature in features:
        layer = groups[key][feature]
        with rasterio.open(pg.ROOT / layer['path']) as src:
            codes = src.read(1)
            assert codes.dtype == np.dtype('uint16') and codes.shape == (278, 278)
            assert src.nodata == 0
        codes = codes.astype('<u2', copy=False).reshape(-1)
        assert sha(codes.tobytes()) == layer['codes_sha256'], (key, feature)
        arrays.append(pa.array(codes))
    return pa.Table.from_arrays(arrays, names=['region_id', 'pixel'] + features)


def load():
    start = time.perf_counter()
    manifest = json.loads((WORLD / 'manifest.json').read_text())
    publication = json.loads((WORLD / 'database.json').read_text())
    groups = {}
    encodings = {}
    for layer in manifest['layers']:
        groups.setdefault(layer['site_key'], {})[layer['feature_key']] = layer
        encodings.setdefault(layer['feature_key'], set()).add((layer['scale'], layer['offset']))
    assert len(groups) == 460 and len(encodings) == 25
    assert all(len(values) == 1 for values in encodings.values())
    features = sorted(encodings)
    encoding = {f: list(next(iter(encodings[f]))) for f in features}
    sites = sorted(groups)
    identity = json.dumps(dict(format_version=1, bundle=publication['bundle_sha256'],
                               encoding=encoding, sites=sites), sort_keys=True)
    conn = duckdb.connect(str(DB))
    settings(conn)
    conn.execute("SET checkpoint_threshold='256MB'")
    exists = conn.execute("SELECT count(*) FROM information_schema.tables WHERE table_name='benchmark_identity'").fetchone()[0]
    if not exists:
        columns = ', '.join(ident(f) + ' USMALLINT NOT NULL' for f in features)
        decoded = ',\n       '.join(
            f'CASE WHEN {ident(f)}=0 THEN NULL ELSE {ident(f)}::DOUBLE '
            f'* {repr(float(encoding[f][0]))}::DOUBLE + {repr(float(encoding[f][1]))}::DOUBLE END AS {ident(f)}'
            for f in features)
        schema = f"""CREATE TABLE benchmark_identity (identity VARCHAR NOT NULL);
CREATE TABLE regions (region_id USMALLINT, site_key VARCHAR);
CREATE TABLE codes (region_id USMALLINT NOT NULL, pixel UINTEGER NOT NULL, {columns});
CREATE VIEW cells AS SELECT region_id, pixel,
       {decoded}
FROM codes;
"""
        conn.execute('BEGIN')
        conn.execute(schema)
        conn.execute('INSERT INTO benchmark_identity VALUES (?)', [identity])
        conn.execute('COMMIT')
        (OUT / 'schema.sql').write_text(schema)
    assert conn.execute('SELECT identity FROM benchmark_identity').fetchone()[0] == identity
    loaded = conn.execute('SELECT region_id,site_key FROM regions ORDER BY region_id').fetchall()
    assert loaded == list(enumerate(sites[:len(loaded)]))
    report_path = OUT / 'load.json'
    report = json.loads(report_path.read_text()) if report_path.exists() else dict(
        bundle_sha256=publication['bundle_sha256'], source_read_seconds=0.0,
        insert_seconds=0.0, roundtrip_check_seconds=0.0, verified_codes=0)
    resumed = len(loaded)
    print('DuckDB', duckdb.__version__, '- importing all 25 U16 columns; resumed regions:', resumed, flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        for first in range(resumed, len(sites), 8):
            keys = list(enumerate(sites))[first:first+8]
            before = time.perf_counter()
            incoming = pa.concat_tables(list(pool.map(lambda item: source_table(item, features, groups), keys)))
            report['source_read_seconds'] += time.perf_counter() - before
            conn.register('incoming', incoming)
            before = time.perf_counter()
            conn.execute('BEGIN')
            try:
                conn.execute('INSERT INTO codes SELECT * FROM incoming')
                report['insert_seconds'] += time.perf_counter() - before
                before = time.perf_counter()
                readback = conn.sql('SELECT * FROM codes WHERE region_id BETWEEN ? AND ? ORDER BY region_id,pixel',
                                    params=[keys[0][0], keys[-1][0]]).to_arrow_table().combine_chunks()
                assert readback.num_rows == len(keys) * PIXELS
                for position, (region_id, key) in enumerate(keys):
                    part = readback.slice(position * PIXELS, PIXELS)
                    assert np.all(part['region_id'].to_numpy() == region_id)
                    assert np.array_equal(part['pixel'].to_numpy(), np.arange(PIXELS, dtype=np.uint32))
                    for feature in features:
                        assert sha(part[feature].to_numpy().astype('<u2', copy=False).tobytes()) == groups[key][feature]['codes_sha256']
                report['roundtrip_check_seconds'] += time.perf_counter() - before
                conn.executemany('INSERT INTO regions VALUES (?,?)', keys)
                conn.execute('COMMIT')
            except BaseException:
                conn.execute('ROLLBACK')
                raise
            finally:
                conn.unregister('incoming')
            report.update(loaded_regions=first+len(keys), verified_codes=(first+len(keys))*PIXELS*25,
                          elapsed_this_run_seconds=time.perf_counter()-start, status='loading')
            atomic_json(report_path, report)
            print('Loaded / byte-verified', report['loaded_regions'], '/ 460 regions', flush=True)
            del incoming, readback
    assert conn.execute('SELECT count(*) FROM codes').fetchone()[0] == ROWS
    conn.execute('CHECKPOINT')
    report.update(status='completed', rows=ROWS, fields=25, stored_type='USMALLINT',
                  file_bytes=DB.stat().st_size, resumed_regions=resumed,
                  elapsed_this_run_seconds=time.perf_counter()-start)
    atomic_json(report_path, report)
    conn.close()
    return sites


def params_for(target):
    return {key: float(value) for key, value in target['query']['params'].items()}


def output(conn, query, params):
    """Fetch unsorted Arrow vectors and scatter into the known row-major grid.

    The timed result is exactly 2 bytes per address. No row-by-row Python calls,
    SQL global sort, cached scores, precomputed preferences, or excluded packing.
    """
    started = time.perf_counter()
    scores = np.empty(ROWS, dtype='<u2')
    reader = conn.sql(query, params=params).to_arrow_reader(batch_size=1 << 20)
    engine_and_fetch = time.perf_counter() - started
    packing = 0.0
    seen = 0
    while True:
        before = time.perf_counter()
        try:
            batch = next(reader)
        except StopIteration:
            engine_and_fetch += time.perf_counter() - before
            break
        engine_and_fetch += time.perf_counter() - before
        before = time.perf_counter()
        addresses = batch.column(0).to_numpy(zero_copy_only=True)
        values = batch.column(1).to_numpy(zero_copy_only=True)
        scores[addresses] = values
        seen += len(values)
        packing += time.perf_counter() - before
    before = time.perf_counter()
    payload = scores.tobytes()
    packing += time.perf_counter() - before
    elapsed = time.perf_counter() - started
    return payload, dict(total_seconds=elapsed, engine_and_arrow_fetch_seconds=engine_and_fetch,
                         scatter_and_packing_seconds=packing, rows=seen, bytes=len(payload))


def verify(payload, sites):
    assert len(payload) == ROWS*2
    for i, site in enumerate(sites):
        assert sha(payload[i*PIXELS*2:(i+1)*PIXELS*2]) == REFERENCE[site], site


def postgres_controls(samples, sites):
    publication = json.loads((WORLD / 'database.json').read_text())
    pg.BUNDLE, pg.NS = publication, publication['schema_name']
    query = pg.per_patch_query(site_filter=True).replace('WITH scored AS (', 'WITH scored AS MATERIALIZED (')
    batches = [sites[i:i+2] for i in range(0, len(sites), 2)]
    measurements = []
    with ExitStack() as stack:
        connections = [stack.enter_context(pg.connection()) for _ in range(8)]
        for conn in connections:
            conn.prepare_threshold = None
            conn.read_only = True
            conn.execute("SET LOCAL ROLE earth_index_reader; SET LOCAL jit=off; SET LOCAL work_mem='64MB'; SET LOCAL max_parallel_workers_per_gather=0")
        def job(item):
            conn, groups = item
            return [row for keys in groups for row in conn.execute(query, params_for(TARGET) | {'chunk_sites': keys}).fetchall()]
        for iteration in range(samples):
            started = time.perf_counter()
            with ThreadPoolExecutor(max_workers=8) as pool:
                groups = list(pool.map(job, [(conn, batches[i::8]) for i, conn in enumerate(connections)]))
            elapsed = time.perf_counter()-started
            actual = {site: sha(bytes(value)) for group in groups for site, value in group}
            assert actual == REFERENCE
            measurements.append(elapsed)
            print('PostgreSQL eight-connection control', iteration+1, round(elapsed, 4), 'seconds; exact', flush=True)
    return dict(samples_seconds=measurements, median_seconds=statistics.median(measurements), exact=True,
                method='Existing optimized narrow-band row query, 8 read-only connections, 2 regions per statement; includes SQL U16 packing and local PostgreSQL socket fetch.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load-only', action='store_true')
    parser.add_argument('--postgres', action='store_true')
    parser.add_argument('--samples', type=int, default=3)
    parser.add_argument('--resume-measurements', action='store_true',
                        help='Keep already validated timing samples when SQL and parameters match exactly')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    sites = load()
    if args.load_only:
        return
    query = query_for(TARGET)
    if args.resume_measurements:
        assert (OUT / 'query.sql').read_text() == query+';\n'
        assert json.loads((OUT / 'params.json').read_text()) == params_for(TARGET)
    (OUT / 'query.sql').write_text(query+';\n')
    (OUT / 'params.json').write_text(json.dumps(params_for(TARGET), indent=2)+'\n')
    report = dict(generated_at=datetime.now(timezone.utc).isoformat(), duckdb_version=duckdb.__version__,
                  pyarrow_version=pa.__version__, rows=ROWS, domain_cells=31881940, fields=25,
                  target='valles', paths={}, compute_only=[], other_targets=[],
                  method='Warm local full-world query; USMALLINT source columns decoded on demand via SQL view. '
                         'All original cells including padding retained. Named parameters and original score formula; '
                         'ties-to-even rounding matches PostgreSQL float8. Full-output timing includes Arrow fetch, '
                         'address scatter and final little-endian U16 serialization. No gRPC/browser/network timing. '
                         'Checksum-only timing has no full result transfer and is reported separately. No production changes.')
    report_path = OUT / 'benchmark.json'
    if args.resume_measurements:
        report = json.loads(report_path.read_text())
        assert report['duckdb_version'] == duckdb.__version__
        assert all(report['paths'][str(n)]['exact'] and len(report['paths'][str(n)]['samples']) >= args.samples
                   for n in (1, 8, 16))
    conn = duckdb.connect(str(DB), read_only=True)
    settings(conn, 8)
    print('Warming complete score query and checking every global output code', flush=True)
    payload, warm = output(conn, query, params_for(TARGET))
    assert warm['rows'] == ROWS
    verify(payload, sites)
    report.setdefault('initial_run', warm | {'exact': True})
    expected_sum = int(np.frombuffer(payload, dtype='<u2').astype(np.uint64).sum())
    report['expected_code_sum'] = expected_sum
    del payload
    atomic_json(report_path, report)
    for iteration in ([] if args.resume_measurements else range(args.samples)):
        order = [1, 8, 16]
        order = order[iteration % 3:] + order[:iteration % 3]
        for threads in order:
            settings(conn, threads)
            payload, measurement = output(conn, query, params_for(TARGET))
            assert measurement['rows'] == ROWS
            verify(payload, sites)
            row = report['paths'].setdefault(str(threads), dict(threads=threads, samples=[], exact=True))
            row['samples'].append(measurement)
            row['median_seconds'] = statistics.median(v['total_seconds'] for v in row['samples'])
            atomic_json(report_path, report)
            print('DuckDB', threads, 'threads run', iteration+1, json.dumps(measurement), 'exact', flush=True)
            del payload
    settings(conn, 8)
    compute = 'SELECT count(*),sum(code::UBIGINT) FROM ('+query+') result'
    for iteration in range(args.samples):
        start = time.perf_counter()
        # Drain the DB-API result fully before switching to the relation API.
        summary = conn.execute(compute, params_for(TARGET)).fetchall()[0]
        elapsed = time.perf_counter()-start
        assert summary == (ROWS, expected_sum), summary
        report['compute_only'].append(elapsed)
        atomic_json(report_path, report)
        print('DuckDB 8-thread compute + checksum', round(elapsed, 4), 'seconds', flush=True)
    # Reuse the independent fixtures already checked against deployed PostgreSQL.
    chosen_sites = ['h08v05', 'h11v11', 'h15v01']
    ids = [sites.index(site) for site in chosen_sites]
    for target in TARGETS:
        rows = conn.sql(query_for(target, regions=True), params=params_for(target) | {'region_ids': ids}).to_arrow_table()
        addresses = rows['address'].to_numpy()
        values = rows['code'].to_numpy()
        for site, region_id in zip(chosen_sites, ids):
            mask = addresses // PIXELS == region_id
            local = np.empty(PIXELS, dtype='<u2')
            local[addresses[mask] % PIXELS] = values[mask]
            expected = (WORLD / 'validation' / f'{target["target"]["key"]}--{site}.bin').read_bytes()
            assert local.tobytes() == expected, (target['target']['key'], site)
        report['other_targets'].append(dict(target=target['target']['key'], score_codes=3*PIXELS, exact=True))
    report['plan'] = conn.execute('EXPLAIN '+query, params_for(TARGET)).fetchall()
    report['process_peak_rss_bytes_including_import'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    report['database_file_bytes'] = DB.stat().st_size
    atomic_json(report_path, report)
    conn.close()
    if args.postgres:
        report['postgres_8_connections'] = postgres_controls(args.samples, sites)
    report['status'] = 'completed'
    atomic_json(report_path, report)
    print('Completed:', report_path, flush=True)


if __name__ == '__main__':
    main()
