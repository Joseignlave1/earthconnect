# /// script
# requires-python = ">=3.11"
# dependencies = ["psycopg[binary]==3.3.9", "h3==4.5.0", "rasterio==1.5.2", "numpy==2.5.4"]
# ///
"""Read-only raster-expression experiment; does not change the builder or DB.

uv run scripts/benchmark_raster_scoring.py
uv run scripts/benchmark_raster_scoring.py --global-workers 8
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import re
import statistics
import struct
import time

import numpy as np

import benchmark_query_settings as baseline


OUT = baseline.ROOT / 'research/world-4km/raster-scoring'
WORLD = json.loads((OUT.parent / 'database.json').read_text())
TARGET = baseline.TARGET
PARAMS = baseline.PARAMS
SITES = ['h08v05', 'h11v11', 'h15v01']


def literal(value):
    return "'" + value.replace("'", "''") + "'"


def formatted(expression, arguments):
    return f"format({literal(expression)}, {', '.join(arguments)})"


def binary(a, b, expression, arguments=(), pixeltype='64BF'):
    expr = formatted(expression, arguments) if arguments else literal(expression)
    return (f"ST_MapAlgebra({a}, 1, {b}, 1, {expr}, '{pixeltype}', "
            "'FIRST', NULL::text, NULL::text, NULL::double precision)")


def raster_query(target=TARGET):
    """Compile the builder's pointwise rules into chained raster expressions.

    Every intermediate is float64 with NODATA -1 (losses are nonnegative).
    Only the final score is uint16, with NODATA 65535 and valid zero scores.
    Parameter values enter expression text through PostgreSQL format('%L').
    """
    fields = {field['key']: field for field in target['fields']}
    active = [(i, c) for i, c in enumerate(target['criteria'])
              if c['enabled'] or c.get('min') is not None or c.get('max') is not None]
    scored = [(i, c) for i, c in active if c['enabled']]
    quality = sorted({fields[c['key']]['qualityKey'] for _, c in active
                      if fields[c['key']]['qualityKey']})
    required = sorted({c['key'] for _, c in active} | set(quality))
    aliases = {key: f'b{i}' for i, key in enumerate(required)}
    groups = {}
    for i, criterion in scored:
        field = fields[criterion['key']]
        groups.setdefault(field.get('group') or field['key'], []).append((i, criterion))

    def normalization(members):
        winner = max(members, key=lambda item: item[1]['weight'])[0]
        return (f':weight_{winner}::double precision',
                '(' + ' + '.join(f':weight_{i}::double precision' for i, _ in members) + ')')

    leaves = []
    for i, criterion in scored:
        field = fields[criterion['key']]
        alias = aliases[field['key']]
        arguments = [f':weight_{i}::double precision', f'{alias}.scale',
                     f'{alias}.offset_value', f':target_{i}::double precision']
        physical = '([rast] * %L::double precision + %L::double precision)'
        if field['categorical']:
            loss = f'CASE WHEN {physical} = %L::double precision THEN 0.0 ELSE 25.0 END'
        else:
            loss = f'power(({physical} - %L::double precision) / %L::double precision, 2)'
            arguments.append(f':tolerance_{i}::double precision')
        expression = f'%L::double precision * ({loss})'
        members = groups[field.get('group') or field['key']]
        if len(members) == 1:
            weight, denominator = normalization(members)
            expression = '%L::double precision * ((' + expression + ') / %L::double precision)'
            arguments = [weight] + arguments + [denominator]
        # The final argument fills source-NODATA pixels; explicitly set band
        # metadata too, so later operations propagate the same sentinel.
        leaves.append(f"ST_SetBandNoDataValue(ST_MapAlgebra({alias}.rast, 1, '64BF', "
                      f"{formatted(expression, arguments)}, -1.0), 1, -1.0) AS l{i}")

    families = []
    group_weights = []
    for n, members in enumerate(groups.values()):
        result = f'l{members[0][0]}'
        weight, denominator = normalization(members)
        group_weights.append(weight)
        for index, (i, _) in enumerate(members[1:], start=1):
            if index == len(members) - 1:
                result = binary(result, f'l{i}',
                                '%L::double precision * (([rast1] + [rast2]) / %L::double precision)',
                                [weight, denominator])
            else:
                result = binary(result, f'l{i}', '[rast1] + [rast2]')
        families.append(f'{result} AS g{n}')

    loss = 'g0'
    for n in range(1, len(families)):
        loss = binary(loss, f'g{n}', '[rast1] + [rast2]')
    # Invalid quality must propagate NODATA, even when a hard filter would fail.
    for key in quality:
        alias = aliases[key]
        loss = binary(loss, f'{alias}.rast',
                      'CASE WHEN [rast2] * %L::double precision + %L::double precision '
                      '< %L::double precision THEN -1.0 ELSE [rast1] END',
                      [f'{alias}.scale', f'{alias}.offset_value', ':minimum_coverage::double precision'])
    score_expression = ('round(65534 * greatest(0.0, least(1.0, '
                        'exp(-0.5 * least(1000.0, [rast] / %L::double precision)))))')
    score = (f"ST_SetBandNoDataValue(ST_MapAlgebra({loss}, 1, '16BUI', "
             f"{formatted(score_expression, ['(' + ' + '.join(group_weights) + ')'])}, 65535.0), 1, 65535.0)")
    for i, criterion in active:
        conditions, arguments = [], []
        alias = aliases[criterion['key']]
        for bound, operator in [('min', '>='), ('max', '<=')]:
            if criterion.get(bound) is not None:
                conditions.append(f'([rast2] * %L::double precision + %L::double precision) '
                                  f'{operator} %L::double precision')
                arguments.extend([f'{alias}.scale', f'{alias}.offset_value',
                                  f':{bound}_{i}::double precision'])
        if conditions:
            score = binary(score, f'{alias}.rast',
                           'CASE WHEN ' + ' AND '.join(conditions) + ' THEN [rast1] ELSE 0.0 END',
                           arguments, pixeltype='16BUI')
    ids = ',\n         '.join(f'max(layer_id) FILTER (WHERE feature_key={literal(key)}) AS id{i}'
                             for i, key in enumerate(required))
    joins = '\n'.join(f'JOIN tiles {aliases[key]} ON {aliases[key]}.layer_id = p.id{i}'
                      for i, key in enumerate(required))
    return f"""-- Raster-only equivalent of the pointwise builder score.
-- Requires the selected datasource's search_path; no stored function or data changes.
-- Output is one inline 16BUI raster WKB per region (64-byte header + score bytes).
WITH packed AS MATERIALIZED (
  SELECT site_key,
         {ids}
  FROM layers
  WHERE site_key = ANY(:chunk_sites::text[])
  GROUP BY site_key
)
SELECT p.site_key, ST_AsBinary(result.rast) AS score_raster
FROM packed p
{joins}
CROSS JOIN LATERAL (
  WITH terms AS MATERIALIZED (
    SELECT {', '.join(leaves)}
  ), families AS MATERIALIZED (
    SELECT {', '.join(families)} FROM terms
  )
  SELECT {score} AS rast FROM families
) result
ORDER BY p.site_key;
"""


def bind(query):
    return re.sub(r'(?<!:):([A-Za-z_]\w*)', r'%(\1)s', query.replace('%', '%%'))


def unpack_raster(payload):
    data = bytes(payload)
    order = '<' if data[0] == 1 else '>'
    header = struct.unpack(order + 'BHHddddddiHH', data[:61])
    assert header[1:3] == (0, 1) and header[-3:] == (930001, 278, 278), header
    assert data[61] & 128 == 0 and data[61] & 64 and data[61] & 15 == 6
    assert struct.unpack(order + 'H', data[62:64])[0] == 65535, data[61:64].hex()
    assert len(data) == 64 + 278 * 278 * 2
    return np.frombuffer(data, dtype=order+'u2', offset=64).astype('<u2', copy=False).tobytes()


def configure(conn):
    conn.prepare_threshold = None
    conn.read_only = True
    conn.execute('SET LOCAL ROLE earth_index_reader')
    conn.execute("SELECT set_config('search_path',%s,true)",
                 (f'pg_catalog,"{WORLD["schema_name"]}",public',))
    conn.execute("SET LOCAL jit=off; SET LOCAL work_mem='64MB'; "
                 "SET LOCAL max_parallel_workers_per_gather=0; SET LOCAL statement_timeout='120s'")


def check_rows(rows, expected, raster=False):
    actual = {site: unpack_raster(value) if raster else bytes(value) for site, value in rows}
    assert set(actual) == set(expected), (set(actual), set(expected))
    for site, payload in actual.items():
        assert hashlib.sha256(payload).hexdigest() == expected[site], f'Score mismatch: {site}'
    return actual


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--global-workers', type=int, default=0, choices=[0, 1, 2, 4, 8])
    parser.add_argument('--skip-region-checks', action='store_true',
                        help='Reuse successful checks only when saved SQL and params match exactly')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    named = raster_query()
    params_text = json.dumps(PARAMS | {'chunk_sites': SITES}, indent=2)+'\n'
    if args.skip_region_checks:
        assert (OUT/'query.sql').read_text() == named
        assert (OUT/'params.json').read_text() == params_text
    else:
        (OUT/'query.sql').write_text(named)
        (OUT/'params.json').write_text(params_text)
    # The measured query returns standard raster WKB, which is endian-aware.
    # This optional adapter matches the existing demo's packed score contract
    # on the verified little-endian PostGIS host: one U16 band has a 64-byte
    # RFC2 header. Validate the WKB first rather than treating arbitrary rasters
    # as a fixed-layout pixel array.
    (OUT/'query-packed.sql').write_text(
        '-- Existing demo output: requires the verified little-endian, one-band 16BUI layout.\n'
        'SELECT site_key, substring(score_raster FROM 65) AS scores\nFROM (\n'
        + named.rstrip().removesuffix(';')
        + '\n) raster_results\nORDER BY site_key;\n')
    query = bind(named)
    reference = json.loads((OUT.parent/'scoring-benchmark.json').read_text())['paths']['production_sql']['all_regions_sha256']
    report_path = OUT/'benchmark.json'
    report = dict(generated_at=datetime.now(timezone.utc).isoformat(), datasource_id=WORLD['datasource_id'],
                  target=TARGET['target']['key'], enabled_criteria=9, quality_bands=2,
                  settings=dict(jit='off', work_mem='64MB', max_parallel_workers_per_gather=0),
                  method='Read-only SQL execution and local result fetch; includes output packing. '
                         'Connection setup and score validation excluded. No production or DB changes. '
                         'Raster WKB adds 64 bytes per region; uint16 payload must match existing scores exactly.',
                  regions={}, global_runs=[])
    if args.skip_region_checks:
        report = json.loads(report_path.read_text())
        for site in SITES:
            assert all(report['regions'][site][name]['exact'] for name in
                       ('current_cells', 'raster_expressions', 'narrow_band_rows'))

    def save():
        report_path.write_text(json.dumps(report, indent=2)+'\n')

    with baseline.connection() as conn:
        configure(conn)
        controls = {
            'current_cells': bind(TARGET['query']['query'].replace('FROM cells', 'FROM cells WHERE site_key=ANY(:chunk_sites::text[])')),
            'raster_expressions': query,
        }
        baseline.BUNDLE = WORLD
        baseline.NS = WORLD['schema_name']
        controls['narrow_band_rows'] = baseline.per_patch_query(site_filter=True).replace('WITH scored AS (', 'WITH scored AS MATERIALIZED (')
        for site in ([] if args.skip_region_checks else SITES):
            print('Verifying region', site, flush=True)
            result = report['regions'][site] = {}
            for name, sql_query in controls.items():
                for iteration in range(3):
                    start = time.perf_counter()
                    rows = conn.execute(sql_query, PARAMS | {'chunk_sites': [site]}).fetchall()
                    elapsed = time.perf_counter() - start
                    check_rows(rows, {site: reference[site]}, name == 'raster_expressions')
                    item = result.setdefault(name, dict(samples_seconds=[], exact=True))
                    item['samples_seconds'].append(elapsed)
                    item['median_seconds'] = statistics.median(item['samples_seconds'])
                    save()
                    print(site, name, iteration+1, round(elapsed, 4), 'seconds; exact', flush=True)

        if 'hard_filter_validation' not in report:
            # Exercise real rejected pixels, not only the default coverage mask.
            filtered = json.loads(json.dumps(TARGET))
            filtered['criteria'][1]['min'] = 0.2  # bare_fraction
            filtered['criteria'][9]['max'] = 5.0  # slope_mean_deg
            filtered_params = PARAMS | {'min_1': 0.2, 'max_9': 5.0, 'chunk_sites': SITES}
            reference_sql = TARGET['query']['query'].replace(
                'ELSE exp(', 'WHEN NOT ("bare_fraction" >= :min_1 AND "slope_mean_deg" <= :max_9) '
                'THEN 0.0 ELSE exp(').replace('FROM cells', 'FROM cells WHERE site_key=ANY(:chunk_sites::text[])')
            expected_rows = conn.execute(bind(reference_sql), filtered_params).fetchall()
            expected = {s: hashlib.sha256(bytes(v)).hexdigest() for s, v in expected_rows}
            rows = conn.execute(bind(raster_query(filtered)), filtered_params).fetchall()
            actual = check_rows(rows, expected, raster=True)
            codes = np.concatenate([np.frombuffer(value, dtype='<u2') for value in actual.values()])
            counts = dict(zero=int((codes == 0).sum()), missing=int((codes == 65535).sum()),
                          positive=int(((codes > 0) & (codes < 65535)).sum()))
            assert all(counts.values()), counts
            report['hard_filter_validation'] = dict(exact=True, regions=SITES, codes=int(codes.size), **counts)
            save()
            print('Hard filter / zero / missing validation:', counts, flush=True)

    if args.global_workers:
        with ExitStack() as stack:
            connections = [stack.enter_context(baseline.connection()) for _ in range(args.global_workers)]
            for conn in connections:
                configure(conn)
            sites = sorted(reference)
            chunks = [sites[i:i+2] for i in range(0, len(sites), 2)]
            global_queries = {'raster_expressions': query, 'narrow_band_rows': controls['narrow_band_rows']}
            for name, sql_query in global_queries.items():
                def job(item):
                    conn, batches = item
                    results = []
                    for keys in batches:
                        results.extend(conn.execute(sql_query, PARAMS | {'chunk_sites': keys}).fetchall())
                    return results
                for iteration in range(2):
                    print('Starting world', name, 'run', iteration+1, 'workers', args.global_workers, flush=True)
                    start = time.perf_counter()
                    with ThreadPoolExecutor(max_workers=args.global_workers) as pool:
                        groups = list(pool.map(job, [(conn, chunks[i::args.global_workers]) for i, conn in enumerate(connections)]))
                    elapsed = time.perf_counter() - start
                    check_rows([row for group in groups for row in group], reference, name == 'raster_expressions')
                    item = dict(path=name, workers=args.global_workers, wall_seconds=elapsed,
                                regions=len(sites), score_codes=35550640, exact=True)
                    report['global_runs'].append(item)
                    save()
                    print(json.dumps(item), flush=True)
    print('Report:', report_path, flush=True)


if __name__ == '__main__':
    main()
