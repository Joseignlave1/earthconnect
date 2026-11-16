# /// script
# requires-python = ">=3.11"
# dependencies = ["psycopg[binary]==3.3.9", "h3==4.5.0", "rasterio==1.5.2", "numpy==2.5.4"]
# ///
"""Compare raster access, quantized cell rows and decoded cell rows using TEMP tables.

No production schema or data changes. Scores every pilot cell without window
grouping or ordering; returns aggregate checksums to isolate server computation.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

from psycopg import sql

from evaluate_ranking_pilot import OUT, query as original_query
from fast_ranking_query import ORIGINAL_SOURCE, cell_source, required_fields
from load_raster_pilot import ROOT, connection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads((OUT / 'spec.json').read_text())
    fields = required_fields(spec, {k: 1 / len(spec['components']) for k in spec['components']})
    metadata = []
    for i, field in enumerate(fields):
        for column, prefix in [('scale', 'scale_'), ('offset_value', 'offset_')]:
            metadata.append(sql.SQL('max({}) FILTER (WHERE feature_key={}) AS {}').format(
                sql.Identifier(column), sql.Literal(field), sql.Identifier(prefix + str(i))))
    bundle = spec['source_bundle_sha256']
    with connection() as conn:
        conn.execute("SET LOCAL statement_timeout='30s'")
        conn.execute(sql.SQL('CREATE TEMP TABLE value_layout_metadata ON COMMIT DROP AS '
            'SELECT site_key,{} FROM research.raster_layers WHERE bundle_sha256=%s GROUP BY site_key').format(
                sql.SQL(',').join(metadata)), (bundle,))
        conn.execute(sql.SQL('CREATE TEMP TABLE value_layout_decoded ON COMMIT DROP AS {}').format(
            cell_source(spec, {k: 1 / len(spec['components']) for k in spec['components']})), (bundle,))
        encoded = []
        decoded = []
        for i, field in enumerate(fields):
            # Store code minus 32768 in PostgreSQL's signed SMALLINT. NULL holds
            # missingness; all valid original uint16 codes round-trip exactly.
            encoded.append(sql.SQL('(round((d.{}-m.{})/m.{})::integer-32768)::smallint AS {}').format(
                sql.Identifier(field), sql.Identifier('offset_' + str(i)),
                sql.Identifier('scale_' + str(i)), sql.Identifier(field)))
            decoded.append(sql.SQL('(q.{}::integer+32768)*m.{}+m.{} AS {}').format(
                sql.Identifier(field), sql.Identifier('scale_' + str(i)),
                sql.Identifier('offset_' + str(i)), sql.Identifier(field)))
        conn.execute(sql.SQL('CREATE TEMP TABLE value_layout_quantized ON COMMIT DROP AS '
            'SELECT d.site_key,d.x,d.y,{} FROM value_layout_decoded d '
            'JOIN value_layout_metadata m USING(site_key)').format(sql.SQL(',').join(encoded)))
        quant_source = sql.SQL('SELECT q.site_key,q.x,q.y,{} FROM value_layout_quantized q '
                              'JOIN value_layout_metadata m USING(site_key)').format(sql.SQL(',').join(decoded))
        # The pilot uses one scale/offset pair per feature across all sites.
        # A constant-decoding path isolates arithmetic from the metadata join.
        meta_rows = conn.execute('SELECT * FROM value_layout_metadata').fetchall()
        constants = []
        for i, field in enumerate(fields):
            pairs = {(r[1 + 2 * i], r[2 + 2 * i]) for r in meta_rows}
            if len(pairs) != 1:
                raise ValueError('Constant-decoding control requires shared per-feature quantization')
            scale, offset = pairs.pop()
            constants.append(sql.SQL('(q.{}::integer+32768)*{}::float8+{}::float8 AS {}').format(
                sql.Identifier(field), sql.Literal(scale), sql.Literal(offset), sql.Identifier(field)))
        constant_source = sql.SQL('SELECT q.site_key,q.x,q.y,{} FROM value_layout_quantized q').format(
            sql.SQL(',').join(constants))
        # Compare every decoded feature at every cell before benchmarking.
        checks = [sql.SQL('count(*) FILTER (WHERE d.{} IS DISTINCT FROM q.{})').format(
            sql.Identifier(f), sql.Identifier(f)) for f in fields]
        checked = conn.execute(sql.SQL('SELECT count(*),{} FROM value_layout_decoded d '
            'FULL JOIN ({}) q USING(site_key,x,y)').format(sql.SQL(',').join(checks), quant_source)).fetchone()
        if checked[0] != 270000 or any(checked[1:]):
            raise AssertionError(f'Quantized source round-trip failed: {checked}')
        constant_checked = conn.execute(sql.SQL('SELECT count(*),{} FROM value_layout_decoded d '
            'FULL JOIN ({}) q USING(site_key,x,y)').format(sql.SQL(',').join(checks), constant_source)).fetchone()
        if constant_checked != checked:
            raise AssertionError('Constant decoding differs from metadata decoding')
        sizes = {}
        for name in ['decoded', 'quantized', 'metadata']:
            conn.execute(sql.SQL('ANALYZE {}').format(sql.Identifier('value_layout_' + name)))
            sizes[name] = conn.execute('SELECT pg_relation_size(%s::regclass)', ('value_layout_' + name,)).fetchone()[0]
            conn.execute(sql.SQL('GRANT SELECT ON {} TO earth_index_reader').format(sql.Identifier('value_layout_' + name)))
        conn.execute('SET LOCAL ROLE earth_index_reader')
        conn.execute("SET LOCAL statement_timeout='10s'")
        results = {}
        suffix = ''' SELECT count(*) AS pixels,
            count(*) FILTER (WHERE gate='candidate') AS candidates,
            count(*) FILTER (WHERE gate='unknown') AS unknown,
            sum(CASE WHEN gate='candidate' THEN score_lower ELSE 0 END) AS lower_sum,
            sum(CASE WHEN gate<>'excluded' THEN score_upper ELSE 0 END) AS upper_sum,
            sum(CASE WHEN gate<>'excluded' THEN evidence ELSE 0 END) AS evidence_sum
            FROM scored'''
        for name, profile in spec['profiles'].items():
            template = original_query(spec, profile['weights']).as_string(conn)
            marker = '    SELECT site_key,((x-1)/'
            if template.count(marker) != 1 or template.count(ORIGINAL_SOURCE) != 1:
                raise ValueError('Frozen scoring template changed; review benchmark')
            template = template.split(marker, 1)[0] + suffix
            sources = {
                'raster': cell_source(spec, profile['weights']).as_string(conn),
                'quantized_rows_metadata': quant_source.as_string(conn),
                'quantized_rows_constants': constant_source.as_string(conn),
                'decoded_rows': 'SELECT * FROM value_layout_decoded',
            }
            selected = sql.SQL(',').join(sql.Identifier(f) for f in ['site_key', 'x', 'y'] + required_fields(spec, profile['weights']))
            sources = {k: sql.SQL('SELECT {} FROM ({}) raw_values').format(selected, sql.SQL(v)).as_string(conn)
                       for k, v in sources.items()}
            statements = {k: sql.SQL(template.replace(ORIGINAL_SOURCE, v)) for k, v in sources.items()}
            # Check the same per-cell scorer and missingness rules on all paths.
            checksums = {k: conn.execute(s, (bundle,) if k == 'raster' else ()).fetchone()
                         for k, s in statements.items()}
            for checksum in checksums.values():
                for a, b in zip(checksums['raster'], checksum):
                    if not math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-7):
                        raise AssertionError('Score checksums differ')
            samples = {k: [] for k in sources}
            plans = {}
            paths = list(sources)
            # Rotate execution order; warm-up/checksum runs above are untimed.
            for repetition in range(5):
                for path in paths[repetition % len(paths):] + paths[:repetition % len(paths)]:
                    plan = conn.execute(sql.SQL('EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {}').format(
                        statements[path]), (bundle,) if path == 'raster' else ()).fetchone()[0][0]
                    samples[path].append(plan['Execution Time'])
                    plans[path] = plan
            medians = {k: statistics.median(v) for k, v in samples.items()}
            results[name] = {'server_execution_ms': samples, 'median_ms': medians,
                             'checksums': checksums, 'last_plans': plans}
            print(json.dumps({name: medians}), flush=True)
    report = {
        'status': 'passed', 'generated_at': datetime.now(timezone.utc).isoformat(),
        'bundle_sha256': bundle, 'spec_sha256': hashlib.sha256((OUT / 'spec.json').read_bytes()).hexdigest(),
        'processor_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'cells_verified': checked[0], 'values_verified': checked[0] * len(fields),
        'temporary_heap_bytes': sizes, 'profiles': results,
        'method': 'Warm-cache server EXPLAIN ANALYZE, 5 rotated-order samples per path/profile, same connection and reader role. TEMP tables with ANALYZE, no added indexes. Same float64 per-cell scorer, no score sorting or window grouping. Checksums force all scores to be evaluated; client serialization is excluded.',
        'limits': ['270000 existing 1km pilot cells; not a global or 4km benchmark.',
                   'Temporary tables and cache behavior differ from persistent production tables.',
                   'Quantized metadata path includes a per-site join; constant path verifies shared per-feature scale/offset and removes this join.',
                   'Heap sizes include row overhead and 10 feature columns; queries materialize only the profile-required columns.',
                   'Decoded values use float64; float32 performance and precision were not tested.',
                   'All temporary tables are dropped at transaction end; no production schema or data change.']}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
