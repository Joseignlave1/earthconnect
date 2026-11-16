import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createClient } from '@connectrpc/connect';
import { createGrpcWebTransport } from '@connectrpc/connect-web';
import { EarthIndex } from '../src/gen/earth_index_pb.ts';
import { buildQuery } from '../src/query.ts';
import { applyMeasurements } from '../src/matching.ts';
import { encodeParams, resultRows } from '../src/values.ts';
import { fields, newCriteria, sample, target } from '../test-support/matching.ts';

const client = createClient(
  EarthIndex,
  createGrpcWebTransport({
    baseUrl: process.env.MATCHING_API ?? 'http://localhost:8080',
  }),
);
// Execute the actual generated SQL on adversarial cells using a CTE. This tests
// DuckDB NULL/CASE, grouping, binding, packing and scoring together.
async function score(criteria, rows, selectedFields = fields) {
  const q = buildQuery(selectedFields, criteria);
  const names = [
    'slope_mean_deg',
    'relief_5km_m',
    'relief_21km_m',
    'tpi_5km_m',
    'tpi_21km_m',
    'bare_fraction',
    'terrain_valid_fraction',
    'landcover_valid_fraction',
    'warmest_month_air_c',
  ];
  const params = { ...q.params };
  const values = rows
    .map(
      (row, i) =>
        `(0::USMALLINT,${i}::UINTEGER,${names
          .map((key, j) => {
            const name = `cell_${i}_${j}`;
            params[name] = row[key] ?? null;
            return `:${name}::double precision`;
          })
          .join(',')})`,
    )
    .join(',');
  const query = q.query.replace(
    'WITH scored AS (',
    `WITH regions AS (SELECT 0::USMALLINT region_id,'fixture' site_key,${rows.length}::UINTEGER width,1::UINTEGER height), measurements AS (SELECT * FROM (VALUES ${values}) AS v(region_id,pixel,${names.join(',')})), scored AS (`,
  );
  const result = resultRows(
    await client.query({
      datasourceId: 'earth-ed5bc9904a5f',
      query,
      params: encodeParams(params),
    }),
  );
  const bytes = result[0].scores;
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  return rows.map((_, i) => view.getUint16(i * 2, true));
}
const perfect = {
  slope_mean_deg: 1,
  relief_5km_m: 100,
  relief_21km_m: 500,
  tpi_5km_m: 0,
  tpi_21km_m: -50,
  bare_fraction: 1,
  terrain_valid_fraction: 1,
  landcover_valid_fraction: 1,
  warmest_month_air_c: 30,
};
const criteria = () => {
  const c = newCriteria();
  applyMeasurements(c, target, sample);
  return c;
};
const live = { skip: !process.env.MATCHING_API };

test(
  'adversarial SQL: oceans, forest, partial coverage and missing evidence cannot fake high matches',
  live,
  async () => {
    const scores = await score(criteria(), [
      perfect,
      { ...perfect, bare_fraction: 0 },
      { ...perfect, bare_fraction: null, landcover_valid_fraction: 0 },
      { ...perfect, landcover_valid_fraction: 0.05 },
      { ...perfect, slope_mean_deg: 11 },
      { ...perfect, relief_21km_m: null },
      { ...perfect, terrain_valid_fraction: 0.89 },
    ]);
    assert.equal(scores[0], 65534);
    assert.ok(scores[1] / 65534 < 0.05, `forest scored ${scores[1]}`);
    assert.equal(scores[2], 65535);
    assert.equal(scores[3], 65535);
    assert.ok(scores[4] / 65534 < 0.05, `conflicting slope scored ${scores[4]}`);
    assert.equal(scores[5], 65535);
    assert.equal(scores[6], 65535);
  },
);

test('adversarial SQL: duplicated terrain scales cannot buy extra weight', live, async () => {
  const c = criteria();
  const row = {
    ...perfect,
    bare_fraction: 0.6,
    relief_5km_m: 150,
    relief_21km_m: 675,
  };
  const a = await score(c, [row]);
  c.find((x) => x.key === 'relief_21km_m').enabled = false;
  const b = await score(c, [row]);
  assert.deepEqual(a, b); // both relief errors were exactly one tolerance
});

test(
  'adversarial SQL: opposing hot/flat cells do not combine into a hot-flat result',
  live,
  async () => {
    const f = fields.filter((f) => ['slope_mean_deg', 'warmest_month_air_c'].includes(f.key));
    const c = [
      { key: 'slope_mean_deg', enabled: true, target: 0, tolerance: 1, weight: 1 },
      { key: 'warmest_month_air_c', enabled: true, target: 30, tolerance: 5, weight: 1 },
    ];
    const result = await score(
      c,
      [
        { ...perfect, slope_mean_deg: 0, warmest_month_air_c: -20 },
        { ...perfect, slope_mean_deg: 10, warmest_month_air_c: 30 },
      ],
      f,
    );
    assert.ok(result.every((s) => s === 0));
  },
);

test(
  'adversarial SQL: hard filters preserve positions; physical zero still matches',
  live,
  async () => {
    const c = criteria();
    c[0].max = 5;
    const result = await score(c, [perfect, { ...perfect, slope_mean_deg: 6 }]);
    assert.deepEqual(result, [65534, 0]);
    const flat = newCriteria();
    const s = {
      measurements: sample.measurements.map((m) => ({
        ...m,
        decoded: m.variableId.includes('elevation') ? 5000 : 0,
      })),
    };
    applyMeasurements(flat, target, s);
    const [zero] = await score(flat, [
      {
        ...perfect,
        slope_mean_deg: 0,
        relief_5km_m: 0,
        relief_21km_m: 0,
        tpi_5km_m: 0,
        tpi_21km_m: 0,
      },
    ]);
    assert.equal(zero, 65534);
  },
);

test(
  'adversarial SQL: enormous finite mismatch returns zero without exp underflow',
  live,
  async () => {
    const c = criteria();
    c[0].target = 1e12;
    c[0].tolerance = 1e-6;
    const [value] = await score(c, [perfect]);
    assert.equal(value, 0);
  },
);
