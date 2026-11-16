import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { createClient } from '@connectrpc/connect';
import { createGrpcWebTransport } from '@connectrpc/connect-web';
import { EarthIndex } from '../src/gen/earth_index_pb.ts';
import { fieldsFor, buildQuery } from '../src/query.ts';
import { encodeParams, resultRows } from '../src/values.ts';
import type { Criterion } from '../src/types.ts';

const baseUrl = process.env.MATCHING_API ?? 'http://localhost:8080';
const client = createClient(EarthIndex, createGrpcWebTransport({ baseUrl }));
assert.equal((await client.health({}, { timeoutMs: 120000 })).ready, true);
const page = await fetch(new URL('/demo/', baseUrl));
assert.equal(page.status, 200);
const html = await page.text();
const asset = html.match(/src="([^"]+\.js)"/)?.[1];
assert(asset, 'Built demo must reference a JavaScript asset');
assert.equal((await fetch(new URL(asset, baseUrl))).status, 200);

const metadata = await client.metadata({}, { timeoutMs: 30000 });
const datasource = metadata.datasources.find((d) => d.id === 'earth-ed5bc9904a5f');
assert(datasource, 'Expected global scientific revision');
assert.equal(datasource.grids.length, 460);
assert.equal(
  datasource.relations.find((r) => r.name === 'cells')?.columns.filter((c) => c.unit).length,
  25,
);
const probe = resultRows(
  await client.query(
    {
      datasourceId: datasource.id,
      query: 'SELECT :value::INTEGER + :value::INTEGER AS answer, :empty::DOUBLE AS missing',
      params: encodeParams({ value: 21, empty: null }),
    },
    { timeoutMs: 30000 },
  ),
);
assert.deepEqual(probe, [{ answer: 42, missing: null }]);
const packed = resultRows(
  await client.query(
    {
      datasourceId: datasource.id,
      query: 'SELECT pack_u16([0, 65534, NULL]::INTEGER[]) AS scores',
    },
    { timeoutMs: 30000 },
  ),
);
assert(packed[0].scores instanceof Uint8Array);
assert.deepEqual([...packed[0].scores], [0, 0, 254, 255, 255, 255]);
console.log('Verified demo assets, health, metadata, named parameters, NULL and uint16 packing');

if (process.argv.includes('--world')) {
  const fields = fieldsFor(datasource);
  const criteria: Criterion[] = fields.map((f) => ({
    key: f.key,
    target: f.key === 'slope_mean_deg' ? 1 : null,
    enabled: f.key === 'slope_mean_deg',
    tolerance: f.tolerance,
    weight: 1,
    min: null,
    max: null,
  }));
  const built = buildQuery(fields, criteria);
  const rows = resultRows(
    await client.query(
      {
        datasourceId: datasource.id,
        query: built.query,
        params: encodeParams(built.params),
      },
      { timeoutMs: 120000 },
    ),
  );
  assert.equal(rows.length, datasource.grids.length);
  const grids = new Map(datasource.grids.map((g) => [g.siteKey, g]));
  const hashes: Record<string, string> = {};
  let bytes = 0,
    positive = 0,
    missing = 0;
  for (const row of rows) {
    assert.equal(typeof row.site_key, 'string');
    const key = row.site_key as string;
    const grid = grids.get(key);
    assert(grid && !hashes[key], `Unknown or duplicate region ${key}`);
    assert(row.scores instanceof Uint8Array);
    assert.equal(row.scores.byteLength, grid.width * grid.height * 2);
    const view = new DataView(row.scores.buffer, row.scores.byteOffset, row.scores.byteLength);
    for (let i = 0; i < view.byteLength; i += 2) {
      const code = view.getUint16(i, true);
      if (code === 65535) missing++;
      else if (code > 0) positive++;
    }
    bytes += row.scores.byteLength;
    hashes[key] = createHash('sha256').update(row.scores).digest('hex');
  }
  assert.equal(bytes, 71_101_280);
  assert(positive > 0 && missing > 0);
  const digest = createHash('sha256')
    .update(JSON.stringify(Object.entries(hashes).sort()))
    .digest('hex');
  console.log(
    JSON.stringify({
      datasource: datasource.id,
      regions: rows.length,
      bytes,
      positive,
      missing,
      sha256: digest,
    }),
  );
}
