import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { performance } from 'node:perf_hooks';
import { createClient } from '@connectrpc/connect';
import { createGrpcWebTransport } from '@connectrpc/connect-web';
import { EarthIndex } from '../src/gen/earth_index_pb.ts';
import { encodeParams, resultRows } from '../src/values.ts';
import WKB from 'ol/format/WKB.js';
import { buildQuery } from '../src/query.ts';
import { createHash } from 'node:crypto';

const out = resolve(import.meta.dirname, '../../research/world-4km');
const database = JSON.parse(await readFile(resolve(out, 'database.json'), 'utf8'));
const fixtures = JSON.parse(await readFile(resolve(out, 'validation/api-cases.json'), 'utf8'));
const targets = JSON.parse(await readFile(resolve(out, '../expansion-4km/validation/targets.json'), 'utf8'));
const api = createClient(EarthIndex, createGrpcWebTransport({ baseUrl: process.env.MATCHING_API ?? 'http://localhost:8080' }));
assert((await api.health({})).ready);
const started = performance.now();
const metadata = await api.metadata({}, { timeoutMs: 15000 });
const metadataMs = performance.now() - started;
const datasource = metadata.datasources.find((d) => d.id === database.datasource_id);
assert(datasource, 'Global datasource must be published');
assert.equal(datasource.grids.length, 460);
assert.equal(datasource.coverage.length, 460);
const columns = datasource.relations.find((r) => r.name === 'cells')!.columns;
assert.equal(columns.filter((c) => c.unit).length, 25);
assert(datasource.grids.every((g) => g.width === 278 && g.height === 278 && g.srid === 930001 && g.cellAddressing.includes('global sinusoidal')));
const annual = columns.find((c) => c.name === 'annual_precip_mm')!;
assert.equal(annual.provenance.length, 460);
for (const footprint of datasource.coverage) {
  const geometry = new WKB().readGeometry(footprint.wkb);
  assert(geometry);
  assert.equal(geometry.getType(), 'MultiPolygon');
  const extent = geometry.getExtent();
  assert(extent.every(Number.isFinite));
  assert(extent[0] >= -180 && extent[1] >= -90 && extent[2] <= 180 && extent[3] <= 90);
}
const results = [];
for (const fixture of fixtures.cases) {
  const target = targets.find((t: { target: { key: string } }) => t.target.key === fixture.target);
  const built = buildQuery(target.fields, target.criteria);
  const query = built.query.replace('FROM measurements', 'FROM measurements WHERE region_id = (SELECT region_id FROM regions WHERE site_key = :region)');
  const start = performance.now();
  const response = await api.query({ datasourceId: datasource.id, query,
    params: encodeParams({ ...built.params, region: fixture.site_key }) }, { timeoutMs: 15000 });
  const rows = resultRows(response);
  const wallMs = performance.now() - start;
  assert.equal(rows.length, 1);
  assert.equal(rows[0].site_key, fixture.site_key);
  assert(rows[0].scores instanceof Uint8Array);
  const expected = await readFile(resolve(out, 'validation', fixture.path));
  assert(Buffer.from(rows[0].scores).equals(expected), `${fixture.target}/${fixture.site_key} score mismatch`);
  results.push({ ...fixture, wallMs, bytes: expected.byteLength, exact: true });
  console.log('Verified', fixture.target, fixture.site_key, fixture.cells, 'scores');
}
const probes = [];
for (const [region, x, y, padding] of [['h14v00', 1, 1, true], ['h17v00', 278, 1, false], ['h00v08', 1, 278, false]] as const) {
  const rows = resultRows(await api.query({ datasourceId: datasource.id,
    query: 'SELECT geom IS NULL AS missing,CASE WHEN geom IS NULL THEN NULL ELSE ST_AsBinary(geom) END AS footprint FROM cells WHERE site_key=:region AND x=:x AND y=:y',
    params: encodeParams({ region, x, y }) }));
  assert.equal(rows.length, 1);
  assert.equal(rows[0].missing, padding);
  if (!padding) {
    assert(rows[0].footprint instanceof Uint8Array);
    const geometry = new WKB().readGeometry(rows[0].footprint);
    assert(geometry);
    const extent = geometry.getExtent();
    assert(extent.every(Number.isFinite));
    assert(extent[0] >= -180 && extent[2] <= 180 && extent[1] >= -90 && extent[3] <= 90);
  }
  probes.push({ region, x, y, padding, passed: true });
}
const target = targets.find((t: {target:{key:string}}) => t.target.key === 'valles');
const built = buildQuery(target.fields,target.criteria);
const reference = JSON.parse(await readFile(resolve(out,'scoring-benchmark.json'),'utf8')).paths.production_sql.all_regions_sha256;
const fullRuns = [];
for (let i=0;i<3;i++) {
  const start=performance.now();
  const response=await api.query({datasourceId:datasource.id,query:built.query,params:encodeParams(built.params)});
  const rows=resultRows(response);
  assert.equal(rows.length,460);
  let bytes=0;
  for (const row of rows) {
    assert(typeof row.site_key === 'string' && row.scores instanceof Uint8Array);
    assert.equal(createHash('sha256').update(row.scores).digest('hex'),reference[row.site_key],row.site_key);
    bytes+=row.scores.byteLength;
  }
  fullRuns.push({wallMs:performance.now()-start,bytes,regions:rows.length,exact:true});
  console.log('Full world:',fullRuns.at(-1));
}
const report = { datasourceId: datasource.id, metadataMs, coverageRegions: datasource.coverage.length,
  fields: 25, scoreCodesChecked: results.reduce((n, r) => n + r.cells, 0), results, geometryProbes: probes,fullRuns,
  scope: 'Deployed binary gRPC-Web and current query builder. Six targets / three regions checked byte-for-byte; all 460 full-world score matrix hashes match the original PostgreSQL reference.' };
await writeFile(resolve(out, 'duckdb-api-validation.json'), JSON.stringify(report, null, 2) + '\n');
console.log('Global DuckDB API verification passed');
