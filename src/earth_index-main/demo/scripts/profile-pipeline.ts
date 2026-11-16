// Read-only timings for the actual world query and binary gRPC-Web pipeline.
import assert from 'node:assert/strict';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { performance } from 'node:perf_hooks';
import { create, toBinary, fromBinary } from '@bufbuild/protobuf';
import { createClient } from '@connectrpc/connect';
import { createGrpcWebTransport } from '@connectrpc/connect-web';
import { EarthIndex, QueryRequestSchema, QueryResponseSchema } from '../src/gen/earth_index_pb.ts';
import { buildQuery } from '../src/query.ts';
import { encodeParams, resultRows } from '../src/values.ts';
import { prepareScores, scoreSummary, scorePixels } from '../src/scores.ts';
import { scoreDistribution } from '../src/picks.ts';
import { scorePyramid } from '../src/score-overviews.ts';
import { strongestAreas } from '../src/score-visuals.ts';
import { createHash } from 'node:crypto';
import type { Field, Criterion } from '../src/types.ts';

const root = resolve(import.meta.dirname, '../../research/world-4km');
const out = resolve(root, 'pipeline-profile');
await mkdir(out, { recursive: true });
const target = (
  JSON.parse(
    await readFile(resolve(root, '../expansion-4km/validation/targets.json'), 'utf8'),
  ) as Array<{ target: { key: string }; fields: Field[]; criteria: Criterion[] }>
).find((t) => t.target.key === 'valles')!;
const built = buildQuery(target.fields, target.criteria);
const request = create(QueryRequestSchema, {
  datasourceId: 'earth-ed5bc9904a5f',
  ...built,
  params: encodeParams(built.params),
});
const wire = toBinary(QueryRequestSchema, request);
await writeFile(resolve(out, 'request.pb'), wire);
await writeFile(resolve(out, 'query.json'), JSON.stringify(built, null, 2));
if (process.argv.includes('--prepare-only')) process.exit(0);
const baseUrl = process.env.MATCHING_API ?? 'http://localhost:8080';
const client = createClient(EarthIndex, createGrpcWebTransport({ baseUrl }));
const datasource = (await client.metadata({})).datasources.find(
  (d) => d.id === request.datasourceId,
)!;
const reference = JSON.parse(await readFile(resolve(root, 'scoring-benchmark.json'), 'utf8')).paths
  .production_sql.all_regions_sha256;
const samples = [];
const payload = new Uint8Array(wire.length + 5);
payload.set(wire, 5);
new DataView(payload.buffer).setUint32(1, wire.length, false);
for (let run = 0; run < 4; run++) {
  const start = performance.now();
  const response = await fetch(`${baseUrl}/earth_index.v1.EarthIndex/Query`, {
    method: 'POST',
    headers: { 'content-type': 'application/grpc-web+proto', 'x-grpc-web': '1' },
    body: payload,
  });
  const headers = performance.now();
  assert(response.ok);
  const body = new Uint8Array(await response.arrayBuffer());
  const received = performance.now();
  let offset = 0,
    message: Uint8Array | undefined,
    status = response.headers.get('grpc-status') ?? '0';
  while (offset < body.length) {
    const flag = body[offset];
    const n = new DataView(body.buffer, body.byteOffset + offset + 1, 4).getUint32(0, false);
    offset += 5;
    if (flag & 128) {
      const trailers = new TextDecoder().decode(body.subarray(offset, offset + n));
      status = /grpc-status:\s*(\d+)/.exec(trailers)?.[1] ?? status;
    } else {
      assert.equal(flag, 0);
      assert(!message);
      message = body.subarray(offset, offset + n);
    }
    offset += n;
  }
  assert.equal(status, '0');
  assert(message);
  const result = fromBinary(QueryResponseSchema, message);
  const decoded = performance.now();
  const rows = resultRows(result);
  const patches = prepareScores(rows, datasource);
  const u16 = performance.now();
  const summary = scoreSummary(patches);
  const distribution = scoreDistribution(patches);
  const analysed = performance.now();
  const overviews = patches.map((p) => scorePyramid(p.scores, p.grid.width, p.grid.height));
  const pyramids = performance.now();
  const areas = strongestAreas(patches, summary.best?.score ?? 0);
  const picks = performance.now();
  let pixelBytes = 0;
  for (const p of patches)
    pixelBytes += scorePixels(p.scores, 0, summary.best?.score ?? 0).byteLength;
  const pixels = performance.now();
  // Validation deliberately sits outside all reported timed intervals.
  for (const row of rows) {
    assert(typeof row.site_key === 'string' && row.scores instanceof Uint8Array);
    assert.equal(createHash('sha256').update(row.scores).digest('hex'), reference[row.site_key]);
  }
  assert.equal(distribution.valid + distribution.missing, 35_550_640);
  const sample = {
    run,
    warmup: run === 0,
    headers_ms: headers - start,
    body_read_ms: received - headers,
    protobuf_decode_ms: decoded - received,
    rpc_decoded_ms: decoded - start,
    u16_decode_ms: u16 - decoded,
    summary_histogram_ms: analysed - u16,
    overviews_ms: pyramids - analysed,
    top_areas_ms: picks - pyramids,
    rgba_buffers_ms: pixels - picks,
    client_cpu_ms: pixels - received,
    body_bytes: body.length,
    regions: patches.length,
    pixel_bytes: pixelBytes,
    overview_levels: overviews.reduce((n, p) => n + p.length, 0),
    top_areas: areas.length,
    exact: true,
  };
  samples.push(sample);
  console.log(JSON.stringify(sample));
}
await writeFile(
  resolve(out, 'client.json'),
  JSON.stringify(
    {
      method:
        'Local Docker release API, current Valles world query, 8 engine threads. Raw binary gRPC-Web, no gzip. Warmup plus 3 timed sequential runs. Explicit buffers and protobuf decoding. UI calculations execute real frontend functions in Node; canvas/map rendering and browser paint are excluded. SHA-256 validation is outside all timed intervals.',
      samples,
    },
    null,
    2,
  ) + '\n',
);
