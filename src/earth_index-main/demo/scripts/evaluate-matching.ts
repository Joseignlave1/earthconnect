import type { Target, Criterion } from '../src/types.ts';
// Live, reproducible diagnostics. No site-specific rules or threshold fitting.
import { writeFile, mkdir } from 'node:fs/promises';
import { performance } from 'node:perf_hooks';
import { createClient } from '@connectrpc/connect';
import { createGrpcWebTransport } from '@connectrpc/connect-web';
import { EarthIndex, PlanetaryBody } from '../src/gen/earth_index_pb.ts';
import { fieldsFor, buildQuery } from '../src/query.ts';
import { applyMeasurements } from '../src/matching.ts';
import { pointMeasurements, numericMeasurement } from '../src/characteristics.ts';
import { encodeParams, resultRows } from '../src/values.ts';
const api = createClient(
  EarthIndex,
  createGrpcWebTransport({
    baseUrl: process.env.MATCHING_API ?? 'http://localhost:8080',
  }),
);
const datasource = (await api.metadata({})).datasources
  .filter((d) => d.body === 'earth')
  .sort(
    (a, b) =>
      b.coverage.length - a.coverage.length ||
      (b.relations.find((r) => r.name === 'cells')?.columns.length ?? 0) -
        (a.relations.find((r) => r.name === 'cells')?.columns.length ?? 0),
  )[0];
const fields = fieldsFor(datasource);
const targets: (Target & { name: string })[] = [
  { name: 'Lunar mare', body: 'moon', longitude: -20, latitude: 30 },
  { name: 'Tycho vicinity', body: 'moon', longitude: -11.36, latitude: -43.31 },
  {
    name: 'Lunar north polar terrain',
    body: 'moon',
    longitude: 0,
    latitude: 85,
  },
  {
    name: 'Meridiani vicinity',
    body: 'mars',
    longitude: 23.47,
    latitude: 0.67,
  },
  { name: 'Gale vicinity', body: 'mars', longitude: 137.4, latitude: -4.6 },
  {
    name: 'Valles Marineris vicinity',
    body: 'mars',
    longitude: -60,
    latitude: -10,
  },
];
type Best = { siteKey: string; index: number; code: number; score: number };
type Evaluation = {
  elapsedMs: number;
  bytes: number;
  valid: number;
  missing: number;
  high90: number;
  high75: number;
  best: Best | null;
  patches: {
    siteKey: string;
    valid: number;
    missing: number;
    mean: number | null;
    max: number | null;
    high90: number;
    high75: number;
  }[];
};
async function evaluate(q: ReturnType<typeof buildQuery>): Promise<Evaluation> {
  const start = performance.now();
  const rows = resultRows(
    await api.query(
      {
        datasourceId: datasource.id,
        query: q.query,
        params: encodeParams(q.params),
      },
      { timeoutMs: 15000 },
    ),
  );
  const elapsedMs = performance.now() - start;
  let valid = 0,
    missing = 0,
    high90 = 0,
    high75 = 0,
    best: Best | null = null,
    bytes = 0;
  const patches = rows.map((row) => {
    if (!(row.scores instanceof Uint8Array) || typeof row.site_key !== 'string')
      throw new Error('Invalid score row');
    const v = new DataView(row.scores.buffer, row.scores.byteOffset, row.scores.byteLength);
    bytes += row.scores.byteLength;
    let n = 0,
      unknown = 0,
      sum = 0,
      max = -1,
      count90 = 0,
      count75 = 0;
    for (let i = 0; i < v.byteLength / 2; i++) {
      const code = v.getUint16(i * 2, true);
      if (code === 65535) {
        missing++;
        unknown++;
        continue;
      }
      const s = code / 65534;
      valid++;
      n++;
      sum += s;
      if (s >= 0.9) {
        high90++;
        count90++;
      }
      if (s >= 0.75) {
        high75++;
        count75++;
      }
      max = Math.max(max, s);
      if (!best || code > best.code) best = { siteKey: row.site_key, index: i, code, score: s };
    }
    return {
      siteKey: row.site_key,
      valid: n,
      missing: unknown,
      mean: n ? sum / n : null,
      max: n ? max : null,
      high90: count90,
      high75: count75,
    };
  });
  return {
    elapsedMs,
    bytes,
    valid,
    missing,
    high90,
    high75,
    best,
    patches: patches.sort((a, b) => (b.max ?? -1) - (a.max ?? -1)),
  };
}
const results = [];
for (const target of targets) {
  const start = performance.now();
  const response = await api.samplePoint(
    {
      body: target.body === 'moon' ? PlanetaryBody.MOON : PlanetaryBody.MARS,
      longitude: target.longitude,
      latitude: target.latitude,
    },
    { timeoutMs: 25000 },
  );
  const sample = { measurements: pointMeasurements(response) };
  const criteria: Criterion[] = fields.map((f) => ({
    key: f.key,
    target: null,
    enabled: false,
    tolerance: f.tolerance,
    weight: 1,
    min: null,
    max: null,
  }));
  const evidence = applyMeasurements(criteria, target, sample);
  const output = {
    before: undefined as Evaluation | undefined,
    after: undefined as Evaluation | undefined,
    target,
    samplingMs: performance.now() - start,
    partial: response.partial,
    evidence,
    criteria: criteria.filter((c) => c.enabled),
    measurements: sample.measurements
      .filter((m) => m.variableId.includes('km'))
      .map(({ variableId, decoded, unit, status, qualityFlags }) => ({
        variableId,
        value: decoded,
        unit,
        status,
        qualityFlags,
      })),
  };
  const originalSlope = numericMeasurement(
    sample.measurements,
    `${target.body}.local_slope`,
    'degree',
  );
  if (originalSlope != null)
    output.before = await evaluate(
      buildQuery(
        [
          {
            ...fields.find((f) => f.key === 'slope_mean_deg')!,
            group: 'slope_mean_deg',
            qualityKey: null,
          },
        ],
        [
          {
            key: 'slope_mean_deg',
            enabled: true,
            target: originalSlope,
            tolerance: 10,
            weight: 1,
          },
        ],
      ),
    );
  if (evidence.measured) output.after = await evaluate(buildQuery(fields, criteria));
  results.push(output);
  console.log(
    JSON.stringify({
      target: target.name,
      evidence,
      samplingMs: Math.round(Number(output.samplingMs)),
      beforeHigh90: output.before?.high90,
      afterHigh90: output.after?.high90,
      best: output.after?.best?.siteKey,
      score: output.after?.best?.score,
      queryMs: Math.round(output.after?.elapsedMs ?? 0),
    }),
  );
}
const report = {
  generatedAt: new Date().toISOString(),
  datasourceId: datasource.id,
  scope:
    'Six predeclared points over the existing 27-patch pilot; diagnostic comparisons, not calibrated planetary analogue ground truth. All defaults are global, with no geographic exclusions or per-site tuning. Before reproduces native slope only with 10 degree tolerance and no coverage gate.',
  results,
};
const destination = new URL(
  '../../research/pilots/database/matching-evaluation.json',
  import.meta.url,
);
await mkdir(new URL('.', destination), { recursive: true });
await writeFile(destination, JSON.stringify(report, null, 2) + '\n');
