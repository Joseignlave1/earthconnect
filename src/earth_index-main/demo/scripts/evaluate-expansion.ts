// Freeze the same target and frontend SQL for both grids; never overwrite old reports.
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { performance } from 'node:perf_hooks';
import { createClient } from '@connectrpc/connect';
import { createGrpcWebTransport } from '@connectrpc/connect-web';
import { EarthIndex, PlanetaryBody } from '../src/gen/earth_index_pb.ts';
import { fieldsFor, buildQuery } from '../src/query.ts';
import { applyMeasurements } from '../src/matching.ts';
import { pointMeasurements } from '../src/characteristics.ts';
import { encodeParams, resultRows } from '../src/values.ts';
import type { Target, Criterion, Field } from '../src/types.ts';
const output = resolve('../research/expansion-4km/validation');
await mkdir(output, { recursive: true });
const api = createClient(EarthIndex, createGrpcWebTransport({ baseUrl: process.env.MATCHING_API ?? 'http://localhost:8080' }));
const datasourceId = 'earth-c994825c6379';
const declarations: (Target & { name: string; key: string })[] = [
  { key: 'mare', name: 'Lunar mare', body: 'moon', longitude: -20, latitude: 30 },
  { key: 'tycho', name: 'Tycho vicinity', body: 'moon', longitude: -11.36, latitude: -43.31 },
  { key: 'lunar_pole', name: 'Lunar north polar terrain', body: 'moon', longitude: 0, latitude: 85 },
  { key: 'meridiani', name: 'Meridiani vicinity', body: 'mars', longitude: 23.47, latitude: .67 },
  { key: 'gale', name: 'Gale vicinity', body: 'mars', longitude: 137.4, latitude: -4.6 },
  { key: 'valles', name: 'Valles Marineris vicinity', body: 'mars', longitude: -60, latitude: -10 },
];
type Frozen = { target: typeof declarations[number]; fields: Field[]; criteria: Criterion[]; evidence: ReturnType<typeof applyMeasurements>; partial: boolean; query: ReturnType<typeof buildQuery>; measurements: unknown };
let targets: Frozen[];
try { targets = JSON.parse(await readFile(resolve(output, 'targets.json'), 'utf8')); }
catch (error) {
  if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error;
  const datasource = (await api.metadata({})).datasources.find(d => d.id === datasourceId);
  if (!datasource) throw new Error('Baseline datasource unavailable');
  const fields = fieldsFor(datasource); targets = [];
  for (const target of declarations) {
    const response = await api.samplePoint({ body: target.body === 'moon' ? PlanetaryBody.MOON : PlanetaryBody.MARS, longitude: target.longitude, latitude: target.latitude }, { timeoutMs: 30000 });
    const sample = { measurements: pointMeasurements(response) };
    const criteria: Criterion[] = fields.map(f => ({ key: f.key, target: null, enabled: false, tolerance: f.tolerance, weight: 1, min: null, max: null }));
    const evidence = applyMeasurements(criteria, target, sample);
    if (!evidence.measured) throw new Error(`No measured terrain for ${target.key}`);
    targets.push({ target, fields, criteria, evidence, partial: response.partial, query: buildQuery(fields, criteria), measurements: sample.measurements });
    console.log('Frozen target', target.key, evidence);
  }
  await writeFile(resolve(output, 'targets.json'), JSON.stringify(targets, (_key, value) => typeof value === 'bigint' ? value.toString() : value, 2) + '\n');
}
const namespace = process.env.EXPANSION_SCHEMA;
if (namespace && !/^raster_[a-f0-9]{32}$/.test(namespace)) throw new Error('Unexpected namespace');
const label = namespace ? 'expanded' : 'baseline';
const report = [];
for (const frozen of targets) {
  const query = namespace ? frozen.query.query.replace('FROM cells', `FROM "${namespace}".cells`) : frozen.query.query;
  const started = performance.now();
  const rows = resultRows(await api.query({ datasourceId, query, params: encodeParams(frozen.query.params) }, { timeoutMs: 15000 }));
  const elapsedMs = performance.now() - started;
  const patches = [];
  for (const row of rows) {
    if (typeof row.site_key !== 'string' || !(row.scores instanceof Uint8Array)) throw new Error('Bad score matrix');
    await writeFile(resolve(output, `${label}-${frozen.target.key}-${row.site_key}.bin`), row.scores);
    const view = new DataView(row.scores.buffer, row.scores.byteOffset, row.scores.byteLength);
    const scores: number[] = [];
    for (let i = 0; i < view.byteLength; i += 2) { const code = view.getUint16(i, true); if (code !== 65535) scores.push(code / 65534); }
    scores.sort((a, b) => a - b);
    patches.push({ siteKey: row.site_key, cells: view.byteLength / 2, valid: scores.length, missing: view.byteLength / 2 - scores.length,
      mean: scores.length ? scores.reduce((a, b) => a + b, 0) / scores.length : null,
      max: scores.at(-1) ?? null, p50: scores[Math.floor(scores.length * .5)] ?? null, p90: scores[Math.floor(scores.length * .9)] ?? null,
      high75: scores.filter(s => s >= .75).length, high90: scores.filter(s => s >= .9).length });
  }
  const result = { target: frozen.target, evidence: frozen.evidence, elapsedMs, patches: patches.sort((a, b) => (b.max ?? -1) - (a.max ?? -1)) };
  report.push(result);
  console.log(label, frozen.target.key, Math.round(elapsedMs), 'ms', result.patches.slice(0, 3).map(p => [p.siteKey, p.max]));
}
await writeFile(resolve(output, `${label}-matching.json`), JSON.stringify({ generatedAt: new Date().toISOString(), label, namespace, datasourceId, results: report }, null, 2) + '\n');
