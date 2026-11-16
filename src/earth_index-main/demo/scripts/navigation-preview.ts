// Export the actual Svelte controls with a compressed real query matrix.
import { writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { gzipSync } from 'node:zlib';
import { build } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';
import { api } from '../src/api.ts';
import { PlanetaryBody } from '../src/gen/earth_index_pb.ts';
import { pointMeasurements } from '../src/characteristics.ts';
import { fieldsFor, buildQuery } from '../src/query.ts';
import { applyMeasurements } from '../src/matching.ts';
import { encodeParams, resultRows } from '../src/values.ts';
import { prepareScores } from '../src/scores.ts';
import type { Criterion, Target } from '../src/types.ts';

const ds = (await api.metadata({})).datasources
  .filter((d) => d.body === 'earth')
  .sort(
    (a, b) =>
      b.coverage.length - a.coverage.length ||
      (b.relations.find((r) => r.name === 'cells')?.columns.length ?? 0) -
        (a.relations.find((r) => r.name === 'cells')?.columns.length ?? 0),
  )[0];
if (!ds) throw new Error('No Earth datasource');
const target: Target = { body: 'mars', longitude: 23.47, latitude: 0.67 };
const response = await api.samplePoint(
  { body: PlanetaryBody.MARS, longitude: target.longitude, latitude: target.latitude },
  { timeoutMs: 25000 },
);
const fields = fieldsFor(ds);
const criteria: Criterion[] = fields.map((f) => ({
  key: f.key,
  target: null,
  enabled: false,
  tolerance: f.tolerance,
  weight: 1,
}));
const evidence = applyMeasurements(criteria, target, { measurements: pointMeasurements(response) });
const q = buildQuery(fields, criteria),
  start = performance.now();
const rows = resultRows(
  await api.query(
    { datasourceId: ds.id, query: q.query, params: encodeParams(q.params) },
    { timeoutMs: 15000 },
  ),
);
const queryMs = performance.now() - start;
const patches = prepareScores(rows, ds);
const matrix = Buffer.concat(
  rows.map((row) => {
    if (!(row.scores instanceof Uint8Array)) throw new Error('Expected packed scores');
    return Buffer.from(row.scores);
  }),
);
const compressed = gzipSync(matrix).toString('base64');
const grids = patches.map(({ grid: { siteKey, width, height } }) => ({ siteKey, width, height }));
const entry = `import {mount} from 'svelte';
import NavigationExport from ${JSON.stringify(fileURLToPath(new URL('../src/components/NavigationExport.svelte', import.meta.url)))};
(async()=>{
 const compressed=Uint8Array.from(atob('${compressed}'),c=>c.charCodeAt(0));
 const raw=new Uint8Array(await new Response(new Blob([compressed]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer());
 const grids=${JSON.stringify(grids)}; let cursor=0;
 const patches=grids.map(grid=>{const scores=new Uint16Array(grid.width*grid.height),v=new DataView(raw.buffer,cursor,scores.length*2);for(let i=0;i<scores.length;i++)scores[i]=v.getUint16(i*2,true);cursor+=scores.length*2;return{grid,scores};});
 mount(NavigationExport,{target:document.querySelector('#controls'),props:{patches}});
})();`;
const entryId = fileURLToPath(new URL('./virtual-navigation.ts', import.meta.url));
const bundle = await build({
  root: fileURLToPath(new URL('..', import.meta.url)),
  configFile: false,
  plugins: [
    svelte(),
    {
      name: 'navigation-export',
      resolveId: (id) => (id === entryId ? entryId : null),
      load: (id) => (id === entryId ? entry : null),
    },
  ],
  build: {
    write: false,
    lib: { entry: entryId, formats: ['iife'], name: 'NavigationExport' },
    minify: true,
  },
});
const output = Array.isArray(bundle) ? bundle[0] : bundle;
if (!('output' in output)) throw new Error('Unexpected navigation build output');
const script = output.output.find((asset) => asset.type === 'chunk');
if (!script || script.type !== 'chunk') throw new Error('No navigation bundle');
const html = `<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>
:root{color-scheme:light dark}body{margin:0;color:var(--foreground);font:13px var(--font-sans)}h3{font-size:15px;margin:0 0 7px}p{color:var(--muted-foreground);font-size:12px;margin:0 0 12px}.pick-controls{display:grid;gap:9px}svg{display:block;width:100%;height:86px;cursor:pointer}.distribution-labels,.threshold-label{display:flex;justify-content:space-between}.distribution-labels{font-size:11px;color:var(--muted-foreground)}input[type=range]{width:100%;margin:0;accent-color:var(--chart-1)}.pick-navigation{display:flex;gap:6px;align-items:center}select{min-width:0;flex:1;width:0}button,select{font:inherit;padding:5px;background:var(--secondary);color:var(--foreground);border:1px solid var(--border);border-radius:var(--radius)}output{font-variant-numeric:tabular-nums}#selected-area{display:block;margin-top:8px;color:var(--muted-foreground);font-size:12px}button:disabled{opacity:.4}
</style></head><body><h3>Score distribution and top areas</h3><p>Mars target: 23.47°, 0.67° · ${ds.coverage.length} Earth patches · ${evidence.approximate ?? 0} approximate thermal comparisons</p><div id="controls"></div><script>${script.code.replaceAll('</script', '<\\/script')}</script></body></html>`;
const path = new URL('../../research/pilots/database/result-navigation.html', import.meta.url);
await writeFile(path, html);
await writeFile(
  new URL('../../research/pilots/database/result-navigation.json', import.meta.url),
  JSON.stringify(
    {
      datasource: ds.id,
      target,
      evidence,
      queryMs,
      criteria: criteria.filter((c) => c.enabled),
      bytes: matrix.length,
    },
    null,
    2,
  ) + '\n',
);
console.log(
  JSON.stringify({
    path: path.pathname,
    datasource: ds.id,
    evidence,
    queryMs: Math.round(queryMs),
    bytes: matrix.length,
  }),
);
