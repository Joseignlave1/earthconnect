// Build-only instrumentation. The application sources and deployed image are untouched.
import { build, preview, type Plugin } from 'vite';
import ts from 'typescript';
import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';

const root = resolve(import.meta.dirname, '..');
const output = process.env.BROWSER_PROFILE_DIR ?? '/tmp/earth-browser-profile';
const runtime = resolve(import.meta.dirname, 'browser-profile-runtime.ts');
const watched: Record<string, Set<string>> = {
  'scores.ts': new Set(['prepareScores', 'scoreSummary', 'draw', 'refresh', 'show', 'threshold']),
  'score-overviews.ts': new Set(['scorePyramid']),
  'score-atlas.ts': new Set(['scoreAtlas', 'registerScoreProjection']),
  'score-raster.ts': new Set(['initialize', 'upload', 'render']),
  'picks.ts': new Set(['scoreDistribution', 'findAreas']),
  'coverage.ts': new Set(['clipCoverage']),
};

const instrumentation: Plugin = {
  name: 'earth-build-only-profile',
  enforce: 'pre',
  resolveId(id) {
    if (id === 'virtual:earth-profile') return runtime;
  },
  transform(source, id) {
    if (!id.startsWith(resolve(root, 'src') + '/')) return;
    const filename = id.split('/').at(-1)!;
    if (filename === 'App.svelte') {
      return source
        .replace('import { onMount }', 'import { onMount, tick }')
        .replace(
          '<script lang="ts">',
          '<script lang="ts">\nimport { profile as __profile } from "virtual:earth-profile";',
        )
        .replace(
          'const response = await api.query(',
          'const __rpcStart = performance.now(); const response = await api.query(',
        )
        .replace(
          'if (revision !== queryRevision) return;',
          "__profile.asyncRecord('rpc.connect', performance.now() - __rpcStart); if (revision !== queryRevision) return;",
        )
        .replace(
          'onMount(() => {',
          `onMount(() => {
          __profile.app = {
            ready: () => !!currentDatasource,
            run: async (request) => {
              queryMode = 'custom'; sql = request.query; params = JSON.stringify(request.params);
              await tick(); await runQuery(); await tick();
              if (queryError) throw new Error(queryStatus);
              return {status: queryStatus, patches: patches.length};
            },
            clear: async () => { invalidateResults(); await tick(); },
            hashes: async () => Object.fromEntries(await Promise.all(patches.map(async p => [
              p.grid.siteKey,
              Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', p.scores.buffer)), b => b.toString(16).padStart(2, '0')).join(''),
            ]))),
          };`,
        );
    }
    if (filename === 'maps.ts') {
      return (
        'import { profile as __profile } from "virtual:earth-profile";\n' +
        source.replace(
          "const scores = config.id === 'earth' ? earthScores(map) : undefined;",
          "const scores = config.id === 'earth' ? earthScores(map) : undefined;\n__profile.maps.set(config.id, {map, scores, reset});",
        )
      );
    }
    if (!watched[filename]) return;
    const parsed = ts.createSourceFile(id, source, ts.ScriptTarget.Latest, true);
    const edits: Array<{ at: number; text: string }> = [];
    const visit = (node: ts.Node) => {
      const isFunction =
        ts.isFunctionDeclaration(node) || ts.isMethodDeclaration(node) || ts.isArrowFunction(node);
      if (isFunction) {
        const name = ts.isArrowFunction(node)
          ? ts.isVariableDeclaration(node.parent)
            ? node.parent.name.getText(parsed)
            : ''
          : (node.name?.getText(parsed) ?? '');
        if (watched[filename].has(name) && node.body && ts.isBlock(node.body)) {
          const label = `${filename}:${name}`;
          edits.push({
            at: node.body.getStart(parsed) + 1,
            text: '\nconst __earthTimer = __profile.begin(); try {\n',
          });
          edits.push({
            at: node.body.end - 1,
            text: `\n} finally { __profile.end(${JSON.stringify(label)}, __earthTimer); }\n`,
          });
        }
      }
      // Separate the per-region CRS registration and source construction costs.
      if (
        filename === 'scores.ts' &&
        ((ts.isCallExpression(node) && node.expression.getText(parsed) === 'register') ||
          (ts.isNewExpression(node) &&
            ['Static', 'ImageLayer'].includes(node.expression.getText(parsed))))
      ) {
        const label = ts.isCallExpression(node)
          ? 'projection.register'
          : `construct.${node.expression.getText(parsed)}`;
        edits.push({
          at: node.getStart(parsed),
          text: `__profile.measure(${JSON.stringify(label)}, () => (`,
        });
        edits.push({ at: node.end, text: '))' });
      }
      ts.forEachChild(node, visit);
    };
    visit(parsed);
    let code = source;
    for (const edit of edits.sort((a, b) => b.at - a.at))
      code = code.slice(0, edit.at) + edit.text + code.slice(edit.at);
    if (filename === 'coverage.ts')
      code = code.replace(
        'color.changed();',
        `
      __profile.coverage = {
        polygons: polygons.length,
        rings: polygons.reduce((n, p) => n + p.length, 0),
        vertices: polygons.reduce((n, p) => n + p.reduce((m, ring) => m + ring.length, 0), 0),
      };
      color.changed();`,
      );
    return 'import { profile as __profile } from "virtual:earth-profile";\n' + code;
  },
};

await mkdir(output, { recursive: true });
const request = JSON.parse(
  await readFile(resolve(root, '../research/world-4km/pipeline-profile/query.json'), 'utf8'),
);
await writeFile(resolve(output, 'request.json'), JSON.stringify(request));
await build({
  root,
  configFile: resolve(root, 'vite.config.ts'),
  plugins: [instrumentation],
  build: { outDir: resolve(output, 'dist'), emptyOutDir: true, sourcemap: false },
});
await writeFile(resolve(output, 'dist/profile-request.json'), JSON.stringify(request));
const server = await preview({
  root,
  configFile: resolve(root, 'vite.config.ts'),
  build: { outDir: resolve(output, 'dist') },
  preview: { host: '0.0.0.0', port: 5174, strictPort: true },
});
server.printUrls();
