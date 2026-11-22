import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { build } from 'vite'
import { hydrateAssets } from './hydrate-assets.mjs'

const root = fileURLToPath(new URL('../', import.meta.url))
const destination = resolve(root, process.argv[2] || 'exports/earth-connect-surface-lab.html')
await hydrateAssets('render')
const assets = [
  ['earthImageUrl', 'textures/earth-blue-marble.jpg', 'image/jpeg'],
  ['terrainDataUrl', 'textures/earth-terrain.bin', 'application/octet-stream'],
]
const embedded = await Promise.all(assets.map(async ([name, path, contentType]) => {
  const bytes = await readFile(resolve(root, 'public', path))
  return `export const ${name} = "data:${contentType};base64,${bytes.toString('base64')}"`
}))
const assetModule = '\0surface-lab-assets'
const noZstdPlugin = {
  name: 'stub-zstd-for-surface-lab',
  enforce: 'pre',
  resolveId(id) {
    if (id === '../earth-score/decode.ts') return '\0no-zstd'
  },
  load(id) {
    if (id === '\0no-zstd') return 'export function decompressZstd() { throw new Error("Zstd terrain is unavailable in standalone Surface Lab.") }'
  },
}
const result = await build({
  configFile: false,
  root,
  publicDir: false,
  logLevel: 'warn',
  define: { 'import.meta.env.VITE_SURFACE_LAB_STANDALONE': JSON.stringify('true') },
  plugins: [noZstdPlugin, {
    name: 'embed-surface-lab-assets',
    enforce: 'pre',
    resolveId(id, importer) {
      if (id === './labAssets.ts') return assetModule
      if (id === './terrain-worker-factory.ts' && importer === resolve(root, 'src/shader-test/terrain-loader.ts')) {
        return resolve(root, 'src/shader-test/embedded-terrain-worker.ts')
      }
    },
    load(id) {
      if (id === assetModule) return [...embedded, 'export const scoreManifestUrl = ""'].join('\n')
    },
  }],
  worker: { plugins: () => [noZstdPlugin] },
  build: {
    write: false,
    minify: true,
    cssCodeSplit: false,
    lib: {
      entry: resolve(root, 'src/shader-test/main.ts'),
      name: 'SurfaceLab',
      formats: ['iife'],
    },
  },
})

const outputs = (Array.isArray(result) ? result : [result]).flatMap((item) => item.output)
const scripts = outputs.filter((item) => item.type === 'chunk')
const styles = outputs.filter((item) => item.type === 'asset' && item.fileName.endsWith('.css'))
const unexpected = outputs.filter((item) => item.type === 'asset' && !item.fileName.endsWith('.css'))
if (scripts.length !== 1 || scripts[0].imports.length ||
    scripts[0].dynamicImports.some(file => file !== scripts[0].fileName) || unexpected.length) {
  throw new Error('The standalone export must contain one script and no external assets or imports.')
}

const css = styles.map((item) => typeof item.source === 'string' ? item.source : Buffer.from(item.source).toString()).join('\n')
const js = scripts[0].code.replace(/<\/script/gi, '<\\/script')
const source = await readFile(resolve(root, 'shader-test/index.html'), 'utf8')
const entry = '<script type="module" src="/src/shader-test/main.ts"></script>'
if (!source.includes(entry)) throw new Error('Could not find the standalone page entry point.')
const credits = await readFile(resolve(root, 'public/textures/CREDITS.txt'), 'utf8')
const html = source
  .replace('</head>', `<style>${css.replace(/<\/style/gi, '<\\/style')}</style>\n  </head>`)
  .replace(entry, () => `<script>${js}</script>`)
  .replace('</html>', `<!--\n${credits.replace(/-->/g, '-- >')}\n-->\n</html>`)
await mkdir(dirname(destination), { recursive: true })
await writeFile(destination, html)
console.log(`${destination}\n${(Buffer.byteLength(html) / 1024).toFixed(0)} KiB · image, CSS, JavaScript and worker embedded`)
