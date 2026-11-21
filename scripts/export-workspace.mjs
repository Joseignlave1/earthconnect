import { createReadStream, createWriteStream } from 'node:fs'
import { mkdir, readFile, rename, rm, stat } from 'node:fs/promises'
import { dirname, resolve } from 'node:path'
import { pipeline } from 'node:stream/promises'
import { fileURLToPath } from 'node:url'
import react from '@vitejs/plugin-react'
import { build, loadEnv } from 'vite'
import { hydrateAssets } from './hydrate-assets.mjs'

const root = fileURLToPath(new URL('../', import.meta.url))
const args = process.argv.slice(2)
const offline = args.includes('--offline')
if (offline) await hydrateAssets()
const destination = resolve(root, args.find(arg => arg !== '--offline') ||
  `exports/earth-connect-workspace${offline ? '-offline' : ''}.html`)
const temporary = `${destination}.${process.pid}.tmp`
const read = file => readFile(resolve(root, file), 'utf8')
const world = 'compute-data/world/'
const terrain = 'textures/'
const image = `${terrain}earth-blue-marble.jpg`
const earth = `${terrain}earth-terrain.bin`
const sky = `${terrain}nasa-deep-star-map.webp`
const textures = [image, earth, sky]
const assets = [...textures, `${terrain}moon-terrain.bin`, `${terrain}mars-terrain.bin`]
const virtualAsset = '\0workspace-asset:'

const wasm = await readFile(resolve(root, 'src/earth-score/zstd.wasm'))
const manifest = offline ? JSON.parse(await read(`${world}manifest.json`)) : undefined

const credits = await Promise.all([
  'public/textures/CREDITS.txt',
  `${world}CREDITS.md`,
  'src/workspace/fonts/DM-Sans-OFL.txt',
  'tools/earth-score-decoder/ZSTD-LICENSE.txt',
  'tools/earth-score-decoder/ZSTD-SYS-LICENSE.txt',
].map(read))

function inlineDecoderWasm() {
  const id = '\0workspace-decoder-wasm'
  const data = `data:application/wasm;base64,${wasm.toString('base64')}`
  return {
    name: 'inline-workspace-decoder-wasm',
    enforce: 'pre',
    resolveId(source) { if (source === './zstd.wasm?url&no-inline') return id },
    load(source) { if (source === id) return `export default ${JSON.stringify(data)}` },
  }
}

const env = loadEnv('production', root, 'VITE_EARTH_INDEX_')
const result = await build({
  root,
  configFile: false,
  publicDir: false,
  logLevel: 'warn',
  define: {
    WORKSPACE_OFFLINE: JSON.stringify(offline),
    'process.env.NODE_ENV': JSON.stringify('production'),
    'import.meta.env.VITE_WORKSPACE_STANDALONE': JSON.stringify(true),
    ...(env.VITE_EARTH_INDEX_API_URL ? {
      'import.meta.env.VITE_EARTH_INDEX_API_URL': JSON.stringify(env.VITE_EARTH_INDEX_API_URL),
    } : {}),
  },
  plugins: [
    react(),
    inlineDecoderWasm(),
    {
      name: 'omit-lab-render-assets',
      enforce: 'pre',
      resolveId(source) {
        const path = source.slice(1, -4)
        if (source.startsWith('/') && source.endsWith('?url') && assets.includes(path)) return virtualAsset + path
      },
      load(id) {
        if (!id.startsWith(virtualAsset)) return
        return 'export default ""'
      },
    },
    {
      name: 'use-embedded-workers',
      enforce: 'pre',
      resolveId(source, importer) {
        if (source === './decoder-worker.ts' && importer === resolve(root, 'src/earth-score/decoder-pool.ts')) {
          return resolve(root, 'src/workspace/embeddedDecoder.ts')
        }
        if (source === './terrain-worker-factory.ts' && importer === resolve(root, 'src/shader-test/terrain-loader.ts')) {
          return resolve(root, 'src/shader-test/embedded-terrain-worker.ts')
        }
      },
    },
  ],
  worker: { plugins: () => [inlineDecoderWasm()] },
  build: {
    write: false,
    cssCodeSplit: false,
    lib: { entry: resolve(root, 'src/workspace/export.tsx'), name: 'EarthConnectWorkspace', formats: ['iife'] },
  },
})

const outputs = (Array.isArray(result) ? result : [result]).flatMap(item => item.output)
const scripts = outputs.filter(item => item.type === 'chunk')
const styles = outputs.filter(item => item.type === 'asset' && item.fileName.endsWith('.css'))
const unexpected = outputs.filter(item => item.type === 'asset' && !item.fileName.endsWith('.css'))
// Rolldown lists imports inlined into the current script as self-references.
if (scripts.length !== 1 || scripts[0].imports.length ||
    scripts[0].dynamicImports.some(file => file !== scripts[0].fileName) || unexpected.length) {
  throw new Error(`Workspace export must have one script and no sidecars. Unexpected output: ${unexpected.map(item => item.fileName).join(', ')}`)
}
const css = styles.map(item => item.source.toString()).join('\n').replace(/<\/style/gi, '<\\/style')
const js = scripts[0].code.replace(/<\/script/gi, '<\\/script')

// Base64 stays outside executable code, and is streamed to keep build memory
// independent of the size of the world. Each script is separately decodable.
async function* html() {
  yield `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Earth Connect · Find</title>
<link rel="icon" href="data:,">
<style>${css}</style>
</head>
<body>
<div id="root"></div>
<noscript>Earth Connect requires JavaScript and WebGPU.</noscript>\n`
  const files = manifest ? [
    ...assets.map(path => [path, `public/${path}`]),
    ...['manifest.json', ...manifest.chunks.map(chunk => chunk.file)].map(file => [world + file, world + file]),
  ] : []
  for (const [key, path] of files) {
    for await (const chunk of createReadStream(resolve(root, path), { highWaterMark: 768 * 1024 })) {
      yield `<script type="application/octet-stream" data-workspace-asset="${key}">${chunk.toString('base64')}</script>\n`
    }
  }
  yield `<script>${js}</script>\n<!--\n${credits.join('\n\n').replace(/-->/g, '-- >')}\n-->\n</body>\n</html>\n`
}

await mkdir(dirname(destination), { recursive: true })
try {
  await pipeline(html(), createWriteStream(temporary, { flags: 'wx' }))
  await rename(temporary, destination)
} catch (error) {
  await rm(temporary, { force: true })
  throw error
}
const { size } = await stat(destination)
console.log(`${destination}\n${(size / 1024 ** 2).toFixed(1)} MiB · ${offline ? 'data embedded for offline use' : 'data loaded from the Earth Connect CDN'}`)
