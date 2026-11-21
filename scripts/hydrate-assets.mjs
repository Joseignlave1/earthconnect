import { createHash, randomUUID } from 'node:crypto'
import { link, mkdir, readFile, rm, writeFile } from 'node:fs/promises'
import { dirname } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'
import release from '../cdn/pinned-release.json' with { type: 'json' }

const root = new URL('../', import.meta.url)
const cdn = 'https://earth-connect-assets.pages.dev/'
const worldDirectory = 'compute-data/world/'
const world = JSON.parse(await readFile(new URL(worldDirectory + 'manifest.json', root), 'utf8'))
const worldPaths = world.chunks.map(chunk => {
  const path = worldDirectory + chunk.file
  if (release.assets[path]?.bytes !== chunk.compressedBytes) {
    throw new Error(`World manifest does not match the pinned release at ${path}.`)
  }
  return path
})
const ignoredPaths = new Set(['data/surface-basins.bin'])
const renderPaths = Object.keys(release.assets).filter(path => !path.startsWith(worldDirectory) && !ignoredPaths.has(path))
const groups = {
  all: [...renderPaths, ...worldPaths],
  render: renderPaths,
  world: worldPaths,
}

async function hydrate(path) {
  const asset = release.assets[path]
  const local = new URL(path.startsWith(worldDirectory) ? path : 'public/' + path, root)
  const verify = bytes => bytes.length === asset.bytes &&
    createHash('sha256').update(bytes).digest('hex') === asset.sha256
  function verifyLocal(bytes) {
    if (!verify(bytes)) {
      throw new Error(`Integrity check failed for local ${path}. Move the file aside before downloading the pinned copy.`)
    }
  }
  const existing = await readFile(local).catch(error => {
    if (error.code !== 'ENOENT') throw error
  })
  if (existing) return verifyLocal(existing)

  const response = await fetch(new URL(asset.key, cdn), { signal: AbortSignal.timeout(120_000) })
  if (!response.ok) throw new Error(`${asset.key} returned HTTP ${response.status}`)
  const bytes = Buffer.from(await response.arrayBuffer())
  if (!verify(bytes)) throw new Error(`Integrity check failed for CDN asset ${asset.key}.`)

  const temporary = new URL(`${local.href}.${randomUUID()}.tmp`)
  await mkdir(dirname(fileURLToPath(local)), { recursive: true })
  try {
    await writeFile(temporary, bytes, { flag: 'wx' })
    await link(temporary, local)
    console.log(`Fetched ${path}`)
  } catch (error) {
    if (error.code !== 'EEXIST') throw error
    verifyLocal(await readFile(local))
  } finally {
    await rm(temporary, { force: true })
  }
}

export async function hydrateAssets(group = 'all') {
  if (!Object.hasOwn(groups, group)) throw new Error('Usage: npm run fetch:assets -- [all|render|world]')
  const files = groups[group]
  for (let i = 0; i < files.length; i += 4) {
    await Promise.all(files.slice(i, i + 4).map(hydrate))
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  await hydrateAssets(process.argv[2])
}
