import { TILE_SIZE, type ScoreManifest } from './manifest.ts'

// MODIS sinusoidal grid used by the frozen Earth Index datasource.
const radius = 6371007.181
const span = Math.PI * radius / 18
const left = -Math.PI * radius
const top = Math.PI * radius / 2
export const PIXELS_PER_SITE = TILE_SIZE ** 2
export const GRID_WIDTH = 36 * TILE_SIZE
export const GRID_HEIGHT = 18 * TILE_SIZE
const radians = Math.PI / 180

function tileOrigin(siteKey: string) {
  return { x: Number(siteKey.slice(1, 3)) * TILE_SIZE, y: Number(siteKey.slice(4, 6)) * TILE_SIZE }
}

function gridRow(y: number) {
  const latitude = (GRID_HEIGHT / 2 - y - .5) * (180 / GRID_HEIGHT)
  const halfWidth = GRID_WIDTH / 2
  const longitudePerColumn = 180 / (halfWidth * Math.cos(latitude * radians))
  const validHalfWidth = Math.floor(180 / longitudePerColumn + .5)
  return { latitude, longitudePerColumn, validHalfWidth }
}

// One origin word per tile, followed by latitude, longitude scale and a
// whole-column validity boundary for each raster row. Compute the boundary
// in JavaScript precision so borderline cells agree with CPU picking.
export function createScoreGeometry(manifest: ScoreManifest) {
  const words = new Uint32Array(manifest.sites.length + GRID_HEIGHT * 3)
  const floats = new Float32Array(words.buffer)
  for (const [index, site] of manifest.sites.entries()) {
    const { x, y } = tileOrigin(site)
    words[index] = y * GRID_WIDTH + x
  }
  for (let y = 0; y < GRID_HEIGHT; y++) {
    const { latitude, longitudePerColumn, validHalfWidth } = gridRow(y)
    const offset = manifest.sites.length + y * 3
    floats[offset] = latitude
    floats[offset + 1] = longitudePerColumn
    words[offset + 2] = validHalfWidth
  }
  return words
}

export type ScoreAddress = Readonly<{
  address: number
  chunk: number
  row: number
  siteKey: string
  pixel: number
  x: number
  y: number
  center: Readonly<{ latitude: number; longitude: number }> | null
}>

export type ChunkLayout = Readonly<{ index: number; firstAddress: number; rows: number }>

export function createLayout(manifest: ScoreManifest) {
  let offset = 0
  const chunks: readonly ChunkLayout[] = Object.freeze(manifest.chunks.map(({ rows }, index) => {
    const range = Object.freeze({ index, firstAddress: offset, rows })
    offset += rows
    return range
  }))
  const sites = new Map(manifest.sites.map((site, i) => [site, i]))

  function locate(address: number): ScoreAddress | null {
    if (!Number.isInteger(address) || address < 0 || address >= manifest.rows) return null
    // Chunks are in address order even if they were uploaded out of order.
    const chunk = chunks.find(c => address < c.firstAddress + c.rows)!
    const siteKey = manifest.sites[Math.floor(address / PIXELS_PER_SITE)]
    const pixel = address % PIXELS_PER_SITE
    const x = pixel % TILE_SIZE
    const y = Math.floor(pixel / TILE_SIZE)
    const origin = tileOrigin(siteKey)
    const { latitude, longitudePerColumn, validHalfWidth } = gridRow(origin.y + y)
    const offset = origin.x + x + .5 - GRID_WIDTH / 2
    const center = Math.abs(offset) < validHalfWidth ? { latitude, longitude: offset * longitudePerColumn } : null
    return { address, chunk: chunk.index, row: address - chunk.firstAddress, siteKey, pixel, x, y, center }
  }

  function addressOf(siteKey: string, pixel: number) {
    const site = sites.get(siteKey)
    if (site === undefined || !Number.isInteger(pixel) || pixel < 0 || pixel >= PIXELS_PER_SITE) return null
    return locate(site * PIXELS_PER_SITE + pixel)
  }

  function addressAt(longitude: number, latitude: number) {
    if (!Number.isFinite(longitude) || !Number.isFinite(latitude) ||
        Math.abs(longitude) > 180 || Math.abs(latitude) > 90) return null
    const lat = latitude * radians
    const u = Math.min(36 - 1e-10, Math.max(0, (radius * longitude * radians * Math.cos(lat) - left) / span))
    const w = Math.min(18 - 1e-10, Math.max(0, (top - radius * lat) / span))
    const h = Math.floor(u), v = Math.floor(w)
    const x = Math.floor((u - h) * TILE_SIZE), y = Math.floor((w - v) * TILE_SIZE)
    return addressOf(`h${String(h).padStart(2, '0')}v${String(v).padStart(2, '0')}`, y * TILE_SIZE + x)
  }

  return Object.freeze({
    rows: manifest.rows, tileSize: TILE_SIZE, sites: manifest.sites, chunks, locate, addressOf, addressAt,
  })
}

export type ScoreLayout = ReturnType<typeof createLayout>
