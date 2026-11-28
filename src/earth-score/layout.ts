import { TILE_SIZE, type ScoreManifest } from './manifest.ts'

// MODIS sinusoidal grid used by the frozen Earth Index datasource.
export const GRID_RADIUS = 6371007.181
export const GRID_COLS = 36
export const GRID_ROWS = 18
export const GRID_SPAN = (Math.PI * GRID_RADIUS) / GRID_ROWS
export const GRID_LEFT = -Math.PI * GRID_RADIUS
export const GRID_TOP = (Math.PI * GRID_RADIUS) / 2
export const GRID_CELL = GRID_SPAN / TILE_SIZE
export const PIXELS_PER_SITE = TILE_SIZE ** 2
export const GRID_WIDTH = GRID_COLS * TILE_SIZE
export const GRID_HEIGHT = GRID_ROWS * TILE_SIZE
export const SCORE_GEOMETRY_ROW_STRIDE = 3
export const SCORE_GEOMETRY_LATITUDE_OFFSET = 0
export const SCORE_GEOMETRY_LON_SCALE_OFFSET = 1
export const SCORE_GEOMETRY_VALID_HALF_WIDTH_OFFSET = 2
const radians = Math.PI / 180

export function parseSiteKey(siteKey: string) {
  return { h: Number(siteKey.slice(1, 3)), v: Number(siteKey.slice(4, 6)) }
}

function tileOrigin(siteKey: string) {
  const { h, v } = parseSiteKey(siteKey)
  return { x: h * TILE_SIZE, y: v * TILE_SIZE }
}

export function sinusoidalCellAddress(longitude: number, latitude: number) {
  if (
    !Number.isFinite(longitude) ||
    !Number.isFinite(latitude) ||
    Math.abs(longitude) > 180 ||
    Math.abs(latitude) > 90
  ) {
    return null
  }
  const lat = latitude * radians
  const u = Math.min(
    GRID_COLS - 1e-10,
    Math.max(0, (GRID_RADIUS * longitude * radians * Math.cos(lat) - GRID_LEFT) / GRID_SPAN),
  )
  const w = Math.min(GRID_ROWS - 1e-10, Math.max(0, (GRID_TOP - GRID_RADIUS * lat) / GRID_SPAN))
  const h = Math.floor(u)
  const v = Math.floor(w)
  return {
    h,
    v,
    x: Math.min(TILE_SIZE - 1, Math.floor((u - h) * TILE_SIZE)),
    y: Math.min(TILE_SIZE - 1, Math.floor((w - v) * TILE_SIZE)),
  }
}

function gridRow(y: number) {
  const latitude = (GRID_HEIGHT / 2 - y - 0.5) * (180 / GRID_HEIGHT)
  const halfWidth = GRID_WIDTH / 2
  const longitudePerColumn = 180 / (halfWidth * Math.cos(latitude * radians))
  const validHalfWidth = Math.floor(180 / longitudePerColumn + 0.5)
  return { latitude, longitudePerColumn, validHalfWidth }
}

// One origin word per tile, followed by latitude, longitude scale and a
// whole-column validity boundary for each raster row. Compute the boundary
// in JavaScript precision so borderline cells agree with CPU picking.
export function createScoreGeometry(manifest: ScoreManifest) {
  const words = new Uint32Array(manifest.sites.length + GRID_HEIGHT * SCORE_GEOMETRY_ROW_STRIDE)
  const floats = new Float32Array(words.buffer)
  for (const [index, site] of manifest.sites.entries()) {
    const { x, y } = tileOrigin(site)
    words[index] = y * GRID_WIDTH + x
  }
  for (let y = 0; y < GRID_HEIGHT; y++) {
    const { latitude, longitudePerColumn, validHalfWidth } = gridRow(y)
    const offset = manifest.sites.length + y * SCORE_GEOMETRY_ROW_STRIDE
    floats[offset + SCORE_GEOMETRY_LATITUDE_OFFSET] = latitude
    floats[offset + SCORE_GEOMETRY_LON_SCALE_OFFSET] = longitudePerColumn
    words[offset + SCORE_GEOMETRY_VALID_HALF_WIDTH_OFFSET] = validHalfWidth
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
  const chunks: readonly ChunkLayout[] = Object.freeze(
    manifest.chunks.map(({ rows }, index) => {
      const range = Object.freeze({ index, firstAddress: offset, rows })
      offset += rows
      return range
    }),
  )
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
    const offset = origin.x + x + 0.5 - GRID_WIDTH / 2
    const center = Math.abs(offset) < validHalfWidth ? { latitude, longitude: offset * longitudePerColumn } : null
    return { address, chunk: chunk.index, row: address - chunk.firstAddress, siteKey, pixel, x, y, center }
  }

  function addressOf(siteKey: string, pixel: number) {
    const site = sites.get(siteKey)
    if (site === undefined || !Number.isInteger(pixel) || pixel < 0 || pixel >= PIXELS_PER_SITE) return null
    return locate(site * PIXELS_PER_SITE + pixel)
  }

  function addressAt(longitude: number, latitude: number) {
    const cell = sinusoidalCellAddress(longitude, latitude)
    if (!cell) return null
    return addressOf(
      `h${String(cell.h).padStart(2, '0')}v${String(cell.v).padStart(2, '0')}`,
      cell.y * TILE_SIZE + cell.x,
    )
  }

  return Object.freeze({
    rows: manifest.rows,
    tileSize: TILE_SIZE,
    sites: manifest.sites,
    chunks,
    locate,
    addressOf,
    addressAt,
  })
}

export type ScoreLayout = ReturnType<typeof createLayout>
