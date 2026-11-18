// Same grid as the backend's scripts/world_grid.py, not a longitude/latitude proxy.
export const RADIUS = 6371007.181
export const SPAN = Math.PI * RADIUS / 18
export const SIZE = 278
export const CELL = SPAN / SIZE
export const LEFT = -Math.PI * RADIUS
export const TOP = Math.PI * RADIUS / 2
export const MISSING = 65535

export type TileAddress = { h: number; v: number }

export function tileRow(v: number, y: number) {
  const latitude = (TOP - v * SPAN - (y + .5) * CELL) / RADIUS
  return { latitude, cos: Math.cos(latitude) }
}

export function tileX(h: number, x: number) { return LEFT + h * SPAN + (x + .5) * CELL }

export function worldTiles(): TileAddress[] {
  const result: TileAddress[] = []
  for (let v = 0; v < 18; v++) {
    const y = v < 9 ? SIZE - 1 : 0
    const limit = Math.PI * RADIUS * tileRow(v, y).cos
    for (let h = 0; h < 36; h++) {
      const x = h < 18 ? SIZE - 1 : 0
      if (Math.abs(tileX(h, x)) <= limit) result.push({ h, v })
    }
  }
  return result
}

export function cellAddress(longitude: number, latitude: number) {
  if (!Number.isFinite(longitude) || !Number.isFinite(latitude) || Math.abs(latitude) > 90 || Math.abs(longitude) > 180) return null
  const lat = latitude * Math.PI / 180
  const x = RADIUS * longitude * Math.PI / 180 * Math.cos(lat)
  const y = RADIUS * lat
  const u = Math.min(36 - 1e-10, Math.max(0, (x - LEFT) / SPAN))
  const v = Math.min(18 - 1e-10, Math.max(0, (TOP - y) / SPAN))
  const h = Math.floor(u), row = Math.floor(v)
  return { h, v: row, x: Math.min(SIZE - 1, Math.floor((u - h) * SIZE)), y: Math.min(SIZE - 1, Math.floor((v - row) * SIZE)) }
}
