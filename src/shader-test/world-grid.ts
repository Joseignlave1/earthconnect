import {
  GRID_CELL,
  GRID_COLS,
  GRID_LEFT,
  GRID_RADIUS,
  GRID_ROWS,
  GRID_SPAN,
  GRID_TOP,
  sinusoidalCellAddress,
} from '../earth-score/layout.ts'
import { TILE_SIZE } from '../earth-score/manifest.ts'
import { SCORE_MISSING } from '../earth-score/scoring.ts'

// Same grid as the backend's scripts/world_grid.py, sourced from earth-score/layout.ts.
export const RADIUS = GRID_RADIUS
export const SPAN = GRID_SPAN
export const SIZE = TILE_SIZE
export const CELL = GRID_CELL
export const LEFT = GRID_LEFT
export const TOP = GRID_TOP
export const MISSING = SCORE_MISSING

export type TileAddress = { h: number; v: number }

export function tileRow(v: number, y: number) {
  const latitude = (TOP - v * SPAN - (y + 0.5) * CELL) / RADIUS
  return { latitude, cos: Math.cos(latitude) }
}

export function tileX(h: number, x: number) {
  return LEFT + h * SPAN + (x + 0.5) * CELL
}

export function worldTiles(): TileAddress[] {
  const result: TileAddress[] = []
  for (let v = 0; v < GRID_ROWS; v++) {
    const y = v < GRID_ROWS / 2 ? SIZE - 1 : 0
    const limit = Math.PI * RADIUS * tileRow(v, y).cos
    for (let h = 0; h < GRID_COLS; h++) {
      const x = h < GRID_COLS / 2 ? SIZE - 1 : 0
      if (Math.abs(tileX(h, x)) <= limit) result.push({ h, v })
    }
  }
  return result
}

export function cellAddress(longitude: number, latitude: number) {
  return sinusoidalCellAddress(longitude, latitude)
}
