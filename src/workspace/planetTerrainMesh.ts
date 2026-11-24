import { createMesh, type Vec3 } from '../shader-test/geometry.ts'
import { elevationAt, poleElevations, type Terrain } from '../shader-test/terrain.ts'
import { bodyRadius } from './solarSystem.ts'

export type TerrainBody = 'moon' | 'mars'

// sRGB fallback albedo stops linearized in the fragment shader when photographic albedo is absent.
export const PLANET_TERRAIN_SETTINGS = {
  moon: { relief: 2.2, interval: 1500, low: [.24, .25, .27], high: [.64, .65, .68] },
  mars: { relief: 3.0, interval: 2400, low: [.28, .12, .07], high: [.72, .39, .24] },
} as const

const clamp = (value: number, low: number, high: number) => Math.min(high, Math.max(low, value))

function normalAt(longitude: number, latitude: number): Vec3 {
  if (Math.abs(latitude) === Math.PI / 2) return [0, Math.sign(latitude), 0]
  const c = Math.cos(latitude)
  return [c * Math.sin(longitude), Math.sin(latitude), c * Math.cos(longitude)]
}

const planetColumns = 512, planetRows = 256
let sharedPlanetGrid: ReturnType<typeof createMesh> | undefined

function getPlanetGrid() {
  return (sharedPlanetGrid ??= createMesh(planetColumns, planetRows))
}

export type PlanetTerrain = ReturnType<typeof buildPlanetTerrain>

export function buildPlanetTerrain(body: TerrainBody, terrain: Terrain, retainElevation = true) {
  const { relief } = PLANET_TERRAIN_SETTINGS[body]
  const radius = bodyRadius(body)
  const columns = planetColumns, rows = planetRows
  const { uv, indices } = getPlanetGrid()
  const [south, north] = poleElevations(terrain)
  const readElevation = (longitude: number, latitude: number) => {
    if (latitude <= -Math.PI / 2) return south
    if (latitude >= Math.PI / 2) return north
    return elevationAt(terrain, longitude, latitude)
  }
  const positions = new Float32Array(uv.length / 2 * 3)
  for (let vertex = 0; vertex < uv.length / 2; vertex++) {
    const longitude = (uv[vertex * 2] - .5) * 2 * Math.PI
    const latitude = (uv[vertex * 2 + 1] - .5) * Math.PI
    const magnitude = radius + readElevation(longitude, latitude) * relief / 1000
    const [x, y, z] = normalAt(longitude, latitude)
    positions.set([magnitude * x, magnitude * y, magnitude * z], vertex * 3)
  }

  const positionAt = (longitude: number, latitude: number): Vec3 => {
    const x = clamp((longitude / (2 * Math.PI) + .5) * columns, 0, columns)
    const y = clamp((latitude / Math.PI + .5) * rows, 0, rows)
    const column = Math.min(columns - 1, Math.floor(x))
    const row = Math.min(rows - 1, Math.floor(y))
    const dx = x - column, dy = y - row
    const bottomLeft = row * (columns + 1) + column
    const topLeft = bottomLeft + columns + 1
    // These weights use the same bottom-right to top-left diagonal as createMesh.
    const interpolate = (axis: number) => {
      const bottomRight = positions[(bottomLeft + 1) * 3 + axis]
      const top = positions[topLeft * 3 + axis]
      return dx + dy <= 1
        ? positions[bottomLeft * 3 + axis] * (1 - dx - dy) + bottomRight * dx + top * dy
        : positions[(topLeft + 1) * 3 + axis] * (dx + dy - 1) + bottomRight * (1 - dy) + top * (1 - dx)
    }
    return [interpolate(0), interpolate(1), interpolate(2)]
  }
  const blocks: { x: number; y: number; endX: number; endY: number; min: Vec3; max: Vec3 }[] = []
  for (let y = 0; y < rows; y += 8) for (let x = 0; x < columns; x += 8) {
    const endX = Math.min(x + 8, columns), endY = Math.min(y + 8, rows)
    const min: Vec3 = [Infinity, Infinity, Infinity], max: Vec3 = [-Infinity, -Infinity, -Infinity]
    for (let row = y; row <= endY; row++) for (let col = x; col <= endX; col++) {
      const i = (row * (columns + 1) + col) * 3
      for (let axis = 0; axis < 3; axis++) {
        const value = positions[i + axis]
        if (value < min[axis]) min[axis] = value
        if (value > max[axis]) max[axis] = value
      }
    }
    blocks.push({ x, y, endX, endY, min, max })
  }

  return { radius, columns, rows, positions, indices, blocks, elevationAt: retainElevation ? readElevation : (() => 0), positionAt }
}
