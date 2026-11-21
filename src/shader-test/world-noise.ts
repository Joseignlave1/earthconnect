import { maxPyramid, perlin } from './noise.ts'
import { MISSING, RADIUS, SIZE, tileRow, tileX, worldTiles, type TileAddress } from './world-grid.ts'

export type WorldNoise = {
  tiles: (TileAddress & { levels: ReturnType<typeof maxPyramid> })[]
  histogram: Uint32Array
  lookup: Uint16Array
  valid: number
  best: number
  bytes: number
  milliseconds: number
}

export function* generateWorldTiles(frequency: number, seed: number): Generator<[number,number], WorldNoise> {
  const started = performance.now(), noise = perlin(seed)
  const addresses = worldTiles(), lookup = new Uint16Array(36 * 18).fill(MISSING)
  const histogram = new Uint32Array(MISSING)
  const tiles: WorldNoise['tiles'] = []
  let valid = 0, best = 0, bytes = 0
  for (const { h, v } of addresses) {
    const scores = new Uint16Array(SIZE * SIZE).fill(MISSING)
    for (let y = 0; y < SIZE; y++) {
      const { latitude, cos } = tileRow(v, y), py = Math.sin(latitude)
      const limit = Math.PI * RADIUS * cos
      for (let x = 0; x < SIZE; x++) {
        const projectedX = tileX(h, x)
        if (Math.abs(projectedX) > limit) continue
        const longitude = projectedX / (RADIUS * cos)
        const px = cos * Math.sin(longitude), pz = cos * Math.cos(longitude)
        let sum = 0, amplitude = 1, normalization = 0, scale = frequency
        for (let octave = 0; octave < 5; octave++) {
          sum += amplitude * noise(px * scale + 17.3, py * scale + 31.7, pz * scale + 8.9)
          normalization += amplitude; amplitude *= .5; scale *= 2
        }
        const code = Math.round(Math.max(0, Math.min(1, .5 + sum / normalization * .85)) * 65534)
        scores[y * SIZE + x] = code
        histogram[code]++; valid++; best = Math.max(best, code)
      }
    }
    const levels = maxPyramid(scores, SIZE, SIZE)
    lookup[v * 36 + h] = tiles.length
    tiles.push({ h, v, levels })
    bytes += levels.reduce((sum, level) => sum + level.scores.byteLength, 0)
    yield [tiles.length,addresses.length]
  }
  return { tiles, histogram, lookup, valid, best, bytes, milliseconds: performance.now() - started }
}
