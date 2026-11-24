import type { EarthScore, ScoreChunk } from '../earth-score/index.ts'
import { MISSING, SIZE } from './world-grid.ts'

export const tileCells = SIZE ** 2
export const scoreMipCount = Math.floor(Math.log2(SIZE)) + 1
export const maxBatchLayers = 64

export type ScoreBatch = {
  chunk: ScoreChunk
  chunkTile: number
  firstSite: number
  bank: number
  layer: number
  layers: number
}

export function computedScoreLayout(earth: Pick<EarthScore, 'chunks' | 'layout'>, bankSize: number, bankCount: number) {
  const { layout, chunks } = earth
  if (
    layout.tileSize !== SIZE ||
    layout.rows !== layout.sites.length * tileCells ||
    !layout.rows ||
    layout.sites.length > bankSize * bankCount
  ) {
    throw new Error('Earth Score does not match the lab’s native score grid.')
  }
  const lookup = new Uint16Array(36 * 18).fill(MISSING)
  layout.sites.forEach((site, index) => {
    const h = Number(site.slice(1, 3)),
      v = Number(site.slice(4, 6))
    lookup[v * 36 + h] = index
  })
  const batches: ScoreBatch[] = []
  let nextSite = 0
  for (const chunk of chunks) {
    const tiles = chunk.rows / tileCells
    if (
      chunk.firstAddress !== nextSite * tileCells ||
      !Number.isInteger(tiles) ||
      tiles < 1 ||
      chunk.scores.size < chunk.rows * 4
    )
      throw new Error('Invalid Earth Score chunk layout.')
    for (let chunkTile = 0; chunkTile < tiles; ) {
      const layer = nextSite % bankSize
      const layers = Math.min(tiles - chunkTile, bankSize - layer, maxBatchLayers)
      batches.push({ chunk, chunkTile, firstSite: nextSite, bank: Math.floor(nextSite / bankSize), layer, layers })
      nextSite += layers
      chunkTile += layers
    }
  }
  if (nextSite !== layout.sites.length) throw new Error('Incomplete Earth Score chunk layout.')
  const bytesPerTile = Array.from({ length: scoreMipCount }, (_, lod) => (SIZE >> lod) ** 2 * 2).reduce(
    (total, bytes) => total + bytes,
    0,
  )
  return { lookup, batches, bytes: bytesPerTile * layout.sites.length }
}
