import { maxPyramid } from './noise.ts'
import { CELL, MISSING, SIZE, worldTiles } from './world-grid.ts'
import type { WorldNoise } from './world-noise.ts'

export type SnapshotMetadata = {
  name: string
  description: string
  fetchedAt: string
  endpoint: string
  datasourceId: string
  query: string
  params: Record<string,number>
  tileSize: number
  cellMetres: number
  tiles: [number,number][]
  encoding: 'u16le' | 'row-delta-byteplanes'
  sha256: string
  valid: number
  best: number
}

export function decodeScoreSnapshot(buffer: ArrayBuffer, progress: (done: number,total: number) => void = () => {}): WorldNoise {
  const tiles = decodeSnapshotTiles(buffer)
  let tile = tiles.next()
  while (!tile.done) {
    progress(...tile.value)
    tile = tiles.next()
  }
  return tile.value
}

export function* decodeSnapshotTiles(buffer: ArrayBuffer): Generator<[number,number], WorldNoise> {
  const started = performance.now(), view = new DataView(buffer)
  if (buffer.byteLength < 8 || view.getUint32(0) !== 0x45435331) throw new Error('Invalid score snapshot header.')
  const length = view.getUint32(4,true), offset = (8 + length + 3) & ~3
  if (length < 2 || offset > buffer.byteLength) throw new Error('Invalid score snapshot metadata length.')
  const metadata = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer,8,length))) as SnapshotMetadata
  const addresses = worldTiles(), cells = SIZE * SIZE
  if (metadata.tileSize !== SIZE || Math.abs(metadata.cellMetres - CELL) > .001 || metadata.tiles?.length !== addresses.length || buffer.byteLength !== offset + addresses.length * cells * 2 || !['u16le','row-delta-byteplanes'].includes(metadata.encoding)) throw new Error('Score snapshot does not match the native 4 km grid.')
  const tiles: WorldNoise['tiles'] = [], lookup = new Uint16Array(36 * 18).fill(MISSING)
  const histogram = new Uint32Array(MISSING)
  let valid = 0, best = 0, bytes = 0
  for (let tile = 0; tile < addresses.length; tile++) {
    const { h,v } = addresses[tile]
    if (metadata.tiles[tile][0] !== h || metadata.tiles[tile][1] !== v) throw new Error('Invalid score snapshot tile order.')
    const begin = offset + tile * cells * 2, scores = new Uint16Array(cells)
    if (metadata.encoding === 'u16le') {
      for (let i = 0; i < cells; i++) scores[i] = view.getUint16(begin + i * 2,true)
    } else {
      const low = new Uint8Array(buffer,begin,cells), high = new Uint8Array(buffer,begin + cells,cells)
      let previous = 0
      for (let i = 0; i < cells; i++) {
        if (i % SIZE === 0) previous = 0
        previous = (previous + low[i] + (high[i] << 8)) & 65535
        scores[i] = previous
      }
    }
    for (const score of scores) if (score !== MISSING) { histogram[score]++; valid++; best = Math.max(best,score) }
    const levels = maxPyramid(scores,SIZE,SIZE)
    bytes += levels.reduce((sum,level) => sum + level.scores.byteLength,0)
    lookup[v * 36 + h] = tiles.length
    tiles.push({ h,v,levels })
    yield [tiles.length,addresses.length]
  }
  if (valid !== metadata.valid || best !== metadata.best) throw new Error('Score snapshot summary does not match its contents.')
  return { tiles,histogram,lookup,valid,best,bytes,milliseconds: performance.now() - started, snapshot: metadata }
}
