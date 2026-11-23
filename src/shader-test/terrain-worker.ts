import { decompressZstd } from '../earth-score/decode.ts'
import { elevationPyramid, loadTerrain, packRg32FloatLevels, packRg8PyramidDirect, poleElevations, type TerrainWorkerRequest, type TerrainWorkerResponse } from './terrain.ts'

const send = globalThis.postMessage as (message: TerrainWorkerResponse, transfer?: Transferable[]) => void

globalThis.onmessage = async ({ data }: MessageEvent<TerrainWorkerRequest>) => {
  try {
    const terrain = await loadTerrain(data.url, new AbortController().signal, decompressZstd)
    const poles = poleElevations(terrain)
    const packed = data.format === 'rg8unorm'
      ? packRg8PyramidDirect(terrain)
      : packRg32FloatLevels(elevationPyramid(terrain, false))
    const transfer = [...new Set<ArrayBuffer>([terrain.pixels.buffer as ArrayBuffer, ...packed.map(level => level.data.buffer as ArrayBuffer)])]
    send({ kind: 'ok', terrain, poles, levels: packed }, transfer)
  } catch (error) {
    send({ kind: 'error', message: error instanceof Error ? error.message : String(error) })
  }
}
