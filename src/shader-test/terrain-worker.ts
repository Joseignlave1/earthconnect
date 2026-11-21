import { elevationPyramid, loadTerrain, packRg32FloatLevels, packRgba8Levels, poleElevations, type TerrainWorkerRequest, type TerrainWorkerResponse } from './terrain.ts'

const send = globalThis.postMessage as (message: TerrainWorkerResponse, transfer?: Transferable[]) => void

globalThis.onmessage = async ({ data }: MessageEvent<TerrainWorkerRequest>) => {
  try {
    const terrain = await loadTerrain(data.url, new AbortController().signal)
    const poles = poleElevations(terrain)
    const levels = elevationPyramid(terrain)
    const packed = data.format === 'rg32float' ? packRg32FloatLevels(levels) : packRgba8Levels(levels)
    const transfer: Transferable[] = [terrain.pixels.buffer as ArrayBuffer, ...packed.map(level => level.data.buffer as ArrayBuffer)]
    send({ kind: 'ok', terrain, poles, levels: packed }, transfer)
  } catch (error) {
    send({ kind: 'error', message: error instanceof Error ? error.message : String(error) })
  }
}
