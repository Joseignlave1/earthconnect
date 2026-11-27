import { decompressZstd } from '../earth-score/decode.ts'
import { prepareTerrainData, type TerrainWorkerRequest, type TerrainWorkerResponse } from './terrain.ts'

const send = globalThis.postMessage as (message: TerrainWorkerResponse, transfer?: Transferable[]) => void

globalThis.onmessage = async ({ data }: MessageEvent<TerrainWorkerRequest>) => {
  try {
    const { terrain, poles, levels } = await prepareTerrainData(
      data.url,
      data.format,
      new AbortController().signal,
      decompressZstd,
    )
    const transfer = [
      ...new Set<ArrayBuffer>([
        terrain.pixels.buffer as ArrayBuffer,
        ...levels.map(level => level.data.buffer as ArrayBuffer),
      ]),
    ]
    send({ kind: 'ok', terrain, poles, levels }, transfer)
  } catch (error) {
    send({ kind: 'error', message: error instanceof Error ? error.message : String(error) })
  }
}
