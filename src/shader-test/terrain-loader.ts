import {
  elevationPyramid, loadTerrain, packRg32FloatLevels, packRg8Levels, packRgba8Levels, poleElevations,
  type PackedTerrainLevel, type Terrain, type TerrainPackFormat, type TerrainWorkerRequest, type TerrainWorkerResponse,
} from './terrain.ts'
import { createTerrainWorker } from './terrain-worker-factory.ts'

export type PreparedTerrain<F extends TerrainPackFormat> = {
  terrain: Terrain
  poles: [number, number]
  levels: PackedTerrainLevel<F extends 'rg32float' ? Float32Array<ArrayBuffer> : Uint8Array<ArrayBuffer>>[] | null
}

async function prepareTerrainOnMainThread<F extends TerrainPackFormat>(url: string, format: F, signal: AbortSignal): Promise<PreparedTerrain<F>> {
  const { decompressZstd } = await import('../earth-score/decode.ts')
  const terrain = await loadTerrain(url, signal, decompressZstd)
  signal.throwIfAborted()
  const poles = poleElevations(terrain)
  const levels = elevationPyramid(terrain, format !== 'rg32float')
  const packed = (format === 'rg32float' ? packRg32FloatLevels(levels) : format === 'rg8unorm' ? packRg8Levels(levels) : packRgba8Levels(levels)) as PreparedTerrain<F>['levels']
  return { terrain, poles, levels: packed }
}

/** Decode gzipped terrain and build its mipmap pyramid off the main thread. */
export function loadPreparedTerrain<F extends TerrainPackFormat>(url: string, format: F, signal: AbortSignal): Promise<PreparedTerrain<F>> {
  signal.throwIfAborted()
  const resolved = typeof globalThis.location?.href === 'string' ? new URL(url, globalThis.location.href).href : url
  return new Promise<PreparedTerrain<F>>((resolve, reject) => {
    let worker: Worker
    try {
      worker = createTerrainWorker()
    } catch {
      void prepareTerrainOnMainThread(resolved, format, signal).then(resolve, reject)
      return
    }
    let settled = false
    const cleanup = () => {
      signal.removeEventListener('abort', onAbort)
      worker.terminate()
    }
    const onAbort = () => {
      if (settled) return
      settled = true
      cleanup()
      reject(signal.reason)
    }
    const fallback = () => {
      if (settled) return
      settled = true
      cleanup()
      void prepareTerrainOnMainThread(resolved, format, signal).then(resolve, reject)
    }
    signal.addEventListener('abort', onAbort, { once: true })
    worker.onmessage = ({ data }: MessageEvent<TerrainWorkerResponse>) => {
      if (settled) return
      settled = true
      cleanup()
      if (data.kind === 'error') reject(new Error(data.message))
      else resolve({ terrain: data.terrain, poles: data.poles, levels: data.levels as PreparedTerrain<F>['levels'] })
    }
    worker.onerror = (event) => { event.preventDefault(); fallback() }
    worker.onmessageerror = fallback
    try {
      worker.postMessage({ url: resolved, format } satisfies TerrainWorkerRequest)
    } catch {
      fallback()
    }
  })
}
