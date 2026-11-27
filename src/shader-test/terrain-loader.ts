import {
  prepareTerrainData,
  type PackedTerrainLevel,
  type Terrain,
  type TerrainPackFormat,
  type TerrainWorkerRequest,
  type TerrainWorkerResponse,
} from './terrain.ts'
import { createTerrainWorker } from './terrain-worker-factory.ts'

export type PreparedTerrain<F extends TerrainPackFormat> = {
  terrain: Terrain
  poles: [number, number]
  levels: PackedTerrainLevel<F extends 'rg32float' ? Float32Array<ArrayBuffer> : Uint8Array<ArrayBuffer>>[] | null
}

async function prepareTerrainOnMainThread<F extends TerrainPackFormat>(
  url: string,
  format: F,
  signal: AbortSignal,
  skipMipmaps: boolean,
): Promise<PreparedTerrain<F>> {
  const { decompressTerrainEct2Wasm, decompressZstd } = await import('../earth-score/decode.ts')
  return prepareTerrainData(url, format, signal, decompressZstd, {
    skipMipmaps,
    decodeEct2Wasm: decompressTerrainEct2Wasm,
  })
}

/** Decode terrain off the main thread; mipmap pyramids are built on the GPU when supported. */
export function loadPreparedTerrain<F extends TerrainPackFormat>(
  url: string,
  format: F,
  signal: AbortSignal,
  skipMipmaps = true,
): Promise<PreparedTerrain<F>> {
  signal.throwIfAborted()
  const resolved = typeof globalThis.location?.href === 'string' ? new URL(url, globalThis.location.href).href : url
  return new Promise<PreparedTerrain<F>>((resolve, reject) => {
    let worker: Worker
    try {
      worker = createTerrainWorker()
    } catch {
      void prepareTerrainOnMainThread(resolved, format, signal, skipMipmaps).then(resolve, reject)
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
      void prepareTerrainOnMainThread(resolved, format, signal, skipMipmaps).then(resolve, reject)
    }
    signal.addEventListener('abort', onAbort, { once: true })
    worker.onmessage = ({ data }: MessageEvent<TerrainWorkerResponse>) => {
      if (settled) return
      settled = true
      cleanup()
      if (data.kind === 'error') reject(new Error(data.message))
      else resolve({ terrain: data.terrain, poles: data.poles, levels: data.levels as PreparedTerrain<F>['levels'] })
    }
    worker.onerror = event => {
      event.preventDefault()
      fallback()
    }
    worker.onmessageerror = fallback
    try {
      worker.postMessage({ url: resolved, format, skipMipmaps } satisfies TerrainWorkerRequest)
    } catch {
      fallback()
    }
  })
}
