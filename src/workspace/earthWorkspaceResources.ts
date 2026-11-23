import { loadEarthScore, type EarthScore, type EarthScoreOptions } from '../earth-score/index.ts'
import { samplePoint } from '../earthIndexApi.ts'
import { raceAbort } from '../shader-test/gpu.ts'
import { createSurfaceDevicePool } from '../shader-test/surface-device.ts'
import type { TerrainPackFormat } from '../shader-test/terrain.ts'
import { loadPreparedTerrain, type PreparedTerrain } from '../shader-test/terrain-loader.ts'
import { canonicalPoint, coordinatesValid, createPlanetaryPointCache, pointKey, type PlanetaryCoordinates } from './planetaryPointCache.ts'

type World = {
  loading: Promise<EarthScore>
  publish: (task: () => Promise<void>) => Promise<void>
}

export type TerrainLoader = <F extends TerrainPackFormat>(
  path: string,
  format: F,
  getUrl: (signal: AbortSignal) => Promise<string>,
  signal: AbortSignal,
) => Promise<PreparedTerrain<F>>

export function createEarthWorkspaceResources(acquireDevice = createSurfaceDevicePool(), lifetime?: AbortSignal) {
  const worlds = new WeakMap<GPUDevice, { world: World; users: number; release: () => void }>()
  const terrains = new Map<string, Promise<PreparedTerrain<TerrainPackFormat>>>()
  const pointLifetime = lifetime ?? new AbortController().signal
  const readPoint = createPlanetaryPointCache(samplePoint, pointLifetime)

  function prefetchPoints(points: readonly PlanetaryCoordinates[]) {
    const unique = new Map<string, PlanetaryCoordinates>()
    for (const raw of points) {
      if (!coordinatesValid(raw.latitude, raw.longitude)) continue
      const point = canonicalPoint(raw)
      const key = pointKey(point)
      if (!unique.has(key)) unique.set(key, point)
    }
    if (!unique.size) return
    const items = [...unique.values()]
    let next = 0
    for (let worker = 0; worker < Math.min(4, items.length); worker++) {
      void (async () => {
        while (next < items.length && !pointLifetime.aborted) {
          const point = items[next++]
          await readPoint(point, pointLifetime).catch(() => {})
        }
      })()
    }
  }

  const acquireTerrain: TerrainLoader = <F extends TerrainPackFormat>(
    path: string,
    format: F,
    getUrl: (signal: AbortSignal) => Promise<string>,
    signal: AbortSignal,
  ): Promise<PreparedTerrain<F>> => {
    signal.throwIfAborted()
    const key = `${format}:${path}`
    let pending = terrains.get(key) as Promise<PreparedTerrain<F>> | undefined
    if (!pending) {
      const controller = new AbortController()
      pending = getUrl(controller.signal)
        .then(url => loadPreparedTerrain(url, format, controller.signal))
      terrains.set(key, pending)
      void pending.catch(() => {
        if (terrains.get(key) === pending) terrains.delete(key)
      })
    }
    return raceAbort(pending, signal)
  }

  return {
    acquireDevice,
    acquireTerrain,
    readPoint,
    prefetchPoints,
    acquireWorld(device: GPUDevice, options: Omit<EarthScoreOptions, 'signal'>) {
      let entry = worlds.get(device)
      if (!entry) {
        let publication = Promise.resolve()
        const controller = new AbortController()
        const world: World = {
          loading: loadEarthScore(device, { ...options, signal: controller.signal }),
          publish(task) {
            // Dispatches share output buffers until the destination has copied them.
            const next = publication.then(task)
            publication = next.catch(() => {})
            return next
          },
        }
        const created = {
          world,
          users: 0,
          release() {
            if (worlds.get(device) === created) worlds.delete(device)
            controller.abort()
          },
        }
        entry = created
        worlds.set(device, created)
        void world.loading.then(earth => {
          if (controller.signal.aborted) earth.destroy()
          else controller.signal.addEventListener('abort', () => earth.destroy(), { once: true })
        }, () => {
          if (worlds.get(device) === created) worlds.delete(device)
        })
      }
      const active = entry
      active.users++
      let released = false
      return {
        ...active.world,
        device,
        release() {
          if (released) return
          released = true
          if (!--active.users) active.release()
        },
      }
    },
  }
}

export type EarthWorkspaceResources = ReturnType<typeof createEarthWorkspaceResources>
