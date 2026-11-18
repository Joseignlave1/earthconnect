import { loadEarthScore, type EarthScore, type EarthScoreOptions } from '../earth-score/index.ts'
import { createSurfaceDevicePool } from '../shader-test/surface-device.ts'

type World = {
  loading: Promise<EarthScore>
  publish: (task: () => Promise<void>) => Promise<void>
}

export function createEarthWorkspaceResources() {
  const worlds = new WeakMap<GPUDevice, { world: World; users: number; release: () => void }>()
  const acquireDevice = createSurfaceDevicePool()

  return {
    acquireDevice,
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
