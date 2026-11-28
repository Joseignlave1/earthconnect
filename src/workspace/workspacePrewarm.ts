import { warmComputedScorePipelines } from '../shader-test/computed-scores.ts'
import type { createSurfaceDevicePool } from '../shader-test/surface-device.ts'
import { createBootGraph } from './bootGraph.ts'
import { createEarthWorkspaceResources, type EarthWorkspaceResources } from './earthWorkspaceResources.ts'
import { scoreManifestUrl, type WorkspaceAssets } from './workspaceAssets.ts'

export type WorkspacePrewarm = {
  assets: WorkspaceAssets
  resources: EarthWorkspaceResources
  /** Release temporary bootstrap GPU/world leases once the mounted workspace holds its own. */
  handoff: () => void
  /** Abort in-flight prewarm work and release all temporary leases (e.g. startup retry/unmount). */
  dispose: () => void
}

export function startWorkspacePrewarm(
  assets: WorkspaceAssets,
  devicePool?: ReturnType<typeof createSurfaceDevicePool>,
  existingDeviceLease?: ReturnType<ReturnType<typeof createSurfaceDevicePool>>,
): WorkspacePrewarm {
  const ownerController = new AbortController()
  const prewarmController = new AbortController()
  const { signal } = prewarmController
  const resources = createEarthWorkspaceResources(devicePool, ownerController.signal)
  const deviceLease = existingDeviceLease ?? resources.acquireDevice()

  let worldLease: ReturnType<typeof resources.acquireWorld> | undefined
  let handedOff = false

  const handoff = () => {
    if (handedOff) return
    handedOff = true
    worldLease?.release()
    worldLease = undefined
    deviceLease.release()
  }

  const dispose = () => {
    prewarmController.abort()
    ownerController.abort()
    handoff()
  }

  // Declarative bootstrap prewarm DAG:
  // - Earth terrain starts immediately in parallel with GPU device acquisition.
  // - Earth score chunks wait for the GPU device, then upload 36 Zstd chunks.
  // - Moon & Mars terrain wait for Earth score chunks so startup CPU/GPU queues stay uncontended.
  const boot = createBootGraph(signal, 'prepare')

  boot.task('gpu-device', {
    phase: 'bootstrap',
    recordTiming: !existingDeviceLease,
    run: () => deviceLease.ready,
  })

  boot.task('earth-terrain', {
    run: ({ signal: s }) =>
      resources.acquireTerrain(
        'textures/earth-terrain.bin',
        'rg32float',
        sig => assets.url('textures/earth-terrain.bin', sig),
        s,
      ),
  })

  boot.task('earth-score', {
    after: ['gpu-device'],
    detail: '36 Zstd chunks + GPU upload',
    run: async ({ get }) => {
      const { device } = get<{ device: GPUDevice }>('gpu-device')
      if (handedOff) return
      warmComputedScorePipelines(device)
      worldLease = resources.acquireWorld(device, {
        manifestUrl: scoreManifestUrl,
        readAsset: assets.readAsset,
      })
      await worldLease.loading
    },
  })

  boot.task('solar-terrain', {
    after: ['earth-score'],
    run: async ({ signal: s }) => {
      await resources.acquireTerrain(
        'textures/moon-terrain.bin',
        'rg8unorm',
        sig => assets.url('textures/moon-terrain.bin', sig),
        s,
      )
      s.throwIfAborted()
      await resources.acquireTerrain(
        'textures/mars-terrain.bin',
        'rg8unorm',
        sig => assets.url('textures/mars-terrain.bin', sig),
        s,
      )
    },
  })

  return { assets, resources, handoff, dispose }
}
