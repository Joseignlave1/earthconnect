import { warmComputedScorePipelines } from '../shader-test/computed-scores.ts'
import type { createSurfaceDevicePool } from '../shader-test/surface-device.ts'
import { createEarthWorkspaceResources, type EarthWorkspaceResources } from './earthWorkspaceResources.ts'
import { endStartupSpan, startStartupSpan } from './startupTiming.ts'
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
  if (!existingDeviceLease) {
    startStartupSpan('gpu-device')
    void deviceLease.ready.then(
      () => endStartupSpan('gpu-device'),
      () => undefined,
    )
  }

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

  // Prewarm Earth terrain immediately; defer Moon & Mars terrain until Earth's
  // Zstd score chunks finish so startup CPU cores and GPU queues stay uncontended.
  startStartupSpan('earth-terrain')
  void resources
    .acquireTerrain('textures/earth-terrain.bin', 'rg32float', s => assets.url('textures/earth-terrain.bin', s), signal)
    .then(
      () => endStartupSpan('earth-terrain'),
      () => undefined,
    )

  void deviceLease.ready.then(
    ({ device }) => {
      if (handedOff || signal.aborted) return
      warmComputedScorePipelines(device)
      startStartupSpan('earth-score', '36 Zstd chunks + GPU upload')
      worldLease = resources.acquireWorld(device, {
        manifestUrl: scoreManifestUrl,
        readAsset: assets.readAsset,
      })
      void worldLease.loading
        .then(async () => {
          endStartupSpan('earth-score')
          if (signal.aborted) return
          startStartupSpan('solar-terrain')
          await resources.acquireTerrain(
            'textures/moon-terrain.bin',
            'rg8unorm',
            s => assets.url('textures/moon-terrain.bin', s),
            signal,
          )
          if (signal.aborted) return
          await resources.acquireTerrain(
            'textures/mars-terrain.bin',
            'rg8unorm',
            s => assets.url('textures/mars-terrain.bin', s),
            signal,
          )
          endStartupSpan('solar-terrain')
        })
        .catch(() => undefined)
    },
    () => undefined,
  )

  return { assets, resources, handoff, dispose }
}
