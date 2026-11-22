import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import '../index.css'
import { warmComputedScorePipelines } from '../shader-test/computed-scores.ts'
import type { createSurfaceDevicePool } from '../shader-test/surface-device.ts'
import { createEarthWorkspaceResources } from './earthWorkspaceResources.ts'
import { WorkspaceStartup } from './WorkspaceStartup.tsx'
import { scoreManifestUrl, type WorkspaceAssets } from './workspaceAssets.ts'
import type { loadWorkspaceRelease } from './workspaceRelease.ts'
import { endStartupSpan, startStartupSpan } from './startupTiming.ts'

// Start loading the workspace + Monaco chunks immediately in parallel with the
// release cache check instead of waiting until assets publish.
const workspaceModule = import('./SharedSkyWorkspace.tsx')

function createOpener(devicePool?: ReturnType<typeof createSurfaceDevicePool>, releaseInitialDevice?: () => void) {
  return async function open(assets: WorkspaceAssets, onReady: () => void, signal: AbortSignal) {
    const resources = createEarthWorkspaceResources(devicePool, signal)
    // Prewarm Earth, Moon, and Mars terrain workers immediately so they decode
    // off the main thread in parallel with Monaco and Zstd chunk loading.
    startStartupSpan('earth-terrain')
    void resources.acquireTerrain('textures/earth-terrain.bin', 'rg32float', s => assets.url('textures/earth-terrain.bin', s), signal)
      .then(() => endStartupSpan('earth-terrain'), () => undefined)
    startStartupSpan('solar-terrain')
    void Promise.all([
      resources.acquireTerrain('textures/moon-terrain.bin', 'rg8unorm', s => assets.url('textures/moon-terrain.bin', s), signal),
      resources.acquireTerrain('textures/mars-terrain.bin', 'rg8unorm', s => assets.url('textures/mars-terrain.bin', s), signal),
    ]).then(() => endStartupSpan('solar-terrain'), () => undefined)
    // Start acquiring the WebGPU device and decoding the 36 world chunks
    // immediately in parallel with Monaco workbench initialization.
    startStartupSpan('gpu-device')
    const deviceLease = resources.acquireDevice()
    releaseInitialDevice?.()
    let worldLease: ReturnType<typeof resources.acquireWorld> | undefined
    let released = false
    const releasePrewarm = () => {
      if (released) return
      released = true
      worldLease?.release()
      worldLease = undefined
      deviceLease.release()
    }
    signal.addEventListener('abort', releasePrewarm, { once: true })
    void deviceLease.ready.then(({ device }) => {
      endStartupSpan('gpu-device')
      if (released || signal.aborted) return
      warmComputedScorePipelines(device)
      startStartupSpan('earth-score', '36 Zstd chunks + GPU upload')
      worldLease = resources.acquireWorld(device, {
        manifestUrl: scoreManifestUrl,
        readAsset: assets.readAsset,
      })
      void worldLease.loading.then(() => endStartupSpan('earth-score'), () => undefined)
    }, () => undefined)
    const { default: SharedSkyWorkspace } = await workspaceModule
    return <SharedSkyWorkspace assets={assets} resources={resources} onReady={onReady} />
  }
}

export function mount(
  loadInitialRelease: typeof loadWorkspaceRelease,
  devicePool?: ReturnType<typeof createSurfaceDevicePool>,
  releaseInitialDevice?: () => void,
) {
  const open = createOpener(devicePool, releaseInitialDevice)
  createRoot(document.getElementById('root')!).render(
    <StrictMode>
      <WorkspaceStartup open={open} loadInitialRelease={loadInitialRelease} />
    </StrictMode>,
  )
}
