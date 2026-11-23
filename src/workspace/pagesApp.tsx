import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import '../index.css'
import type { createSurfaceDevicePool } from '../shader-test/surface-device.ts'
import { WorkspaceStartup } from './WorkspaceStartup.tsx'
import type { WorkspaceAssets } from './workspaceAssets.ts'
import { startWorkspacePrewarm, type WorkspacePrewarm } from './workspacePrewarm.ts'
import type { loadWorkspaceRelease } from './workspaceRelease.ts'

// Reuse the workspace module fetch started in pagesBootstrap.
const workspaceModule = import('./SharedSkyWorkspace.tsx')

function createOpener(
  devicePool?: ReturnType<typeof createSurfaceDevicePool>,
  releaseUnclaimedDevice?: () => void,
  getPrewarm?: () => Promise<WorkspacePrewarm> | undefined,
) {
  return async function open(assets: WorkspaceAssets, onReady: () => void, signal: AbortSignal) {
    const existing = await getPrewarm?.()
    signal.throwIfAborted()
    let prewarm: WorkspacePrewarm
    if (existing && existing.assets === assets) {
      prewarm = existing
    } else {
      existing?.dispose()
      prewarm = startWorkspacePrewarm(assets, devicePool)
      releaseUnclaimedDevice?.()
    }
    signal.addEventListener('abort', prewarm.dispose, { once: true })
    const { default: SharedSkyWorkspace } = await workspaceModule
    signal.throwIfAborted()
    const handleReady = () => {
      prewarm.handoff()
      onReady()
    }
    return <SharedSkyWorkspace assets={assets} resources={prewarm.resources} onReady={handleReady} />
  }
}

export function mount(
  loadInitialRelease: typeof loadWorkspaceRelease,
  devicePool?: ReturnType<typeof createSurfaceDevicePool>,
  releaseUnclaimedDevice?: () => void,
  getPrewarm?: () => Promise<WorkspacePrewarm> | undefined,
) {
  const open = createOpener(devicePool, releaseUnclaimedDevice, getPrewarm)
  createRoot(document.getElementById('root')!).render(
    <StrictMode>
      <WorkspaceStartup open={open} loadInitialRelease={loadInitialRelease} />
    </StrictMode>,
  )
}
