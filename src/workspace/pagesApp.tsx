import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import '../index.css'
import { WorkspaceStartup } from './WorkspaceStartup.tsx'
import type { WorkspaceAssets } from './workspaceAssets.ts'
import type { loadWorkspaceRelease } from './workspaceRelease.ts'

async function open(assets: WorkspaceAssets, onReady: () => void) {
  const { default: SharedSkyWorkspace } = await import('./SharedSkyWorkspace.tsx')
  return <SharedSkyWorkspace assets={assets} onReady={onReady} />
}

export function mount(loadInitialRelease: typeof loadWorkspaceRelease) {
  createRoot(document.getElementById('root')!).render(
    <StrictMode>
      <WorkspaceStartup open={open} loadInitialRelease={loadInitialRelease} />
    </StrictMode>,
  )
}
