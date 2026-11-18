import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import '../index.css'
import { embeddedWorkspaceAssets } from './embeddedAssets.ts'
import { WorkspaceStartup } from './WorkspaceStartup.tsx'
import type { WorkspaceAssets } from './workspaceAssets.ts'
import './export.css'

declare const WORKSPACE_OFFLINE: boolean

async function openWorkspace(assets: WorkspaceAssets, onReady: () => void) {
  const { default: SharedSkyWorkspace } = await import('./SharedSkyWorkspace.tsx')
  return <SharedSkyWorkspace assets={assets} onReady={onReady} />
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <WorkspaceStartup open={openWorkspace} embedded={WORKSPACE_OFFLINE ? embeddedWorkspaceAssets : undefined} />
  </StrictMode>,
)
