import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import '../index.css'
import { embeddedWorkspaceAssets } from './embeddedAssets.ts'
import { WorkspaceStartup } from './WorkspaceStartup.tsx'
import { readShaderProject } from './shaderProject.ts'
import type { WorkspaceAssets } from './workspaceAssets.ts'
import './export.css'

declare const WORKSPACE_OFFLINE: boolean

async function openWorkspace(assets: WorkspaceAssets, onReady: () => void) {
  const [{ default: SharedSkyWorkspace }, initialProject] = await Promise.all([
    import('./SharedSkyWorkspace.tsx'),
    import.meta.env.VITE_WORKSPACE_STANDALONE ? Promise.resolve(undefined) : readShaderProject(),
  ])
  return <SharedSkyWorkspace assets={assets} initialProject={initialProject} onReady={onReady} />
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <WorkspaceStartup open={openWorkspace} embedded={WORKSPACE_OFFLINE ? embeddedWorkspaceAssets : undefined} />
  </StrictMode>,
)
