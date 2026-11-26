import { StrictMode, useCallback, useEffect, useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'
import '../index.css'
import { BridgeHero } from '../hero/BridgeHero.tsx'
import type { createSurfaceDevicePool } from '../shader-test/surface-device.ts'
import type { EarthWorkspaceResources } from './earthWorkspaceResources.ts'
import { readShaderProject, type ShaderProject } from './shaderProject.ts'
import { WorkspaceStartup } from './WorkspaceStartup.tsx'
import type { WorkspaceAssets } from './workspaceAssets.ts'
import { startWorkspacePrewarm, type WorkspacePrewarm } from './workspacePrewarm.ts'
import type { loadWorkspaceRelease } from './workspaceRelease.ts'

// Reuse the workspace module fetch started in pagesBootstrap.
const workspaceModule = import('./SharedSkyWorkspace.tsx')

type AppRoute = 'hero' | 'workspace'

function routeFromLocation(): AppRoute {
  const path = window.location.pathname.replace(/\/+$/, '') || '/'
  return path === '/find' || path.startsWith('/find/') ? 'workspace' : 'hero'
}

type SharedSkyWorkspaceComponent = typeof import('./SharedSkyWorkspace.tsx')['default']

// eslint-disable-next-line react-refresh/only-export-components
function EarthConnectShell({
  WorkspaceComponent,
  assets,
  resources,
  initialProject,
  onReady,
}: {
  WorkspaceComponent: SharedSkyWorkspaceComponent
  assets: WorkspaceAssets
  resources: EarthWorkspaceResources
  initialProject: ShaderProject
  onReady: () => void
}) {
  const [initialRoute] = useState<AppRoute>(routeFromLocation)
  const [route, setRoute] = useState<AppRoute>(initialRoute)
  const [heroMounted, setHeroMounted] = useState(initialRoute === 'hero')
  const [heroLeaving, setHeroLeaving] = useState(false)
  const [workspaceReady, setWorkspaceReady] = useState(false)
  const [heroReady, setHeroReady] = useState(initialRoute !== 'hero')
  const reportedReady = useRef(false)
  const leaveTimer = useRef<number | undefined>(undefined)

  useEffect(() => {
    if (reportedReady.current) return
    if (workspaceReady && heroReady) {
      reportedReady.current = true
      onReady()
    }
  }, [workspaceReady, heroReady, onReady])

  const leaveHero = useCallback(() => {
    window.clearTimeout(leaveTimer.current)
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    if (reduced) {
      setHeroLeaving(false)
      setHeroMounted(false)
      return
    }
    setHeroLeaving(true)
    leaveTimer.current = window.setTimeout(() => {
      setHeroLeaving(false)
      setHeroMounted(false)
    }, 340)
  }, [])

  const openWorkspace = useCallback(() => {
    if (route === 'workspace') return
    if (routeFromLocation() !== 'workspace') window.history.pushState(null, '', '/find')
    setRoute('workspace')
    leaveHero()
  }, [route, leaveHero])

  useEffect(() => {
    const onPopState = () => {
      const next = routeFromLocation()
      window.clearTimeout(leaveTimer.current)
      setRoute(next)
      if (next === 'hero') {
        setHeroLeaving(false)
        setHeroMounted(true)
      } else {
        leaveHero()
      }
    }
    window.addEventListener('popstate', onPopState)
    return () => {
      window.removeEventListener('popstate', onPopState)
      window.clearTimeout(leaveTimer.current)
    }
  }, [leaveHero])

  const handleWorkspaceReady = useCallback(() => setWorkspaceReady(true), [])
  const handleHeroReady = useCallback(() => setHeroReady(true), [])

  return (
    <>
      <div inert={route === 'hero'}>
        <WorkspaceComponent
          assets={assets}
          resources={resources}
          initialProject={initialProject}
          onReady={handleWorkspaceReady}
        />
      </div>
      {heroMounted && (
        <BridgeHero
          assets={assets}
          resources={resources}
          leaving={heroLeaving}
          onOpenWorkspace={openWorkspace}
          onReady={handleHeroReady}
        />
      )}
    </>
  )
}

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
    const [{ default: SharedSkyWorkspace }, initialProject] = await Promise.all([workspaceModule, readShaderProject()])
    signal.throwIfAborted()
    const handleReady = () => {
      prewarm.handoff()
      onReady()
    }
    return (
      <EarthConnectShell
        WorkspaceComponent={SharedSkyWorkspace}
        assets={assets}
        resources={prewarm.resources}
        initialProject={initialProject}
        onReady={handleReady}
      />
    )
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
