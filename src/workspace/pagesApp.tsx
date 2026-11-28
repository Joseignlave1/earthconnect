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

// Lazily loaded when entering /find so the landing page (/) never pays for Monaco/WGSL analyzer
// or a hidden duplicate WebGPU SurfaceRenderer upfront.
let workspaceModulePromise: Promise<typeof import('./SharedSkyWorkspace.tsx')> | null = null
let projectPromise: Promise<ShaderProject> | null = null
function loadWorkspaceModule() {
  if (!workspaceModulePromise) {
    const pending = import('./SharedSkyWorkspace.tsx').catch(error => {
      if (workspaceModulePromise === pending) workspaceModulePromise = null
      throw error
    })
    workspaceModulePromise = pending
  }
  return workspaceModulePromise
}
function loadSavedProject() {
  if (!projectPromise) {
    const pending = readShaderProject().catch(error => {
      if (projectPromise === pending) projectPromise = null
      throw error
    })
    projectPromise = pending
  }
  return projectPromise
}
function prefetchWorkspaceAssets() {
  void loadWorkspaceModule().catch(() => undefined)
  void loadSavedProject().catch(() => undefined)
}

type AppRoute = 'hero' | 'workspace'

function routeFromLocation(): AppRoute {
  const path = window.location.pathname.replace(/\/+$/, '') || '/'
  return path === '/find' || path.startsWith('/find/') ? 'workspace' : 'hero'
}

type SharedSkyWorkspaceComponent = typeof import('./SharedSkyWorkspace.tsx')['default']

// eslint-disable-next-line react-refresh/only-export-components
function EarthConnectShell({
  initialWorkspaceComponent,
  assets,
  resources,
  initialProject,
  onReadyChange,
}: {
  initialWorkspaceComponent: SharedSkyWorkspaceComponent | null
  assets: WorkspaceAssets
  resources: EarthWorkspaceResources
  initialProject: ShaderProject | null
  onReadyChange: (ready: boolean) => void
}) {
  const [initialRoute] = useState<AppRoute>(routeFromLocation)
  const [route, setRoute] = useState<AppRoute>(initialRoute)
  const [workspaceComponent, setWorkspaceComponent] = useState<SharedSkyWorkspaceComponent | null>(
    () => initialWorkspaceComponent,
  )
  const [project, setProject] = useState<ShaderProject | null>(() => initialProject)
  const [heroMounted, setHeroMounted] = useState(initialRoute === 'hero')
  const [workspaceReady, setWorkspaceReady] = useState(false)
  const [workspaceLoadError, setWorkspaceLoadError] = useState<string | null>(null)
  const [workspaceLoadAttempt, setWorkspaceLoadAttempt] = useState(0)
  const [heroReady, setHeroReady] = useState(false)
  const lastReportedReady = useRef<boolean | null>(null)
  const routeRef = useRef(route)
  const heroMountedRef = useRef(heroMounted)
  useEffect(() => {
    routeRef.current = route
    heroMountedRef.current = heroMounted
  }, [route, heroMounted])

  // Keep the loading screen up whenever the active view hasn't finished rendering its first complete frame.
  const activeRouteReady = route === 'hero' ? heroReady : workspaceReady || Boolean(workspaceLoadError)
  useEffect(() => {
    if (lastReportedReady.current === null && !activeRouteReady) return
    if (lastReportedReady.current === activeRouteReady) return
    lastReportedReady.current = activeRouteReady
    onReadyChange(activeRouteReady)
  }, [activeRouteReady, onReadyChange])

  // Load workspace bundle + saved WGSL project on demand when navigating into /find.
  useEffect(() => {
    if (route !== 'workspace' || (workspaceComponent && project)) return
    let cancelled = false
    void Promise.all([
      workspaceComponent ? Promise.resolve({ default: workspaceComponent }) : loadWorkspaceModule(),
      project ? Promise.resolve(project) : loadSavedProject(),
    ])
      .then(([mod, loadedProject]) => {
        if (cancelled) return
        setWorkspaceLoadError(null)
        setWorkspaceComponent(() => mod.default)
        setProject(loadedProject)
      })
      .catch(error => {
        if (cancelled) return
        setWorkspaceLoadError(error instanceof Error ? error.message : 'Could not load the workspace.')
      })
    return () => {
      cancelled = true
    }
  }, [route, workspaceComponent, project, workspaceLoadAttempt])

  // Silently warm the workspace bundle after the hero has revealed and settled,
  // so clicking Explore doesn't wait to download & parse Monaco from scratch.
  useEffect(() => {
    if (!heroReady || route !== 'hero' || workspaceComponent) return
    let idleId: number | undefined
    const timer = window.setTimeout(() => {
      const ric = (window as { requestIdleCallback?: (cb: () => void, opts?: { timeout: number }) => number })
        .requestIdleCallback
      if (typeof ric === 'function') {
        idleId = ric(() => prefetchWorkspaceAssets(), { timeout: 2000 })
      } else {
        prefetchWorkspaceAssets()
      }
    }, 1000)
    return () => {
      window.clearTimeout(timer)
      if (idleId !== undefined) {
        ;(window as { cancelIdleCallback?: (id: number) => void }).cancelIdleCallback?.(idleId)
      }
    }
  }, [heroReady, route, workspaceComponent])

  const openWorkspace = useCallback(() => {
    if (route === 'workspace' && !workspaceLoadError) return
    if (routeFromLocation() !== 'workspace') window.history.pushState(null, '', '/find')
    setWorkspaceLoadError(null)
    setWorkspaceReady(false)
    setRoute('workspace')
    if (workspaceLoadError) setWorkspaceLoadAttempt(n => n + 1)
  }, [route, workspaceLoadError])

  useEffect(() => {
    const onPopState = () => {
      const next = routeFromLocation()
      setRoute(next)
      if (next === 'hero') {
        setWorkspaceReady(false)
        if (!heroMountedRef.current) {
          // Cover synchronously in the same React render batch as remounting BridgeHero
          // so zero unloaded frames leak before effects run.
          lastReportedReady.current = false
          onReadyChange(false)
          setHeroReady(false)
          setHeroMounted(true)
        }
      } else {
        setWorkspaceReady(false)
        if (!heroMountedRef.current) {
          lastReportedReady.current = false
          onReadyChange(false)
        }
      }
    }
    window.addEventListener('popstate', onPopState)
    return () => window.removeEventListener('popstate', onPopState)
  }, [onReadyChange])

  const handleWorkspaceReady = useCallback(() => {
    setWorkspaceReady(true)
    if (routeRef.current === 'workspace' && heroMountedRef.current) {
      setHeroMounted(false)
      setHeroReady(false)
    }
  }, [])
  const handleHeroReady = useCallback(() => setHeroReady(true), [])

  const openingWorkspace = route === 'workspace' && !workspaceReady && !workspaceLoadError
  const Workspace = workspaceComponent
  return (
    <>
      {route === 'workspace' && Workspace && project && (
        <div>
          <Workspace
            assets={assets}
            resources={resources}
            initialProject={project}
            onProjectChange={setProject}
            onReady={handleWorkspaceReady}
          />
        </div>
      )}
      {route === 'workspace' && workspaceLoadError && (
        <div
          role="alert"
          style={{
            position: 'fixed',
            inset: 0,
            zIndex: 100,
            display: 'grid',
            placeItems: 'center',
            background: '#05070c',
            color: '#e6edf5',
            padding: '24px',
          }}
        >
          <div style={{ maxWidth: 420, textAlign: 'center', display: 'grid', gap: 12 }}>
            <p style={{ margin: 0 }}>Could not open the workspace: {workspaceLoadError}</p>
            <div style={{ display: 'flex', gap: 10, justifyContent: 'center' }}>
              <button
                type="button"
                onClick={() => {
                  setWorkspaceLoadError(null)
                  setWorkspaceLoadAttempt(n => n + 1)
                }}
              >
                Retry
              </button>
              <button
                type="button"
                onClick={() => {
                  setWorkspaceLoadError(null)
                  window.history.pushState(null, '', '/')
                  setRoute('hero')
                  setHeroMounted(true)
                }}
              >
                Back
              </button>
            </div>
          </div>
        </div>
      )}
      {heroMounted && (
        <BridgeHero
          assets={assets}
          resources={resources}
          leaving={false}
          opening={openingWorkspace}
          onPrefetchWorkspace={prefetchWorkspaceAssets}
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
  return async function open(assets: WorkspaceAssets, onReady: (ready?: boolean) => void, signal: AbortSignal) {
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

    let initialWorkspaceComponent: SharedSkyWorkspaceComponent | null = null
    let initialProject: ShaderProject | null = null
    if (routeFromLocation() === 'workspace') {
      const [{ default: SharedSkyWorkspace }, loadedProject] = await Promise.all([
        loadWorkspaceModule(),
        readShaderProject(),
      ])
      signal.throwIfAborted()
      initialWorkspaceComponent = SharedSkyWorkspace
      initialProject = loadedProject
    }

    const handleReadyChange = (ready: boolean) => {
      if (ready) prewarm.handoff()
      onReady(ready)
    }
    return (
      <EarthConnectShell
        initialWorkspaceComponent={initialWorkspaceComponent}
        assets={assets}
        resources={prewarm.resources}
        initialProject={initialProject}
        onReadyChange={handleReadyChange}
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
