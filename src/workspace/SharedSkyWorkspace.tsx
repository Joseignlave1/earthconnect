import type { WorkspaceAssets } from './workspaceAssets.ts'
import {
  Component,
  lazy,
  Suspense,
  useCallback,
  useEffect,
  useId,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type MouseEvent as ReactMouseEvent,
  type ReactNode,
} from 'react'
import { createPortal } from 'react-dom'
import type { BodyId } from '../explorerData'
import { DEFAULT_SCORE_SHADER, SCORE_SHADER_PRELUDE } from '../earth-score/custom-shader.ts'
import earthCamera from '../assets/earth-camera.webp'
import marsCamera from '../assets/mars-camera.webp'
import moonCamera from '../assets/moon-camera.webp'
import type { WgslDiagnostic } from './WgslEditor'
import { helloWorldExample, helloWorldInitialView } from './helloWorldExample.ts'
import { createEarthWorkspaceResources, type EarthWorkspaceResources } from './earthWorkspaceResources'
import { useEarthWorkspace } from './useEarthWorkspace'
import {
  assembleShaderProject,
  defaultShaderProject,
  defaultShaderSource,
  findShaderLocation,
  readShaderProject,
  resolveShaderFiles,
  saveShaderProject,
  selectShaderModule,
  shaderModuleId,
  type ShaderProject,
} from './shaderProject'
import { formatPointCall, type PlanetaryCoordinates } from './planetaryPointCache.ts'
import { initialPreviewFileId } from './shaderLayout.ts'
import { endStartupSpan, startStartupSpan } from './startupTiming.ts'
import './SharedSkyWorkspace.css'

const wgslEditorImport = import.meta.env.VITE_WORKSPACE_STANDALONE ? null : import('./WgslEditor')
const WgslEditor = wgslEditorImport ? lazy(() => wgslEditorImport) : null

class ShaderEditorBoundary extends Component<{ children: ReactNode; onError: () => void }> {
  state = { failed: false }

  static getDerivedStateFromError() {
    return { failed: true }
  }

  componentDidCatch(error: Error) {
    console.error('Could not load the shader editor', error)
    this.props.onError()
  }

  render() {
    if (this.state.failed) {
      return <p role="status">The shader editor could not load. Reload the page to try again.</p>
    }
    return this.props.children
  }
}

type World = 'earth' | BodyId
type Projection = 'globe' | 'plane'
type Point = { latitude: number; longitude: number }

const worlds = ['earth', 'moon', 'mars'] as const
const worldName = { earth: 'Earth', moon: 'Moon', mars: 'Mars' }
const worldImage = { earth: earthCamera, moon: moonCamera, mars: marsCamera }
const formatPoint = ({ latitude, longitude }: Point) =>
  `${Math.abs(latitude).toFixed(3)}° ${latitude < 0 ? 'S' : 'N'}, ${Math.abs(longitude).toFixed(3)}° ${longitude < 0 ? 'W' : 'E'}`

const vimPreference = 'earth-connect-wgsl-vim'
function readVim() {
  try {
    return localStorage.getItem(vimPreference) === 'true'
  } catch {
    return false
  }
}

function PointPicker({
  body,
  point,
  onInsert,
  onCopy,
  actions,
  onActions,
}: {
  body: BodyId
  point?: Point
  onInsert?: (text: string) => void
  onCopy: (reference: PlanetaryCoordinates, signal: AbortSignal) => Promise<{ code: string; partial: boolean }>
  actions: { x: number; y: number; trigger: HTMLElement } | null
  onActions: (actions: { x: number; y: number; trigger: HTMLElement } | null) => void
}) {
  const menu = useRef<HTMLDivElement>(null)
  const copying = useRef<AbortController>(null)
  const [copyStatus, setCopyStatus] = useState('')
  const [copyFallback, setCopyFallback] = useState('')
  const fallback = useRef<HTMLTextAreaElement>(null)
  const menuId = useId()
  useEffect(() => () => copying.current?.abort(), [])
  useEffect(() => {
    if (copyFallback) fallback.current?.focus()
  }, [copyFallback])

  useEffect(() => {
    const popup = menu.current
    if (!popup || !actions) return
    popup.showPopover()
    const { width, height } = popup.getBoundingClientRect()
    popup.style.left = `${Math.max(4, Math.min(actions.x, window.innerWidth - width - 4))}px`
    popup.style.top = `${Math.max(4, Math.min(actions.y, window.innerHeight - height - 4))}px`
    popup.querySelector('button')?.focus()
  }, [actions])

  async function copy(kind: 'coordinates' | 'expression' | 'sample') {
    if (!point) return
    const reference = { body, ...point }
    copying.current?.abort()
    const controller = new AbortController()
    copying.current = controller
    setCopyFallback('')
    let code = kind === 'expression' ? formatPointCall(reference) : `${point.latitude}, ${point.longitude}`
    let partial = false
    if (kind === 'sample') {
      setCopyStatus('Sampling point…')
      try {
        ;({ code, partial } = await onCopy(reference, controller.signal))
      } catch (error) {
        if (!controller.signal.aborted) setCopyStatus(error instanceof Error ? error.message : String(error))
        return
      }
    }
    if (controller.signal.aborted) return
    try {
      await navigator.clipboard.writeText(code)
      if (!controller.signal.aborted)
        setCopyStatus(partial ? 'Copied available values. Coverage is partial.' : 'Copied.')
    } catch {
      if (controller.signal.aborted) return
      setCopyFallback(code)
      setCopyStatus(
        partial
          ? 'Copy the available values below. Coverage is partial.'
          : 'Clipboard unavailable. Copy the text below.',
      )
    }
  }

  function closeMenu(restoreFocus = true) {
    menu.current?.hidePopover()
    if (restoreFocus) actions?.trigger.focus()
    onActions(null)
  }

  function openAtButton(event: ReactMouseEvent<HTMLButtonElement>) {
    const { left, top } = event.currentTarget.getBoundingClientRect()
    event.preventDefault()
    onActions({ x: left, y: top, trigger: event.currentTarget })
  }

  if (!point) return <span>Pick a point on {worldName[body]}.</span>
  const text = `${worldName[body]} · ${formatPoint(point)}`

  return (
    <div className="sky-point-picker">
      <button
        type="button"
        title="Copy coordinates"
        aria-label={`Copy coordinates for ${text}`}
        onClick={() => void copy('coordinates')}
        onContextMenu={openAtButton}
      >
        {text}
      </button>
      <button
        type="button"
        title="Point actions"
        aria-label={`${worldName[body]} point actions`}
        aria-haspopup="menu"
        aria-expanded={actions !== null}
        aria-controls={menuId}
        onClick={openAtButton}
      >
        ⋯
      </button>
      <div
        id={menuId}
        ref={menu}
        popover="auto"
        className="sky-point-menu"
        role="menu"
        aria-label="Point actions"
        onToggle={event => {
          if ((event.nativeEvent as ToggleEvent).newState === 'closed') onActions(null)
        }}
        onKeyDown={event => {
          if (event.key === 'Escape') {
            event.preventDefault()
            closeMenu()
            return
          }
          if (event.key === 'Tab') {
            closeMenu()
            return
          }
          const keys = ['ArrowDown', 'ArrowUp', 'Home', 'End']
          if (!keys.includes(event.key)) return
          event.preventDefault()
          const items = Array.from(event.currentTarget.querySelectorAll('button'))
          const current = items.findIndex(item => item === document.activeElement)
          const next =
            event.key === 'Home'
              ? 0
              : event.key === 'End'
                ? items.length - 1
                : (current + (event.key === 'ArrowDown' ? 1 : -1) + items.length) % items.length
          items[next]?.focus()
        }}
      >
        <button
          type="button"
          role="menuitem"
          tabIndex={-1}
          onClick={() => {
            closeMenu()
            void copy('expression')
          }}
        >
          Copy as WGSL expression
        </button>
        {onInsert && (
          <button
            type="button"
            role="menuitem"
            tabIndex={-1}
            onClick={() => {
              closeMenu(false)
              onInsert(formatPointCall({ body, ...point }))
            }}
          >
            Insert WGSL expression
          </button>
        )}
        <button
          type="button"
          role="menuitem"
          tabIndex={-1}
          onClick={() => {
            closeMenu()
            void copy('sample')
          }}
        >
          Copy sampled WGSL values
        </button>
      </div>
      {copyStatus && (
        <span className="sky-point-feedback" role="status">
          {copyStatus}
        </span>
      )}
      {copyFallback && (
        <div className="sky-point-fallback">
          <textarea
            ref={fallback}
            readOnly
            aria-label="Text to copy"
            value={copyFallback}
            onFocus={event => event.target.select()}
          />
          <button
            type="button"
            onClick={() => {
              setCopyFallback('')
              setCopyStatus('')
            }}
          >
            Dismiss
          </button>
        </div>
      )}
    </div>
  )
}

type SceneRun = { id: number; source: string }
type SceneShader = ReturnType<typeof assembleShaderProject>
const introShader = assembleShaderProject([helloWorldExample])

function SkyScene({
  assets,
  resources,
  fileId,
  assembled,
  run,
  onDiagnostics,
  onInsert,
  footer,
  onReady,
  initialView,
}: {
  onReady: () => void
  initialView?: { latitude: number; longitude: number }
  footer?: boolean
  assets: WorkspaceAssets
  resources?: EarthWorkspaceResources
  fileId?: string
  assembled: SceneShader
  run?: SceneRun
  onDiagnostics?: (diagnostics: WgslDiagnostic[] | undefined) => void
  onInsert?: (text: string) => void
}) {
  const canvas = useRef<HTMLCanvasElement>(null)
  const [pointActions, setPointActions] = useState<{ x: number; y: number; trigger: HTMLElement } | null>(null)
  const [body, setBody] = useState<World>('earth')
  const [projection, setProjection] = useState<Projection>('globe')
  const [points, setPoints] = useState<Partial<Record<BodyId, Point>>>({})
  const engine = useEarthWorkspace(canvas, assets, pickPoint, resources, initialView)
  const pendingPreview = useRef<number>(undefined)
  const previousProgram = useRef<{ source?: string }>(undefined)
  const previousRun = useRef<number | undefined>(undefined)
  const queuedProgram = useRef<{ source: string; delay: number }>(undefined)
  const diagnostics = engine.shaderDiagnostics
  const failed = engine.shaderSource === assembled.source && diagnostics.find(item => item.severity === 'error')
  const { runShader, cancelShader } = engine
  const computed = engine.frameSource === 'computed' && engine.shaderActive
  const populated = computed || (Boolean(engine.frameSource) && Boolean(failed))
  useEffect(() => {
    if ((populated && engine.worldSettled) || engine.surfaceFailed) onReady()
  }, [populated, engine.worldSettled, engine.surfaceFailed, onReady])

  useEffect(() => {
    if (engine.shaderSource !== assembled.source) {
      onDiagnostics?.(undefined)
      return
    }
    onDiagnostics?.(
      diagnostics.map(diagnostic => {
        const location = findShaderLocation(assembled.segments, diagnostic.line)
        return { ...diagnostic, fileId: location?.fileId ?? fileId, line: location?.line ?? diagnostic.line }
      }),
    )
  }, [engine.shaderSource, assembled, diagnostics, fileId, onDiagnostics])

  const sourceCode = assembled.source
  useEffect(() => {
    const forced = run && previousRun.current !== run.id
    const previous = previousProgram.current
    if (forced || previous?.source !== sourceCode) {
      cancelShader()
      previousProgram.current = { source: sourceCode }
      previousRun.current = run?.id
      queuedProgram.current = forced
        ? { source: run.source, delay: 0 }
        : sourceCode !== undefined
          ? { source: sourceCode, delay: previous ? 75 : 0 }
          : undefined
    }
    const pending = queuedProgram.current
    if (!pending) return
    pendingPreview.current = window.setTimeout(() => {
      queuedProgram.current = undefined
      runShader(pending.source)
    }, pending.delay)
    return () => window.clearTimeout(pendingPreview.current)
  }, [sourceCode, run, runShader, cancelShader])

  function project(next: Projection) {
    setProjection(next)
    engine.setProjection(next)
  }

  function focus(next: World) {
    setPointActions(null)
    if (next !== 'earth' && projection === 'plane') project('globe')
    setBody(next)
    engine.focus(next)
  }

  function stagePoint(source: BodyId, latitude: number, longitude: number) {
    setPoints(previous => ({ ...previous, [source]: { latitude, longitude } }))
    engine.place({ body: source, latitude, longitude })
    resources?.prefetchPoints([{ body: source, latitude, longitude }])
  }

  function pickPoint(source: BodyId, latitude: number, longitude: number) {
    if (body === source) stagePoint(source, latitude, longitude)
  }

  function openSurfaceActions(event: ReactMouseEvent<HTMLCanvasElement>) {
    const point = engine.pickSourceAt(event.clientX, event.clientY)
    if (!point || point.body !== body) return
    event.preventDefault()
    stagePoint(point.body, point.latitude, point.longitude)
    setPointActions({ x: event.clientX, y: event.clientY, trigger: event.currentTarget })
  }

  const source = body === 'moon' || body === 'mars' ? body : null
  const earth = body === 'earth'
  const inspection = engine.inspection

  const progress = (engine.busy || (earth && !engine.ready && !engine.error)) && engine.status
  const partial = earth && engine.shaderActive && engine.shaderPartial
  const scoreLabel = engine.shaderActive ? 'score' : 'similarity'
  const inspectionScore =
    inspection?.score == null
      ? engine.shaderActive
        ? 'Score unavailable'
        : 'Similarity unavailable'
      : `${(100 * inspection.score).toFixed(1)}% ${scoreLabel}`
  const inspectionElevation =
    inspection?.elevation == null ? undefined : `${Math.round(inspection.elevation).toLocaleString()} m`
  const point = source ? points[source] : undefined
  const pointPicker = source && (
    <PointPicker
      key={`${source}/${point?.latitude}/${point?.longitude}`}
      body={source}
      point={point}
      onInsert={onInsert}
      onCopy={engine.freezePoint}
      actions={pointActions}
      onActions={setPointActions}
    />
  )

  return (
    <div
      className="sky-map"
      data-focus={body}
      data-projection={projection}
      data-preview-file={fileId}
      data-with-footer={footer || undefined}
    >
      <div className="sky-preview-viewport">
        <canvas
          ref={canvas}
          className="sky-canvas"
          onContextMenu={openSurfaceActions}
          aria-label="Interactive solar system. Use arrow keys to move and plus or minus to zoom."
          tabIndex={0}
        />
        <div className="sky-map-overlay">
          <nav className="sky-nav" aria-label="Center camera">
            {worlds.map(world => (
              <button
                type="button"
                key={world}
                aria-label={worldName[world]}
                title={worldName[world]}
                aria-pressed={body === world}
                onClick={() => focus(world)}
              >
                <img src={worldImage[world]} alt="" width="40" height="40" />
              </button>
            ))}
          </nav>
          {!footer && (
            <section className="sky-status" aria-label="Map status">
              <div className="sky-status-live" role="status">
                {partial && <p className="sky-progress">Planetary point coverage is partial.</p>}
                {progress && <p className="sky-progress">{progress}</p>}
              </div>
              {failed && !engine.error && (
                <div className="sky-error">
                  <p role="alert">{failed.message}</p>
                </div>
              )}
              {engine.error && (
                <div className="sky-error">
                  <p role="alert">{engine.error}</p>
                  <button type="button" onClick={engine.retry}>
                    Retry
                  </button>
                </div>
              )}
            </section>
          )}
        </div>

        {!footer && earth && engine.ready && (
          <div className="sky-earth-tools">
            <div className="sky-projection sky-glass" role="group" aria-label="Earth projection">
              <button type="button" aria-pressed={projection === 'globe'} onClick={() => project('globe')}>
                Globe
              </button>
              <button type="button" aria-pressed={projection === 'plane'} onClick={() => project('plane')}>
                Flat map
              </button>
            </div>
            {engine.mapName && (
              <aside
                className="sky-legend sky-glass"
                aria-label={engine.shaderActive ? 'Relative score' : 'Relative similarity'}
              >
                <strong>{engine.shaderActive ? 'Relative score' : 'Relative similarity'}</strong>
                <i aria-hidden="true" />
                <div>
                  <span>Lower</span>
                  <span>Higher</span>
                </div>
              </aside>
            )}
          </div>
        )}
        {!footer && earth && inspection && (
          <div className="sky-inspection sky-glass">
            <strong>{formatPoint(inspection)}</strong>
            <div>
              <span>{inspectionScore}</span>
              {inspectionElevation && <span>{inspectionElevation}</span>}
            </div>
          </div>
        )}
        {!footer && source && <div className="sky-standalone-point-picker">{pointPicker}</div>}
      </div>

      {footer && (
        <footer className="sky-preview-footer" aria-label="Preview controls and status">
          {pointPicker}
          {earth && engine.ready && (
            <button
              type="button"
              title={projection === 'globe' ? 'Switch to flat map' : 'Switch to globe'}
              onClick={() => project(projection === 'globe' ? 'plane' : 'globe')}
            >
              {projection === 'globe' ? 'Globe' : 'Flat map'}
            </button>
          )}
          {earth && engine.ready && engine.mapName && (
            <div className="sky-preview-legend">
              <span>{engine.shaderActive ? 'Relative score' : 'Relative similarity'}</span>
              <span
                className="sky-preview-ramp"
                role="img"
                aria-label="Lower scores in dark purple to higher scores in pale yellow"
              >
                <span aria-hidden="true">Lower</span>
                <i />
                <span aria-hidden="true">Higher</span>
              </span>
            </div>
          )}
          {earth && inspection && (
            <div className="sky-preview-reading">
              <span>{formatPoint(inspection)}</span>
              <span>{inspectionScore}</span>
              {inspectionElevation && <span title="Elevation">{inspectionElevation}</span>}
            </div>
          )}
          {progress && <span role="status">{progress}</span>}
          {partial && (
            <span className="sky-preview-warning" role="status">
              Planetary point coverage is partial.
            </span>
          )}
          {(engine.error || failed) && (
            <div className="sky-preview-error">
              <span role="alert">{engine.error || (failed && failed.message)}</span>
              {engine.error && (
                <button type="button" onClick={engine.retry}>
                  Retry
                </button>
              )}
            </div>
          )}
        </footer>
      )}
    </div>
  )
}

function FilePreview({
  assets,
  file,
  files,
  resources,
  run,
  onDiagnostics,
  onInsert,
  footer,
  onPreviewReady,
}: {
  onPreviewReady: (fileId: string) => void
  footer?: boolean
  assets: WorkspaceAssets
  file: string
  files: ReturnType<typeof resolveShaderFiles>
  resources: EarthWorkspaceResources
  run?: SceneRun
  onDiagnostics: (fileId: string, diagnostics: WgslDiagnostic[] | undefined) => void
  onInsert: (fileId: string, text: string) => void
}) {
  const assembled = useMemo(() => assembleShaderProject(selectShaderModule(files, file)), [files, file])
  const diagnostics = useCallback(
    (value: WgslDiagnostic[] | undefined) => onDiagnostics(file, value),
    [file, onDiagnostics],
  )
  const insert = useCallback((text: string) => onInsert(file, text), [file, onInsert])
  const onReady = useCallback(() => onPreviewReady(file), [file, onPreviewReady])
  return (
    <SkyScene
      assets={assets}
      onReady={onReady}
      fileId={file}
      footer={footer}
      assembled={assembled}
      resources={resources}
      run={run}
      onDiagnostics={diagnostics}
      onInsert={insert}
      initialView={file === helloWorldExample.id ? helloWorldInitialView : undefined}
    />
  )
}

type WorkspaceProps = {
  assets: WorkspaceAssets
  resources?: EarthWorkspaceResources
  initialProject?: ShaderProject
  onProjectChange?: (project: ShaderProject) => void
  onReady: () => void
}

function StandaloneSkyWorkspace({ assets, onReady }: WorkspaceProps) {
  const sceneReady = useCallback(() => {
    onReady()
  }, [onReady])
  return (
    <main className="shared-sky">
      <SkyScene assets={assets} assembled={introShader} onReady={sceneReady} initialView={helloWorldInitialView} />
    </main>
  )
}

export default function SharedSkyWorkspace({
  assets,
  resources,
  initialProject,
  onProjectChange,
  onReady,
}: WorkspaceProps) {
  if (!WgslEditor) return <StandaloneSkyWorkspace assets={assets} onReady={onReady} />
  return (
    <ShaderWorkspace
      assets={assets}
      resources={resources}
      initialProject={initialProject}
      onProjectChange={onProjectChange}
      onReady={onReady}
    />
  )
}

function ShaderWorkspace({
  assets,
  resources: sharedResources,
  initialProject,
  onProjectChange,
  onReady,
}: WorkspaceProps) {
  const [resources] = useState(() => sharedResources ?? createEarthWorkspaceResources())
  const [workbenchFailed, setWorkbenchFailed] = useState(false)
  const [projectFiles, setProjectFiles] = useState<ShaderProject>(() => initialProject ?? defaultShaderProject())
  const currentProject = useRef(projectFiles)
  const projectLoaded = useRef(initialProject !== undefined)
  const persistedProject = useRef(projectFiles)
  const [vim, setVim] = useState(readVim)
  const [insertion, setInsertion] = useState<{ id: number; fileId: string; text: string }>()
  const clearInsertion = useCallback(
    (id: number) => setInsertion(previous => (previous?.id === id ? undefined : previous)),
    [],
  )
  const toggleVim = useCallback(() => setVim(previous => !previous), [])
  const sequence = useRef(0)
  const [hosts, setHosts] = useState(new Map<string, HTMLElement>())
  const [openPreviews, setOpenPreviews] = useState<readonly string[]>([])
  const [visiblePreviews, setVisiblePreviews] = useState<readonly string[]>([])
  const [startedPreviews, setStartedPreviews] = useState<ReadonlySet<string>>(() => new Set())
  const [readyPreviews, setReadyPreviews] = useState<ReadonlySet<string>>(() => new Set())
  const [layoutReady, setLayoutReady] = useState(false)
  const [fallbackReady, setFallbackReady] = useState(false)
  const [analyzerEnabled, setAnalyzerEnabled] = useState(false)
  const [analyzerReady, setAnalyzerReady] = useState(false)
  const onAnalyzerReady = useCallback(() => setAnalyzerReady(true), [])
  const [shaderRuns, setShaderRuns] = useState<Record<string, SceneRun>>({})
  const [sceneDiagnostics, setSceneDiagnostics] = useState<Record<string, WgslDiagnostic[]>>({})
  const prewarmParking = useRef<HTMLDivElement>(null)
  const prewarmedHosts = useRef(new Map<string, HTMLElement>())
  const [initialPreview] = useState(() => {
    startStartupSpan('monaco', 'prepare')
    return initialPreviewFileId(projectFiles.files, projectFiles.activeFileId)
  })

  // Pre-mount the initial preview canvas into an offscreen parking container
  // immediately so SurfaceRenderer, SolarScene, and the first WGSL score run in
  // parallel with Monaco workbench initialization, then move seamlessly into
  // the Monaco preview pane once it mounts.
  useLayoutEffect(() => {
    const parking = prewarmParking.current
    if (!initialPreview || !parking) return
    const host = document.createElement('div')
    host.className = 'sky-preview-host'
    parking.append(host)
    const map = prewarmedHosts.current
    map.set(initialPreview, host)
    setHosts(previous =>
      previous.get(initialPreview) === host ? previous : new Map(previous).set(initialPreview, host),
    )
    setOpenPreviews(previous => (previous.includes(initialPreview) ? previous : [...previous, initialPreview]))
    setStartedPreviews(previous => (previous.has(initialPreview) ? previous : new Set([...previous, initialPreview])))
    return () => {
      if (map.get(initialPreview) === host) {
        map.delete(initialPreview)
        host.remove()
        setHosts(previous => {
          if (previous.get(initialPreview) !== host) return previous
          const next = new Map(previous)
          next.delete(initialPreview)
          return next
        })
      }
    }
  }, [initialPreview])

  const takePrewarmedHost = useCallback((fileId: string) => {
    const host = prewarmedHosts.current.get(fileId)
    if (host) prewarmedHosts.current.delete(fileId)
    return host
  }, [])

  const shaderFiles = useMemo(() => resolveShaderFiles(projectFiles, DEFAULT_SCORE_SHADER), [projectFiles])
  const diagnostics = useMemo(() => Object.values(sceneDiagnostics).flat(), [sceneDiagnostics])
  const activeFile = projectFiles.activeFileId
  const onVisiblePreviews = useCallback((ids: readonly string[]) => {
    setVisiblePreviews(ids)
    setStartedPreviews(previous => new Set([...previous, ...ids]))
  }, [])
  const onPreviewReady = useCallback((id: string) => {
    setReadyPreviews(previous => (previous.has(id) ? previous : new Set([...previous, id])))
  }, [])
  const onLayoutReady = useCallback(() => {
    // Discard any prewarmed preview host that the restored layout didn't claim.
    if (prewarmedHosts.current.size) {
      const unclaimed = [...prewarmedHosts.current.entries()]
      prewarmedHosts.current.clear()
      for (const [, host] of unclaimed) host.remove()
      const ids = new Set(unclaimed.map(([id]) => id))
      setHosts(previous => {
        const next = new Map(previous)
        for (const id of ids) next.delete(id)
        return next
      })
      setOpenPreviews(previous => previous.filter(id => !ids.has(id)))
    }
    endStartupSpan('monaco')
    setLayoutReady(true)
  }, [])
  const onFallbackReady = useCallback(() => setFallbackReady(true), [])
  useEffect(() => {
    if (workbenchFailed ? fallbackReady : layoutReady && visiblePreviews.every(id => readyPreviews.has(id))) {
      onReady()
      let frame2 = 0
      const frame1 = requestAnimationFrame(() => {
        frame2 = requestAnimationFrame(() => setAnalyzerEnabled(true))
      })
      return () => {
        cancelAnimationFrame(frame1)
        cancelAnimationFrame(frame2)
      }
    }
  }, [workbenchFailed, fallbackReady, layoutReady, visiblePreviews, readyPreviews, onReady])
  useEffect(() => {
    if (!analyzerReady) return
    const controller = new AbortController()
    const { signal } = controller
    const priority = new Set<string>([activeFile, ...openPreviews])
    const ordered = [
      ...shaderFiles.filter(file => priority.has(file.id)),
      ...shaderFiles.filter(file => !file.closed && !priority.has(file.id)),
    ]
    void import('./planetaryShaderAnalysis.ts')
      .then(async ({ analyzePlanetaryShader, mentionsPlanetaryName }) => {
        for (const file of ordered) {
          if (signal.aborted) return
          const assembled = assembleShaderProject(selectShaderModule(shaderFiles, file.id)).source
          if (mentionsPlanetaryName(assembled) < 0) continue
          try {
            const calls = await analyzePlanetaryShader(assembled, signal)
            if (signal.aborted) return
            if (calls.length) resources.prefetchPoints(calls)
          } catch {
            // Ignore incomplete edits or analysis errors during background prefetch.
          }
        }
      })
      .catch(() => {})
    return () => controller.abort()
  }, [shaderFiles, activeFile, openPreviews, analyzerReady, resources])
  useEffect(() => {
    if (projectLoaded.current) return
    let cancelled = false
    void readShaderProject().then(loaded => {
      if (cancelled || projectLoaded.current) return
      projectLoaded.current = true
      persistedProject.current = loaded
      currentProject.current = loaded
      setProjectFiles(loaded)
      onProjectChange?.(loaded)
    })
    return () => {
      cancelled = true
    }
  }, [onProjectChange])
  useEffect(() => {
    if (!projectLoaded.current || projectFiles === persistedProject.current) return
    persistedProject.current = projectFiles
    void saveShaderProject(projectFiles)
  }, [projectFiles])
  useEffect(() => {
    try {
      localStorage.setItem(vimPreference, String(vim))
    } catch {
      /* Storage may be unavailable. */
    }
  }, [vim])

  function updateProject(next: ShaderProject) {
    projectLoaded.current = true
    currentProject.current = next
    setProjectFiles(next)
    onProjectChange?.(next)
  }

  function changeShader(id: string, source: string) {
    const project = currentProject.current
    const usesDefault =
      project.files.find(file => file.id === id)?.kind !== 'shader' &&
      shaderModuleId(id, project.files) === id &&
      source === defaultShaderSource(id, DEFAULT_SCORE_SHADER)
    updateProject({
      ...project,
      files: project.files.map(file => (file.id === id ? { ...file, source: usesDefault ? null : source } : file)),
    })
  }

  function showFile(id: string) {
    const project = currentProject.current
    const file = project.files.find(file => file.id === id)
    if (!file || (id === project.activeFileId && !file.closed)) return
    updateProject({
      activeFileId: id,
      files: project.files.map(file => (file.id === id && file.closed ? { ...file, closed: false } : file)),
    })
  }

  function addFile(kind: 'shader' | 'helper') {
    const { files } = currentProject.current
    let number = 1
    while (files.some(file => file.name.toLowerCase() === `shader-${number}.wgsl`)) number++
    const id = crypto.randomUUID()
    const source = kind === 'shader' ? DEFAULT_SCORE_SHADER : ''
    updateProject({ activeFileId: id, files: [...files, { id, name: `shader-${number}.wgsl`, source, kind }] })
  }

  function closeFile(id: string) {
    const project = currentProject.current
    const visible = project.files.filter(file => !file.closed)
    const index = visible.findIndex(file => file.id === id)
    if (index < 0) return false
    const replacement = visible[index + 1] ?? visible[index - 1]
    updateProject({
      activeFileId: project.activeFileId === id && replacement ? replacement.id : project.activeFileId,
      files: project.files.map(file => (file.id === id ? { ...file, closed: true } : file)),
    })
    return true
  }

  function renameFile(id: string, name: string) {
    const project = currentProject.current
    updateProject({ ...project, files: project.files.map(file => (file.id === id ? { ...file, name } : file)) })
  }

  function previewShader(fileId: string, source: string) {
    const run = { id: ++sequence.current, source }
    setShaderRuns(previous => ({ ...previous, [fileId]: run }))
  }

  const onPreviewHost = useCallback((fileId: string, host?: HTMLElement) => {
    if (host) {
      host.classList.add('sky-preview-host')
      setHosts(previous => {
        if (previous.get(fileId) === host) return previous
        return new Map(previous).set(fileId, host)
      })
    } else {
      setHosts(previous => {
        if (!previous.has(fileId)) return previous
        const next = new Map(previous)
        next.delete(fileId)
        return next
      })
      setReadyPreviews(previous => new Set([...previous].filter(id => id !== fileId)))
      setStartedPreviews(previous => new Set([...previous].filter(id => id !== fileId)))
      setShaderRuns(previous => Object.fromEntries(Object.entries(previous).filter(([id]) => id !== fileId)))
      setSceneDiagnostics(previous => {
        if (!previous[fileId]) return previous
        const next = { ...previous }
        delete next[fileId]
        return next
      })
    }
  }, [])

  const updateDiagnostics = useCallback((fileId: string, diagnostics: WgslDiagnostic[] | undefined) => {
    setSceneDiagnostics(previous => {
      if (diagnostics?.length) return { ...previous, [fileId]: diagnostics }
      if (!previous[fileId]) return previous
      const next = { ...previous }
      delete next[fileId]
      return next
    })
  }, [])
  const insertPoint = useCallback((fileId: string, text: string) => {
    setInsertion({ id: ++sequence.current, fileId, text })
  }, [])

  function renderPreview(fileId: string) {
    return (
      <FilePreview
        key={fileId}
        assets={assets}
        file={fileId}
        onPreviewReady={workbenchFailed ? onFallbackReady : onPreviewReady}
        footer={!workbenchFailed}
        files={shaderFiles}
        resources={resources}
        run={shaderRuns[fileId]}
        onDiagnostics={updateDiagnostics}
        onInsert={insertPoint}
      />
    )
  }

  return (
    <main className="shared-sky" data-with-editor="" data-workbench-failed={workbenchFailed ? '' : undefined}>
      <div
        ref={prewarmParking}
        aria-hidden="true"
        style={{
          position: 'fixed',
          left: 0,
          top: 0,
          width: '60vw',
          height: '100vh',
          visibility: 'hidden',
          pointerEvents: 'none',
          zIndex: -1,
        }}
      />
      {WgslEditor && (
        <section className="sky-workbench" aria-label="Earth score query">
          <div className="sky-wgsl-panel">
            <ShaderEditorBoundary onError={() => setWorkbenchFailed(true)}>
              <Suspense fallback={null}>
                <WgslEditor
                  onPreviewHost={onPreviewHost}
                  onPreviewOpenChange={setOpenPreviews}
                  onVisiblePreviews={onVisiblePreviews}
                  takePrewarmedHost={takePrewarmedHost}
                  analyzerEnabled={analyzerEnabled}
                  onAnalyzerReady={onAnalyzerReady}
                  onLayoutReady={onLayoutReady}
                  onStartError={() => setWorkbenchFailed(true)}
                  files={shaderFiles}
                  activeFileId={activeFile}
                  onChange={changeShader}
                  diagnostics={diagnostics}
                  onActiveFileChange={showFile}
                  onCloseFile={closeFile}
                  onNewFile={addFile}
                  onRenameFile={renameFile}
                  onReopenFile={showFile}
                  onToggleVim={toggleVim}
                  prelude={SCORE_SHADER_PRELUDE}
                  label="Earth score WGSL"
                  vim={vim}
                  onRun={previewShader}
                  insertion={insertion}
                  onInserted={clearInsertion}
                />
              </Suspense>
            </ShaderEditorBoundary>
          </div>
        </section>
      )}
      {workbenchFailed
        ? renderPreview(activeFile)
        : openPreviews.map(fileId => {
            const host = hosts.get(fileId)
            if (host && startedPreviews.has(fileId)) return createPortal(renderPreview(fileId), host, fileId)
            return null
          })}
    </main>
  )
}
