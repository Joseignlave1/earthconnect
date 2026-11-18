import type { WorkspaceAssets } from './workspaceAssets.ts'
import { Component, lazy, Suspense, useCallback, useEffect, useId, useMemo, useRef, useState, type MouseEvent as ReactMouseEvent, type ReactNode } from 'react'
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
import { assembleShaderProject, defaultShaderSource, findShaderLocation, readShaderProject, resolveShaderFiles, saveShaderProject, selectShaderModule, shaderModuleId, type ShaderProject } from './shaderProject'
import { formatPointCall, type PlanetaryCoordinates } from './planetaryPointCache.ts'
import './SharedSkyWorkspace.css'

const WgslEditor = import.meta.env.VITE_WORKSPACE_STANDALONE ? null : lazy(() => import('./WgslEditor'))

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
  try { return localStorage.getItem(vimPreference) === 'true' } catch { return false }
}

function PointPicker({ body, point, onInsert, onCopy, actions, onActions }: {
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
        ({ code, partial } = await onCopy(reference, controller.signal))
      } catch (error) {
        if (!controller.signal.aborted) setCopyStatus(error instanceof Error ? error.message : String(error))
        return
      }
    }
    if (controller.signal.aborted) return
    try {
      await navigator.clipboard.writeText(code)
      if (!controller.signal.aborted) setCopyStatus(partial ? 'Copied available values. Coverage is partial.' : 'Copied.')
    } catch {
      if (controller.signal.aborted) return
      setCopyFallback(code)
      setCopyStatus(partial ? 'Copy the available values below. Coverage is partial.' : 'Clipboard unavailable. Copy the text below.')
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

  return <div className="sky-point-picker">
    <button type="button" title="Copy coordinates" aria-label={`Copy coordinates for ${text}`}
      onClick={() => void copy('coordinates')} onContextMenu={openAtButton}>
      {text}
    </button>
    <button type="button" title="Point actions" aria-label={`${worldName[body]} point actions`}
      aria-haspopup="menu" aria-expanded={actions !== null} aria-controls={menuId} onClick={openAtButton}>⋯</button>
    <div id={menuId} ref={menu} popover="auto" className="sky-point-menu" role="menu" aria-label="Point actions"
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
        const next = event.key === 'Home' ? 0 : event.key === 'End' ? items.length - 1
          : (current + (event.key === 'ArrowDown' ? 1 : -1) + items.length) % items.length
        items[next]?.focus()
      }}>
      <button type="button" role="menuitem" tabIndex={-1} onClick={() => { closeMenu(); void copy('expression') }}>Copy as WGSL expression</button>
      {onInsert && <button type="button" role="menuitem" tabIndex={-1} onClick={() => { closeMenu(false); onInsert(formatPointCall({ body, ...point })) }}>
        Insert WGSL expression
      </button>}
      <button type="button" role="menuitem" tabIndex={-1} onClick={() => { closeMenu(); void copy('sample') }}>Copy sampled WGSL values</button>
    </div>
    {copyStatus && <span className="sky-point-feedback" role="status">{copyStatus}</span>}
    {copyFallback && <div className="sky-point-fallback">
      <textarea ref={fallback} readOnly aria-label="Text to copy" value={copyFallback} onFocus={event => event.target.select()} />
      <button type="button" onClick={() => { setCopyFallback(''); setCopyStatus('') }}>Dismiss</button>
    </div>}
  </div>
}

type SceneRun = { id: number; source: string }
type SceneShader = ReturnType<typeof assembleShaderProject>
const introShader = assembleShaderProject([helloWorldExample])

function SkyScene({ assets, resources, fileId, assembled, snapshotOnly, run, onDiagnostics, onInsert, footer, onReady, preloadWorld = true, initialView }: {
  onReady: (computed: boolean) => void
  preloadWorld?: boolean
  initialView?: { latitude: number; longitude: number }
  footer?: boolean
  assets: WorkspaceAssets
  resources?: EarthWorkspaceResources
  fileId?: string
  assembled?: SceneShader
  snapshotOnly?: boolean
  run?: SceneRun
  onDiagnostics?: (diagnostics: WgslDiagnostic[] | undefined) => void
  onInsert?: (text: string) => void
}) {
  const canvas = useRef<HTMLCanvasElement>(null)
  const [pointActions, setPointActions] = useState<{ x: number; y: number; trigger: HTMLElement } | null>(null)
  const [body, setBody] = useState<World>('earth')
  const [projection, setProjection] = useState<Projection>('globe')
  const [points, setPoints] = useState<Partial<Record<BodyId, Point>>>({})
  const engine = useEarthWorkspace(canvas, assets, pickPoint, resources, preloadWorld, initialView)
  const pendingPreview = useRef<number>(undefined)
  const previousProgram = useRef<{ source?: string; snapshotOnly?: boolean }>(undefined)
  const previousRun = useRef<number | undefined>(undefined)
  const queuedProgram = useRef<{ source: string; delay: number }>(undefined)
  const diagnostics = engine.shaderDiagnostics
  const failed = engine.shaderSource === assembled?.source && diagnostics.find(item => item.severity === 'error')
  const { runShader, cancelShader } = engine
  const needsComputedScore = assembled !== undefined && (!snapshotOnly || run !== undefined)
  const computed = engine.frameSource === 'computed' && engine.shaderActive
  const populated = needsComputedScore
    ? computed || (engine.frameSource === 'snapshot' && failed)
    : engine.frameSource === 'snapshot'
  const dataSettled = !preloadWorld || engine.worldSettled
  useEffect(() => {
    if ((populated && dataSettled) || engine.surfaceFailed) onReady(computed)
  }, [populated, dataSettled, computed, engine.surfaceFailed, onReady])

  useEffect(() => {
    if (!assembled || engine.shaderSource !== assembled.source) {
      onDiagnostics?.(undefined)
      return
    }
    onDiagnostics?.(diagnostics.map(diagnostic => {
      const location = findShaderLocation(assembled.segments, diagnostic.line)
      return { ...diagnostic, fileId: location?.fileId ?? fileId, line: location?.line ?? diagnostic.line }
    }))
  }, [engine.shaderSource, assembled, diagnostics, fileId, onDiagnostics])

  const sourceCode = assembled?.source
  useEffect(() => {
    const forced = run && previousRun.current !== run.id
    const previous = previousProgram.current
    if (forced || previous?.source !== sourceCode) {
      cancelShader()
      previousProgram.current = { source: sourceCode, snapshotOnly }
      previousRun.current = run?.id
      queuedProgram.current = forced ? { source: run.source, delay: 0 } :
        sourceCode !== undefined && !(snapshotOnly && (!previous || previous.snapshotOnly))
          ? { source: sourceCode, delay: previous ? 550 : 0 }
          : undefined
    }
    const pending = queuedProgram.current
    if (!pending) return
    pendingPreview.current = window.setTimeout(() => {
      queuedProgram.current = undefined
      runShader(pending.source)
    }, pending.delay)
    return () => window.clearTimeout(pendingPreview.current)
  }, [sourceCode, snapshotOnly, run, runShader, cancelShader])

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
  const inspectionScore = inspection?.score == null ? (engine.shaderActive ? 'Score unavailable' : 'Similarity unavailable')
    : `${(100 * inspection.score).toFixed(1)}% ${scoreLabel}`
  const inspectionElevation = inspection?.elevation == null ? undefined : `${Math.round(inspection.elevation).toLocaleString()} m`
  const point = source ? points[source] : undefined
  const pointPicker = source && <PointPicker key={`${source}/${point?.latitude}/${point?.longitude}`} body={source} point={point} onInsert={onInsert}
    onCopy={engine.freezePoint} actions={pointActions} onActions={setPointActions} />

  return (
    <div className="sky-map" data-focus={body} data-projection={projection} data-preview-file={fileId} data-with-footer={footer || undefined}>
      <div className="sky-preview-viewport">
        <canvas ref={canvas} className="sky-canvas" onContextMenu={openSurfaceActions} aria-label="Interactive solar system. Use arrow keys to move and plus or minus to zoom." tabIndex={0} />
        <div className="sky-map-overlay">
          <nav className="sky-nav" aria-label="Center camera">
            {worlds.map(world => (
              <button type="button" key={world} aria-label={worldName[world]} title={worldName[world]}
                aria-pressed={body === world} onClick={() => focus(world)}>
                <img src={worldImage[world]} alt="" width="40" height="40" />
              </button>
            ))}
          </nav>
          {!footer && <section className="sky-status" aria-label="Map status">
            <div className="sky-status-live" role="status">
              {partial && <p className="sky-progress">Planetary point coverage is partial.</p>}
              {progress && <p className="sky-progress">{progress}</p>}
            </div>
            {failed && !engine.error && <div className="sky-error"><p role="alert">{failed.message}</p></div>}
            {engine.error && <div className="sky-error"><p role="alert">{engine.error}</p><button type="button" onClick={engine.retry}>Retry</button></div>}
          </section>}
        </div>

        {!footer && earth && engine.ready && <div className="sky-earth-tools">
          <div className="sky-projection sky-glass" role="group" aria-label="Earth projection">
            <button type="button" aria-pressed={projection === 'globe'} onClick={() => project('globe')}>Globe</button>
            <button type="button" aria-pressed={projection === 'plane'} onClick={() => project('plane')}>Flat map</button>
          </div>
          {engine.mapName && <aside className="sky-legend sky-glass" aria-label={engine.shaderActive ? 'Relative score' : 'Relative similarity'}>
            <strong>{engine.shaderActive ? 'Relative score' : 'Relative similarity'}</strong><i aria-hidden="true" />
            <div><span>Lower</span><span>Higher</span></div>
          </aside>}
        </div>}
        {!footer && earth && inspection && <div className="sky-inspection sky-glass">
          <strong>{formatPoint(inspection)}</strong>
          <div><span>{inspectionScore}</span>{inspectionElevation && <span>{inspectionElevation}</span>}</div>
        </div>}
        {!footer && source && <div className="sky-standalone-point-picker">{pointPicker}</div>}
      </div>

      {footer && <footer className="sky-preview-footer" aria-label="Preview controls and status">
        {pointPicker}
        {earth && engine.ready && <button type="button"
          title={projection === 'globe' ? 'Switch to flat map' : 'Switch to globe'}
          onClick={() => project(projection === 'globe' ? 'plane' : 'globe')}>
          {projection === 'globe' ? 'Globe' : 'Flat map'}
        </button>}
        {earth && engine.ready && engine.mapName && <div className="sky-preview-legend">
          <span>{engine.shaderActive ? 'Relative score' : 'Relative similarity'}</span>
          <span className="sky-preview-ramp" role="img" aria-label="Lower scores in dark purple to higher scores in pale yellow">
            <span aria-hidden="true">Lower</span><i /><span aria-hidden="true">Higher</span>
          </span>
        </div>}
        {earth && inspection && <div className="sky-preview-reading">
            <span>{formatPoint(inspection)}</span>
            <span>{inspectionScore}</span>
            {inspectionElevation && <span title="Elevation">{inspectionElevation}</span>}
        </div>}
        {progress && <span role="status">{progress}</span>}
        {partial && <span className="sky-preview-warning" role="status">Planetary point coverage is partial.</span>}
        {(engine.error || failed) && <div className="sky-preview-error">
          <span role="alert">{engine.error || (failed && failed.message)}</span>
          {engine.error && <button type="button" onClick={engine.retry}>Retry</button>}
        </div>}
      </footer>}
    </div>
  )

}

function FilePreview({ assets, file, files, snapshotOnly, resources, run, onDiagnostics, onInsert, footer, onPreviewReady }: {
  onPreviewReady: (fileId: string) => void
  footer?: boolean
  assets: WorkspaceAssets
  file: string
  files: ReturnType<typeof resolveShaderFiles>
  snapshotOnly?: boolean
  resources: EarthWorkspaceResources
  run?: SceneRun
  onDiagnostics: (fileId: string, diagnostics: WgslDiagnostic[] | undefined) => void
  onInsert: (fileId: string, text: string) => void
}) {
  const assembled = useMemo(() => assembleShaderProject(selectShaderModule(files, file)), [files, file])
  const diagnostics = useCallback((value: WgslDiagnostic[] | undefined) => onDiagnostics(file, value), [file, onDiagnostics])
  const insert = useCallback((text: string) => onInsert(file, text), [file, onInsert])
  const onReady = useCallback(() => onPreviewReady(file), [file, onPreviewReady])
  return <SkyScene assets={assets} onReady={onReady} fileId={file} footer={footer} assembled={assembled} snapshotOnly={snapshotOnly} resources={resources}
    run={run} onDiagnostics={diagnostics} onInsert={insert} initialView={file === helloWorldExample.id ? helloWorldInitialView : undefined} />
}

type WorkspaceProps = { assets: WorkspaceAssets; onReady: () => void }

const standaloneIntroKey = 'earth-connect-standalone-intro-seen'

function StandaloneSkyWorkspace({ assets, onReady }: WorkspaceProps) {
  const [showIntro] = useState(() => {
    try { return localStorage.getItem(standaloneIntroKey) !== 'true' } catch { return true }
  })
  const sceneReady = useCallback((computed: boolean) => {
    if (showIntro && computed) {
      try { localStorage.setItem(standaloneIntroKey, 'true') } catch { /* Storage may be unavailable. */ }
    }
    onReady()
  }, [showIntro, onReady])
  return <main className="shared-sky">
    <SkyScene assets={assets} assembled={showIntro ? introShader : undefined} preloadWorld={showIntro} onReady={sceneReady}
      initialView={showIntro ? helloWorldInitialView : undefined} />
  </main>
}

export default function SharedSkyWorkspace({ assets, onReady }: WorkspaceProps) {
  if (!WgslEditor) return <StandaloneSkyWorkspace assets={assets} onReady={onReady} />
  return <ShaderWorkspace assets={assets} onReady={onReady} />
}

function ShaderWorkspace({ assets, onReady }: WorkspaceProps) {
  const [resources] = useState(createEarthWorkspaceResources)
  const [workbenchFailed, setWorkbenchFailed] = useState(false)
  const [projectFiles, setProjectFiles] = useState(readShaderProject)
  const currentProject = useRef(projectFiles)
  const [vim, setVim] = useState(readVim)
  const [insertion, setInsertion] = useState<{ id: number; fileId: string; text: string }>()
  const sequence = useRef(0)
  const [hosts, setHosts] = useState(new Map<string, HTMLElement>())
  const [openPreviews, setOpenPreviews] = useState<readonly string[]>([])
  const [visiblePreviews, setVisiblePreviews] = useState<readonly string[]>([])
  const [startedPreviews, setStartedPreviews] = useState<ReadonlySet<string>>(() => new Set())
  const [readyPreviews, setReadyPreviews] = useState<ReadonlySet<string>>(() => new Set())
  const [layoutReady, setLayoutReady] = useState(false)
  const [fallbackReady, setFallbackReady] = useState(false)
  const [shaderRuns, setShaderRuns] = useState<Record<string, SceneRun>>({})
  const [sceneDiagnostics, setSceneDiagnostics] = useState<Record<string, WgslDiagnostic[]>>({})

  const shaderFiles = useMemo(() => resolveShaderFiles(projectFiles, DEFAULT_SCORE_SHADER), [projectFiles])
  const diagnostics = useMemo(() => Object.values(sceneDiagnostics).flat(), [sceneDiagnostics])
  const sampleFile = projectFiles.files.find(file => file.id === 'score')!
  const activeFile = projectFiles.activeFileId
  const onVisiblePreviews = useCallback((ids: readonly string[]) => {
    setVisiblePreviews(ids)
    setStartedPreviews(previous => new Set([...previous, ...ids]))
  }, [])
  const onPreviewReady = useCallback((id: string) => {
    setReadyPreviews(previous => previous.has(id) ? previous : new Set([...previous, id]))
  }, [])
  const onLayoutReady = useCallback(() => setLayoutReady(true), [])
  const onFallbackReady = useCallback(() => setFallbackReady(true), [])
  useEffect(() => {
    if (workbenchFailed ? fallbackReady : layoutReady && visiblePreviews.every(id => readyPreviews.has(id))) {
      onReady()
    }
  }, [workbenchFailed, fallbackReady, layoutReady, visiblePreviews, readyPreviews, onReady])
  useEffect(() => { saveShaderProject(projectFiles) }, [projectFiles])
  useEffect(() => {
    try { localStorage.setItem(vimPreference, String(vim)) } catch { /* Storage may be unavailable. */ }
  }, [vim])

  function updateProject(next: ShaderProject) {
    currentProject.current = next
    setProjectFiles(next)
  }

  function changeShader(id: string, source: string) {
    const project = currentProject.current
    const usesDefault = project.files.find(file => file.id === id)?.kind !== 'shader' && shaderModuleId(id, project.files) === id && source === defaultShaderSource(id, DEFAULT_SCORE_SHADER)
    updateProject({
      ...project,
      files: project.files.map(file => file.id === id ? { ...file, source: usesDefault ? null : source } : file),
    })
  }

  function showFile(id: string) {
    const project = currentProject.current
    const file = project.files.find(file => file.id === id)
    if (!file || (id === project.activeFileId && !file.closed)) return
    updateProject({
      activeFileId: id,
      files: project.files.map(file => file.id === id && file.closed ? { ...file, closed: false } : file),
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
      files: project.files.map(file => file.id === id ? { ...file, closed: true } : file),
    })
    return true
  }

  function renameFile(id: string, name: string) {
    const project = currentProject.current
    updateProject({ ...project, files: project.files.map(file => file.id === id ? { ...file, name } : file) })
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
    const snapshotOnly = fileId === 'score' && sampleFile.source === null &&
      selectShaderModule(shaderFiles, 'score').every(file => file.id === 'score' || !file.source.trim())
    return <FilePreview key={fileId} assets={assets} file={fileId} onPreviewReady={workbenchFailed ? onFallbackReady : onPreviewReady} footer={!workbenchFailed} files={shaderFiles} snapshotOnly={snapshotOnly} resources={resources}
      run={shaderRuns[fileId]}
      onDiagnostics={updateDiagnostics} onInsert={insertPoint} />
  }

  return (
    <main className="shared-sky" data-with-editor="" data-workbench-failed={workbenchFailed ? '' : undefined}>
      {WgslEditor && <section className="sky-workbench" aria-label="Earth score query">
        <div className="sky-wgsl-panel">
          <ShaderEditorBoundary onError={() => setWorkbenchFailed(true)}>
            <Suspense fallback={null}>
              <WgslEditor onPreviewHost={onPreviewHost} onPreviewOpenChange={setOpenPreviews} onVisiblePreviews={onVisiblePreviews}
                onLayoutReady={onLayoutReady} onStartError={() => setWorkbenchFailed(true)}
                files={shaderFiles} activeFileId={activeFile} onChange={changeShader} diagnostics={diagnostics}
                onActiveFileChange={showFile} onCloseFile={closeFile}
                onNewFile={addFile} onRenameFile={renameFile} onReopenFile={showFile}
                onToggleVim={() => setVim(previous => !previous)}
                prelude={SCORE_SHADER_PRELUDE} label="Earth score WGSL" vim={vim}
                onRun={previewShader}
                insertion={insertion}
                onInserted={id => setInsertion(previous => previous?.id === id ? undefined : previous)} />
            </Suspense>
          </ShaderEditorBoundary>
        </div>
      </section>}
      {workbenchFailed ? renderPreview(activeFile) : openPreviews.map(fileId => {
        const host = hosts.get(fileId)
        if (host && startedPreviews.has(fileId)) return createPortal(renderPreview(fileId), host, fileId)
        return null
      })}
    </main>
  )
}
