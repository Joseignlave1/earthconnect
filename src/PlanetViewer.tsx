import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties, type FormEvent } from 'react'
import * as Cesium from 'cesium'
import 'cesium/Build/Cesium/Widgets/widgets.css'
import marsRender from './assets/mars-render.png'
import moonRender from './assets/moon-render.png'
import { initialCriteria, type BodyId, type Criterion, type EarthMatch } from './explorerData'
import { API_BASE_URL, applySample, buildMatchQuery, criteriaFor, friendlyApiError, loadDatasource, rankScoreDataset, runQuery, samplePoint, type EarthDatasource, type SampleMeasurement, type ScoreDataset } from './earthIndexApi'
import './PlanetViewer.css'

type Target = { lat: number; lon: number }
type Tool = 'filters' | 'console'

const PLANETS: Record<BodyId, { name: string; image: string; tileRoot: string }> = {
  moon: {
    name: 'Moon',
    image: moonRender,
    tileRoot: 'https://trek.nasa.gov/tiles/Moon/EQ/LRO_WAC_Mosaic_Global_303ppd_v02/1.0.0/default/default028mm',
  },
  mars: {
    name: 'Mars',
    image: marsRender,
    tileRoot: 'https://trek.nasa.gov/tiles/Mars/EQ/Mars_Viking_MDIM21_ClrMosaic_global_232m/1.0.0/default/default028mm',
  },
}

function makeViewer(container: HTMLDivElement, ellipsoid: Cesium.Ellipsoid, provider: Cesium.ImageryProvider) {
  const viewer = new Cesium.Viewer(container, {
    baseLayer: new Cesium.ImageryLayer(provider),
    terrainProvider: new Cesium.EllipsoidTerrainProvider({ ellipsoid }),
    baseLayerPicker: false, geocoder: false, homeButton: false, sceneModePicker: false, navigationHelpButton: false,
    animation: false, timeline: false, fullscreenButton: false, infoBox: false, selectionIndicator: false,
  })
  viewer.scene.highDynamicRange = false
  viewer.scene.fog.enabled = false
  viewer.scene.globe.enableLighting = false
  viewer.scene.globe.showGroundAtmosphere = false
  viewer.scene.globe.baseColor = Cesium.Color.fromCssColorString('#080c13')
  const imagery = viewer.imageryLayers.get(0)
  imagery.brightness = .72
  imagery.contrast = 1.12
  imagery.saturation = .58
  imagery.gamma = .94
  viewer.scene.screenSpaceCameraController.inertiaSpin = 0
  viewer.scene.screenSpaceCameraController.inertiaTranslate = 0
  viewer.scene.screenSpaceCameraController.inertiaZoom = 0
  if (viewer.scene.skyAtmosphere) viewer.scene.skyAtmosphere.show = false
  return viewer
}

function PlanetGlobe({ body, target, onSelect }: { body: BodyId; target: Target; onSelect: (target: Target) => void }) {
  const orbRef = useRef<HTMLDivElement>(null)
  const dragRef = useRef({ x: 0, rotation: 0, moved: false })
  const [rotation, setRotation] = useState(0)
  const [zoom, setZoom] = useState(1)
  const [dragging, setDragging] = useState(false)
  const clampZoom = (value: number) => Math.max(1, Math.min(8, value))
  const textureTiles = Array.from({ length: 8 }, (_, index) => {
    const x = index % 4
    const y = Math.floor(index / 4)
    return `${PLANETS[body].tileRoot}/1/${y}/${x}.jpg`
  })
  const selectFromPointer = (clientX: number, clientY: number) => {
    const orb = orbRef.current
    if (!orb) return
    const rect = orb.getBoundingClientRect()
    const x = (clientX - rect.left) / rect.width
    const y = (clientY - rect.top) / rect.height
    const dx = (x - .5) * 2
    const dy = (y - .5) * 2
    if (dx * dx + dy * dy > 1) return
    const visibleLongitude = Cesium.Math.toDegrees(Math.asin(Math.max(-1, Math.min(1, dx))))
    const longitude = ((((visibleLongitude + rotation + 180) % 360) + 360) % 360) - 180
    const latitude = Cesium.Math.toDegrees(Math.asin(Math.max(-1, Math.min(1, -dy))))
    onSelect({ lat: latitude, lon: longitude })
  }
  const relativeLongitude = Cesium.Math.negativePiToPi(Cesium.Math.toRadians(target.lon - rotation))
  const markerX = 50 + Math.sin(relativeLongitude) * 34
  const markerY = 50 - Math.sin(Cesium.Math.toRadians(target.lat)) * 34
  const markerVisible = Math.cos(relativeLongitude) > 0
  const normalizedRotation = ((rotation % 360) + 360) % 360
  return <div className="ex-planet-stage">
    <div
      aria-label={`Interactive ${PLANETS[body].name} globe. Drag to rotate and click to select a point.`}
      className="ex-planet-orb"
      onPointerDown={(event) => {
        event.currentTarget.setPointerCapture(event.pointerId)
        dragRef.current = { x: event.clientX, rotation, moved: false }
        setDragging(true)
      }}
      onPointerMove={(event) => {
        if (!event.currentTarget.hasPointerCapture(event.pointerId)) return
        const delta = event.clientX - dragRef.current.x
        if (Math.abs(delta) > 3) dragRef.current.moved = true
        setRotation(dragRef.current.rotation - delta * .7)
      }}
      onPointerUp={(event) => {
        if (!dragRef.current.moved) selectFromPointer(event.clientX, event.clientY)
        event.currentTarget.releasePointerCapture(event.pointerId)
        setDragging(false)
      }}
      onPointerCancel={() => setDragging(false)}
      onWheel={(event) => {
        event.preventDefault()
        setZoom((current) => clampZoom(current - event.deltaY * .004))
      }}
      ref={orbRef}
      role="application"
      tabIndex={0}
    >
      <span className="ex-planet-surface" style={{ backgroundImage: `url(${PLANETS[body].image})`, transform: `scale(${zoom})` }}>
        <span className="ex-texture-track" style={{ transform: `translateX(${-normalizedRotation / 360 * 50}%)` }}>
          {[0, 1].map((copy) => <span className="ex-texture-world" key={copy}>
            {textureTiles.map((tile) => <img alt="" draggable="false" key={`${copy}-${tile}`} src={tile} />)}
          </span>)}
        </span>
        <span className="ex-planet-shade" />
      </span>
      <span className="ex-planet-marker" style={{ left: `${markerX}%`, opacity: markerVisible ? 1 : 0, top: `${markerY}%` }}><i /></span>
    </div>
    <span className="ex-orbit-line" aria-hidden="true" />
    <div className="ex-planet-controls" aria-label="Planet zoom controls">
      <button aria-label="Zoom out" disabled={zoom <= 1} onClick={() => setZoom((current) => clampZoom(current - .5))} type="button">−</button>
      <button aria-label="Reset rotation and zoom" className="ex-zoom-readout" onClick={() => { setRotation(0); setZoom(1) }} type="button">{dragging ? 'Rotating' : `${zoom.toFixed(1)}×`}</button>
      <button aria-label="Zoom in" disabled={zoom >= 8} onClick={() => setZoom((current) => clampZoom(current + .5))} type="button">+</button>
    </div>
  </div>
}

function CoordinatePicker({ target, onSelect }: { target: Target; onSelect: (target: Target) => void }) {
  const [latitude, setLatitude] = useState(target.lat.toFixed(3))
  const [longitude, setLongitude] = useState(target.lon.toFixed(3))
  const [error, setError] = useState('')
  const submit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    const lat = Number(latitude)
    const lon = Number(longitude)
    if (!Number.isFinite(lat) || lat < -90 || lat > 90) {
      setError('Latitude must be between −90 and 90')
      return
    }
    if (!Number.isFinite(lon) || lon < -180 || lon > 180) {
      setError('Longitude must be between −180 and 180')
      return
    }
    setError('')
    onSelect({ lat, lon })
  }
  return <form className="ex-coordinates" onSubmit={submit}>
    <span>Select by coordinates</span>
    <div className="ex-coordinate-fields">
      <label>Lat<input aria-label="Latitude" max="90" min="-90" onChange={(event) => setLatitude(event.target.value)} required step="any" type="number" value={latitude} /></label>
      <label>Lon<input aria-label="Longitude" max="180" min="-180" onChange={(event) => setLongitude(event.target.value)} required step="any" type="number" value={longitude} /></label>
      <button type="submit">Load</button>
    </div>
    <i className={error ? 'is-error' : ''}>{error || `${target.lat.toFixed(3)}°, ${target.lon.toFixed(3)}° selected`}</i>
  </form>
}

function showMatches(viewer: Cesium.Viewer, matches: EarthMatch[]) {
  viewer.entities.removeAll()
  matches.forEach((match, index) => {
    const featured = index < 5
    const color = Cesium.Color.fromCssColorString(featured ? '#ff6b4a' : '#83b9ff')
    viewer.entities.add({
      id: `match-${match.id}`,
      name: match.title,
      position: Cesium.Cartesian3.fromDegrees(match.lon, match.lat, 6500),
      point: {
        pixelSize: featured ? 8 : 4,
        color: color.withAlpha(featured ? .78 : .24),
        outlineColor: featured ? Cesium.Color.WHITE.withAlpha(.62) : color.withAlpha(.38),
        outlineWidth: featured ? 1.5 : 1,
        disableDepthTestDistance: 0,
      },
    })
  })
}

function EarthGlobe({ matches, onOpen }: { matches: EarthMatch[]; onOpen: (match: EarthMatch) => void }) {
  const containerRef = useRef<HTMLDivElement>(null)
  const viewerRef = useRef<Cesium.Viewer | null>(null)
  const matchesRef = useRef(matches)
  useEffect(() => { matchesRef.current = matches }, [matches])
  useEffect(() => {
    const container = containerRef.current
    if (!container) return
    let cancelled = false
    let viewer: Cesium.Viewer | null = null
    let handler: Cesium.ScreenSpaceEventHandler | null = null
    void Cesium.TileMapServiceImageryProvider.fromUrl('/cesium/Assets/Textures/NaturalEarthII').then((provider) => {
      if (cancelled) return
      const ellipsoid = Cesium.Ellipsoid.WGS84
      viewer = makeViewer(container, ellipsoid, provider)
      viewerRef.current = viewer
      viewer.camera.setView({ destination: Cesium.Cartesian3.fromDegrees(-18, 12, ellipsoid.maximumRadius * 2.45), orientation: { heading: 0, pitch: -Cesium.Math.PI_OVER_TWO, roll: 0 } })
      showMatches(viewer, matchesRef.current)
      handler = new Cesium.ScreenSpaceEventHandler(viewer.scene.canvas)
      handler.setInputAction((event: Cesium.ScreenSpaceEventHandler.PositionedEvent) => {
        if (!viewer) return
        const id = viewer.scene.pick(event.position)?.id?.id
        const match = typeof id === 'string' && id.startsWith('match-') ? matchesRef.current.find((item) => item.id === id.slice(6)) : undefined
        if (match) onOpen(match)
      }, Cesium.ScreenSpaceEventType.LEFT_CLICK)
    })
    return () => {
      cancelled = true
      handler?.destroy()
      viewerRef.current = null
      if (viewer && !viewer.isDestroyed()) viewer.destroy()
    }
  }, [onOpen])
  useEffect(() => {
    const viewer = viewerRef.current
    if (!viewer || viewer.isDestroyed()) return
    showMatches(viewer, matches)
  }, [matches])
  return <div className="ex-earth-canvas" ref={containerRef} />
}

function NumberField({ value, label, onChange }: { value: number | null; label: string; onChange: (value: number | null) => void }) {
  return <input aria-label={label} type="number" step="any" value={value ?? ''} onChange={(event) => onChange(event.target.value === '' ? null : event.target.valueAsNumber)} />
}

function measurementValue(measurement: SampleMeasurement) {
  if (measurement.decoded == null) return measurement.reason || 'No data'
  if (measurement.decoded instanceof Uint8Array) return `${measurement.decoded.byteLength} bytes`
  if (typeof measurement.decoded === 'object') return JSON.stringify(measurement.decoded)
  return `${measurement.decoded}${measurement.unit ? ` ${measurement.unit}` : ''}`
}

function FilterPanel({ criteria, measurements, onChange, onRun, running, disabled }: { criteria: Criterion[]; measurements: SampleMeasurement[]; onChange: (criteria: Criterion[]) => void; onRun: () => void; running: boolean; disabled: boolean }) {
  const update = (index: number, patch: Partial<Criterion>) => onChange(criteria.map((item, itemIndex) => itemIndex === index ? { ...item, ...patch } : item))
  return <div className="ex-tool-content">
    <div className="ex-tool-heading"><div><span>Selected point</span><strong>{criteria.filter((item) => item.enabled).length} active · {measurements.length} measurements</strong></div><button className="ex-primary-small" disabled={running || disabled} onClick={onRun} type="button">{running ? 'Scoring…' : 'Find matches'}</button></div>
    <div className="ex-filter-scroll">
      {measurements.length > 0 && <details className="ex-measurements"><summary>Measurements <span>{measurements.length}</span></summary><div>{measurements.map((measurement) => <p key={`${measurement.variableId}-${measurement.sourceId}`}><span><strong>{measurement.label || measurement.variableId}</strong><small>{measurement.variableId}</small></span><em>{measurementValue(measurement)}</em></p>)}</div></details>}
      {criteria.map((criterion, index) => <article className={`ex-filter-row${criterion.enabled ? ' is-enabled' : ''}`} key={criterion.key} title={criterion.origin}>
      <label className="ex-filter-name"><input checked={criterion.enabled} disabled={criterion.target == null && criterion.min == null && criterion.max == null} onChange={(event) => update(index, { enabled: event.target.checked, selectionTouched: true, selectedByUser: event.target.checked })} type="checkbox" /><span><strong>{criterion.label}</strong><small>{criterion.group === criterion.key ? 'Other' : criterion.group} · {criterion.unit}{criterion.kind ? ` · ${criterion.kind}` : ''}</small></span></label>
      <div className="ex-filter-values">
        <label>Target<NumberField label={`${criterion.label} target`} value={criterion.target} onChange={(value) => update(index, { target: value, targetTouched: true })} /></label>
        <label>Tolerance<NumberField label={`${criterion.label} tolerance`} value={criterion.tolerance} onChange={(value) => update(index, { tolerance: value ?? 0, toleranceTouched: true })} /></label>
        <label>Weight<NumberField label={`${criterion.label} weight`} value={criterion.weight} onChange={(value) => update(index, { weight: value ?? 0, weightTouched: true })} /></label>
        <label>Min<NumberField label={`${criterion.label} minimum`} value={criterion.min} onChange={(value) => update(index, { min: value })} /></label>
        <label>Max<NumberField label={`${criterion.label} maximum`} value={criterion.max} onChange={(value) => update(index, { max: value })} /></label>
      </div>
    </article>)}</div>
  </div>
}

function ConsolePanel({ sql, setSql, params, setParams, onRun, running, disabled, status, output }: { sql: string; setSql: (sql: string) => void; params: string; setParams: (params: string) => void; onRun: () => void; running: boolean; disabled: boolean; status: string; output: string }) {
  return <div className="ex-console"><div className="ex-console-bar"><span>earth_index.sql</span><i>{disabled ? 'Connecting' : 'gRPC-Web · connected'}</i></div><textarea aria-label="SQL query" spellCheck="false" value={sql} onChange={(event) => setSql(event.target.value)} /><label>Query parameters<textarea aria-label="Query parameters" className="ex-params" spellCheck="false" value={params} onChange={(event) => setParams(event.target.value)} /></label>{output && <pre className="ex-console-output">{output}</pre>}<div className="ex-console-actions"><span>{status}</span><button className="ex-primary-small" disabled={running || disabled} onClick={onRun} type="button">{running ? 'Running…' : 'Run query'}</button></div></div>
}

function ScoreControls({ dataset, threshold, resultLimit, cellCount, areaCount, shownCount, busy, onThreshold, onResultLimit }: { dataset: ScoreDataset; threshold: number; resultLimit: number; cellCount: number; areaCount: number; shownCount: number; busy: boolean; onThreshold: (value: number) => void; onResultLimit: (value: number) => void }) {
  const [draft, setDraft] = useState(threshold)
  const committed = useRef(threshold)
  const largest = Math.max(1, ...dataset.distribution.bins)
  const commit = (value: number) => {
    const next = Math.max(0, Math.min(100, value))
    setDraft(next)
    if (Math.abs(next - committed.current) >= .05) {
      committed.current = next
      onThreshold(next)
    }
  }
  return <div className="ex-score-controls">
    <div className="ex-distribution">
      <svg aria-label="Score distribution with logarithmic cell counts" role="img" viewBox="0 0 250 38">
        {dataset.distribution.bins.map((count, index) => {
          const height = count ? Math.max(1, Math.log1p(count) / Math.log1p(largest) * 32) : 0
          return <rect fill={`rgba(${105 + index * 2},${135 + index},247,${.25 + index / 75})`} height={height} key={index} onClick={() => commit(index * 2)} width="4" x={index * 5} y={35 - height}><title>{index * 2}–{(index + 1) * 2}% · {count.toLocaleString()} cells</title></rect>
        })}
        <line stroke="#fff" strokeWidth="1.5" x1={draft * 2.5} x2={draft * 2.5} y1="1" y2="37" />
      </svg>
      <span><i>0%</i><b>Distribution · log count</b><i>100%</i></span>
    </div>
    <label className="ex-threshold-control" title="Lower the score to expand matching areas; raise it to shrink them."><span>Minimum score <output>{draft.toFixed(1)}%</output></span><input defaultValue={threshold} disabled={busy} max="100" min="0" onBlur={(event) => commit(event.currentTarget.valueAsNumber)} onChange={(event) => setDraft(event.currentTarget.valueAsNumber)} onKeyUp={(event) => commit(event.currentTarget.valueAsNumber)} onPointerUp={(event) => commit(event.currentTarget.valueAsNumber)} step="0.1" type="range" /><small>{cellCount.toLocaleString()} cells · {areaCount.toLocaleString()} areas</small></label>
    <label className="ex-top-control"><span>Show</span><select aria-label="Number of ranked locations to show" disabled={busy} onChange={(event) => onResultLimit(Number(event.target.value))} value={resultLimit}>{[5, 10, 25, 50].map((value) => <option key={value} value={value}>{value}</option>)}</select><small>{shownCount} visible</small></label>
  </div>
}

function Brand() { return <a className="ex-brand" href="/"><span className="ex-brand-orbit"><i /></span><span><b>EARTH</b> CONNECT</span></a> }

function MatchDetail({ match, onBack }: { match: EarthMatch; onBack: () => void }) {
  return <main className="ex-root ex-detail-root"><div className="ex-stars" aria-hidden="true" /><header className="ex-header"><Brand /><button className="ex-back" onClick={onBack} type="button">← Back to matches</button></header><section className="ex-detail-wrap"><article className="ex-detail-card"><div className="ex-detail-score"><strong>{match.score}%</strong><span>similarity</span></div><p className="ex-eyebrow">Earth match · #{match.id.slice(-4).toUpperCase()}</p><h1>{match.title}</h1><p className="ex-detail-location">{match.region} · {Math.abs(match.lat).toFixed(2)}°{match.lat >= 0 ? 'N' : 'S'} / {Math.abs(match.lon).toFixed(2)}°{match.lon >= 0 ? 'E' : 'W'}</p><p className="ex-detail-copy">{match.description}</p><div className="ex-detail-tags">{match.tags.map((tag) => <span key={tag}>{tag}</span>)}</div><div className="ex-metric-grid">{match.metrics.map((metric) => <div key={metric.label}><span>{metric.label}</span><strong>{metric.value}</strong></div>)}</div><div className="ex-detail-note"><i />Earth Index · 4 km grid</div></article></section></main>
}

export default function PlanetViewer() {
  const initialBody = useMemo<BodyId>(() => new URLSearchParams(window.location.search).get('body') === 'mars' ? 'mars' : 'moon', [])
  const initialTarget = useMemo(() => initialBody === 'mars' ? { lat: 18.7, lon: -77.1 } : { lat: -18.4, lon: 33.2 }, [initialBody])
  const [body, setBody] = useState<BodyId>(initialBody)
  const [target, setTarget] = useState<Target>(initialTarget)
  const [tool, setTool] = useState<Tool>('filters')
  const [criteria, setCriteria] = useState<Criterion[]>(() => initialCriteria().map((item) => ({ ...item, enabled: false, target: null })))
  const [matches, setMatches] = useState<EarthMatch[]>([])
  const [scoreDataset, setScoreDataset] = useState<ScoreDataset | null>(null)
  const [scoreThreshold, setScoreThreshold] = useState(75)
  const [resultLimit, setResultLimit] = useState(25)
  const [matchingCellCount, setMatchingCellCount] = useState(0)
  const [matchingAreaCount, setMatchingAreaCount] = useState(0)
  const [measurements, setMeasurements] = useState<SampleMeasurement[]>([])
  const [datasource, setDatasource] = useState<EarthDatasource | null>(null)
  const [metadataStatus, setMetadataStatus] = useState('Starting Earth Index…')
  const [metadataError, setMetadataError] = useState('')
  const [sampleStatus, setSampleStatus] = useState('Loading measurements…')
  const [queryStatus, setQueryStatus] = useState('Loading Earth data…')
  const [queryError, setQueryError] = useState('')
  const [sql, setSql] = useState('SELECT *\nFROM regions\nORDER BY site_key\nLIMIT :limit;')
  const [params, setParams] = useState('{\n  "limit": 25\n}')
  const [consoleOutput, setConsoleOutput] = useState('')
  const [running, setRunning] = useState(false)
  const [detail, setDetail] = useState<EarthMatch | null>(null)
  const queryController = useRef<AbortController | null>(null)
  const cancelQuery = useCallback(() => {
    queryController.current?.abort()
    queryController.current = null
    setRunning(false)
  }, [])

  useEffect(() => {
    const controller = new AbortController()
    void loadDatasource(controller.signal).then((loaded) => {
      setDatasource(loaded)
      setCriteria((current) => criteriaFor(loaded, current))
      setMetadataStatus(`${loaded.name} · ${loaded.grids.length.toLocaleString()} grids`)
      setQueryStatus('Ready')
    }).catch((error: unknown) => {
      if (controller.signal.aborted) return
      setMetadataError(friendlyApiError(error))
      setMetadataStatus('Earth Index offline')
      setQueryStatus('Reload to retry')
    })
    return () => controller.abort()
  }, [])

  useEffect(() => {
    const controller = new AbortController()
    void samplePoint(body, target.lat, target.lon, controller.signal).then((sample) => {
      setMeasurements(sample.measurements)
      setCriteria((current) => applySample(current, body, sample))
      const available = sample.measurements.filter((item) => item.status === 1).length
      setSampleStatus(`${available}/${sample.measurements.length} measurements${sample.cached ? ' · cached' : ''}${sample.partial ? ' · partial' : ''}`)
    }).catch((error: unknown) => {
      if (controller.signal.aborted) return
      setSampleStatus(friendlyApiError(error))
    })
    return () => controller.abort()
  }, [body, target.lat, target.lon])

  useEffect(() => () => queryController.current?.abort(), [])
  const selectTarget = useCallback((next: Target) => {
    if (Math.abs(next.lat - target.lat) < 1e-8 && Math.abs(next.lon - target.lon) < 1e-8) return
    cancelQuery()
    setMeasurements([])
    setMatches([])
    setScoreDataset(null)
    setMatchingCellCount(0)
    setMatchingAreaCount(0)
    setSampleStatus('Loading measurements…')
    setCriteria((current) => current.map((item) => item.kind ? { ...item, target: null, enabled: false } : item))
    setTarget(next)
  }, [cancelQuery, target.lat, target.lon])
  const openMatch = useCallback((match: EarthMatch) => setDetail(match), [])
  const chooseBody = (nextBody: BodyId) => {
    if (nextBody === body) return
    const nextTarget = nextBody === 'moon' ? { lat: -18.4, lon: 33.2 } : { lat: 18.7, lon: -77.1 }
    cancelQuery()
    setMeasurements([])
    setSampleStatus('Loading measurements…')
    setCriteria((current) => current.map((item) => item.kind ? { ...item, target: null, enabled: false } : item))
    setBody(nextBody)
    setTarget(nextTarget)
    setMatches([])
    setScoreDataset(null)
    setMatchingCellCount(0)
    setMatchingAreaCount(0)
  }
  const runSearch = async () => {
    if (!datasource) return
    queryController.current?.abort()
    const controller = new AbortController()
    queryController.current = controller
    setRunning(true)
    setMatches([])
    setScoreDataset(null)
    setMatchingCellCount(0)
    setMatchingAreaCount(0)
    setQueryError('')
    setConsoleOutput('')
    setQueryStatus('Scoring 35.5 million cells…')
    try {
      let query = sql
      let queryParams: unknown
      if (tool === 'filters') {
        const built = buildMatchQuery(criteria)
        query = built.query
        queryParams = built.params
        setSql(built.query)
        setParams(JSON.stringify(built.params, null, 2))
      } else {
        queryParams = JSON.parse(params)
      }
      const result = await runQuery(datasource, query, queryParams, controller.signal, tool === 'filters' ? criteria : [])
      if (result.kind === 'matches') {
        setMatches(result.matches)
        setScoreDataset(result.dataset)
        setScoreThreshold(result.threshold)
        setMatchingCellCount(result.cellCount)
        setMatchingAreaCount(result.areaCount)
        setQueryStatus(result.matches.length ? result.summary : 'No cells met the cutoff')
        if (tool === 'console') setConsoleOutput(result.summary)
      } else {
        setQueryStatus(result.summary)
        setConsoleOutput(JSON.stringify(result.table.rows.slice(0, 50), (_key, value) => value instanceof Uint8Array ? `[${value.byteLength} binary bytes]` : value, 2))
      }
    } catch (error: unknown) {
      if (controller.signal.aborted) return
      const message = friendlyApiError(error)
      setQueryError(message)
      setQueryStatus(message)
    } finally {
      if (queryController.current === controller) {
        queryController.current = null
        setRunning(false)
      }
    }
  }
  const changeThreshold = async (value: number) => {
    if (!scoreDataset) return
    queryController.current?.abort()
    const controller = new AbortController()
    queryController.current = controller
    setScoreThreshold(value)
    setRunning(true)
    setQueryError('')
    setQueryStatus(`Applying ${value.toFixed(1)}% cutoff…`)
    try {
      const ranking = await rankScoreDataset(scoreDataset, criteria, value, controller.signal)
      setMatches(ranking.matches)
      setMatchingCellCount(ranking.cellCount)
      setMatchingAreaCount(ranking.areaCount)
      setQueryStatus(`${ranking.areaCount.toLocaleString()} areas · ${value.toFixed(1)}% minimum`)
    } catch (error: unknown) {
      if (controller.signal.aborted) return
      const message = friendlyApiError(error)
      setQueryError(message)
      setQueryStatus(message)
    } finally {
      if (queryController.current === controller) {
        queryController.current = null
        setRunning(false)
      }
    }
  }
  const visibleMatches = matches.slice(0, resultLimit)
  if (detail) return <MatchDetail match={detail} onBack={() => setDetail(null)} />
  const activeCount = criteria.filter((item) => item.enabled).length
  const connected = Boolean(datasource && !metadataError)
  return <main className="ex-root"><div className="ex-stars" aria-hidden="true" /><header className="ex-header"><Brand /><div className={`ex-header-meta${connected ? ' is-live' : ''}`} title={API_BASE_URL}><span>{metadataStatus}</span><i />{connected ? 'API online' : 'Connecting'}</div></header>
    <div className="ex-layout">
      <section className="ex-target-section"><div className="ex-section-label"><span>01</span><p>Select a reference world</p></div><div className="ex-body-tabs" role="tablist" aria-label="Reference planet">{(['moon', 'mars'] as BodyId[]).map((id) => <button aria-selected={body === id} className={body === id ? 'is-active' : ''} key={id} onClick={() => chooseBody(id)} role="tab" type="button">{PLANETS[id].name}</button>)}</div>
        <div className="ex-planet-card"><div className="ex-planet-copy"><p>{PLANETS[body].name}</p><span>Drag · Scroll to zoom · Click to choose</span></div><PlanetGlobe body={body} key={body} onSelect={selectTarget} target={target} /><CoordinatePicker key={`${body}-${target.lat}-${target.lon}`} onSelect={selectTarget} target={target} /><div className="ex-sample-state">{sampleStatus}</div></div>
        <div className="ex-tool-tabs" role="tablist" aria-label="Query tools"><button aria-selected={tool === 'filters'} className={tool === 'filters' ? 'is-active' : ''} onClick={() => setTool('filters')} role="tab" type="button">Filters <span>{activeCount}</span></button><button aria-selected={tool === 'console'} className={tool === 'console' ? 'is-active' : ''} onClick={() => setTool('console')} role="tab" type="button">Console <span>SQL</span></button></div>
        <div className="ex-tool-panel">{tool === 'filters' ? <FilterPanel criteria={criteria} disabled={!datasource} measurements={measurements} onChange={(next) => { cancelQuery(); setCriteria(next); setMatches([]); setScoreDataset(null); setMatchingCellCount(0); setMatchingAreaCount(0); setQueryStatus('Filters changed') }} onRun={() => void runSearch()} running={running} /> : <ConsolePanel disabled={!datasource} onRun={() => void runSearch()} output={consoleOutput} params={params} running={running} setParams={setParams} setSql={setSql} sql={sql} status={queryStatus} />}</div>
      </section>
      <section className="ex-earth-section"><div className="ex-section-label"><span>02</span><p>Find matches on Earth</p></div><div className="ex-earth-card"><div className="ex-earth-title"><div><p>Earth matches</p><span>{visibleMatches.length ? `Showing ${visibleMatches.length} of ${matchingAreaCount.toLocaleString()} areas` : 'Run Find matches to update the map'}</span></div><div className="ex-legend" aria-label="Map legend"><span><i className="is-best" />Top 5</span><span><i />Other matches</span></div></div><EarthGlobe matches={visibleMatches} onOpen={openMatch} /><div className="ex-map-note"><span>{matchingCellCount.toLocaleString()} cells above cutoff</span><span>{running ? 'Scoring…' : 'Select a marker for details'}</span></div></div>
        <div className="ex-results-head"><div><p>Ranked locations</p><span>{queryError || queryStatus}</span></div><strong>{matches[0] ? `${matches[0].score.toFixed(1)}% best match` : connected ? 'Ready' : 'Starting'}</strong>{scoreDataset && <ScoreControls areaCount={matchingAreaCount} busy={running} cellCount={matchingCellCount} dataset={scoreDataset} key={`${scoreDataset.distribution.best}-${scoreThreshold}`} onResultLimit={setResultLimit} onThreshold={(value) => void changeThreshold(value)} resultLimit={resultLimit} shownCount={visibleMatches.length} threshold={scoreThreshold} />}</div><div className="ex-results-list">{matches.length === 0 && <div className="ex-results-empty"><span>{running ? '⌁' : '◎'}</span><strong>{running ? 'Scoring Earth' : metadataError ? 'Earth Index offline' : 'No results'}</strong><p>{running ? '35.5 million cells. This can take up to two minutes.' : metadataError || 'Choose a point, check the filters, then run Find matches.'}</p></div>}{visibleMatches.map((match, index) => <button className="ex-result-card" key={match.id} onClick={() => setDetail(match)} type="button"><span className="ex-rank">{String(index + 1).padStart(2, '0')}</span><span className="ex-result-body"><strong>{match.title}</strong><small>{match.region} · {match.country}</small><em>{match.description}</em>{match.reasons && <span className="ex-match-reasons">{match.reasons.map((reason) => <i key={reason.label}><b>{reason.label}</b>{reason.comparison}<strong>{reason.similarity.toFixed(0)}%</strong></i>)}</span>}<span className="ex-tags">{match.tags.map((tag) => <i key={tag}>{tag}</i>)}</span></span><span className="ex-score"><strong>{match.score.toFixed(1)}</strong><small>% match</small><i style={{ '--score': `${match.score}%` } as CSSProperties} /></span></button>)}</div>
      </section>
    </div>
  </main>
}
