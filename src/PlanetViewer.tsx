import { type FormEvent, useEffect, useRef, useState } from 'react'
import * as Cesium from 'cesium'
import 'cesium/Build/Cesium/Widgets/widgets.css'
import './PlanetViewer.css'

type AstroId = 'earth' | 'moon' | 'mars'
type Coordinates = { lat: number; lon: number }

const NASA_MAXIMUM_LEVEL = 7
const MARKER_ID = 'selected-location'

const ASTROS: Record<AstroId, { name: string; url: string; credit: string }> = {
  earth: { name: 'Tierra', url: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png', credit: '© OpenStreetMap contributors' },
  moon: { name: 'Luna', url: 'https://trek.nasa.gov/tiles/Moon/EQ/LRO_WAC_Mosaic_Global_303ppd_v02/1.0.0/default/default028mm/{z}/{y}/{x}.jpg', credit: 'NASA/JPL Solar System Trek' },
  mars: { name: 'Marte', url: 'https://trek.nasa.gov/tiles/Mars/EQ/Mars_Viking_MDIM21_ClrMosaic_global_232m/1.0.0/default/default028mm/{z}/{y}/{x}.jpg', credit: 'NASA/JPL Solar System Trek' },
}

type PlanetViewerProps = {
  active: boolean
  onHome: () => void
}

function createEllipsoid(astro: AstroId) {
  if (astro === 'earth') return Cesium.Ellipsoid.WGS84
  if (astro === 'moon') return new Cesium.Ellipsoid(1737400, 1737400, 1737400)
  return new Cesium.Ellipsoid(3396190, 3396190, 3376200)
}

function addMarker(viewer: Cesium.Viewer, position: Cesium.Cartesian3) {
  viewer.entities.removeById(MARKER_ID)
  viewer.entities.add({
    id: MARKER_ID,
    position,
    point: {
      pixelSize: 12,
      color: Cesium.Color.HOTPINK,
      outlineColor: Cesium.Color.WHITE,
      outlineWidth: 2,
      disableDepthTestDistance: Number.POSITIVE_INFINITY,
    },
  })
}

export default function PlanetViewer({ active, onHome }: PlanetViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const viewerRef = useRef<Cesium.Viewer | null>(null)
  const ellipsoidRef = useRef<Cesium.Ellipsoid | null>(null)
  const coordinatesRef = useRef<Coordinates | null>(null)
  const [activeAstro, setActiveAstro] = useState<AstroId>('earth')
  const [coordinates, setCoordinates] = useState<Coordinates | null>(null)
  const [latitudeInput, setLatitudeInput] = useState('')
  const [longitudeInput, setLongitudeInput] = useState('')
  const [inputError, setInputError] = useState('')
  const [copyStatus, setCopyStatus] = useState('')

  useEffect(() => {
    if (!active) return

    const container = containerRef.current
    if (!container) return

    const ellipsoid = createEllipsoid(activeAstro)
    const previousDefaultEllipsoid = Cesium.Ellipsoid.default
    Cesium.Ellipsoid.default = ellipsoid
    const astro = ASTROS[activeAstro]
    const provider = activeAstro === 'earth'
      ? new Cesium.UrlTemplateImageryProvider({ url: astro.url, maximumLevel: 19, credit: astro.credit })
      : new Cesium.UrlTemplateImageryProvider({
          url: astro.url,
          tilingScheme: new Cesium.GeographicTilingScheme({ ellipsoid }),
          maximumLevel: NASA_MAXIMUM_LEVEL,
          credit: astro.credit,
        })

    const viewer = new Cesium.Viewer(container, {
      contextOptions: { webgl: { alpha: true } },
      baseLayer: new Cesium.ImageryLayer(provider),
      terrainProvider: new Cesium.EllipsoidTerrainProvider({ ellipsoid }),
      baseLayerPicker: false,
      geocoder: false,
      homeButton: false,
      sceneModePicker: false,
      navigationHelpButton: false,
      animation: false,
      timeline: false,
      fullscreenButton: false,
      infoBox: false,
      selectionIndicator: false,
    })
    viewerRef.current = viewer
    ellipsoidRef.current = ellipsoid
    viewer.scene.backgroundColor = Cesium.Color.TRANSPARENT
    viewer.scene.highDynamicRange = false
    viewer.scene.fog.enabled = false
    viewer.scene.globe.enableLighting = false
    viewer.scene.globe.showGroundAtmosphere = false
    if (viewer.scene.skyAtmosphere) viewer.scene.skyAtmosphere.show = false
    if (viewer.scene.skyBox) viewer.scene.skyBox.show = false
    viewer.scene.globe.baseColor = Cesium.Color.fromCssColorString('#222')
    viewer.camera.setView({
      destination: Cesium.Cartesian3.fromDegrees(0, 0, ellipsoid.maximumRadius * 3, ellipsoid),
      orientation: { heading: 0, pitch: -Cesium.Math.PI_OVER_TWO, roll: 0 },
    })

    if (coordinatesRef.current) {
      const { lat, lon } = coordinatesRef.current
      addMarker(viewer, Cesium.Cartesian3.fromDegrees(lon, lat, 0, ellipsoid))
    }

    const handler = new Cesium.ScreenSpaceEventHandler(viewer.scene.canvas)
    handler.setInputAction((click: Cesium.ScreenSpaceEventHandler.PositionedEvent) => {
      const point = viewer.camera.pickEllipsoid(click.position, ellipsoid)
      if (!point) return
      const cartographic = ellipsoid.cartesianToCartographic(point)
      const lat = Cesium.Math.toDegrees(cartographic.latitude)
      const lon = Cesium.Math.toDegrees(cartographic.longitude)
      const nextCoordinates = { lat, lon }
      coordinatesRef.current = nextCoordinates
      setCoordinates(nextCoordinates)
      setLatitudeInput(lat.toFixed(5))
      setLongitudeInput(lon.toFixed(5))
      setInputError('')
      setCopyStatus('')
      addMarker(viewer, point)
    }, Cesium.ScreenSpaceEventType.LEFT_CLICK)

    return () => {
      handler.destroy()
      if (!viewer.isDestroyed()) viewer.destroy()
      if (viewerRef.current === viewer) viewerRef.current = null
      ellipsoidRef.current = null
      Cesium.Ellipsoid.default = previousDefaultEllipsoid
    }
  }, [active, activeAstro])

  useEffect(() => { coordinatesRef.current = coordinates }, [coordinates])

  const goToCoordinates = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    const lat = Number(latitudeInput)
    const lon = Number(longitudeInput)
    if (!latitudeInput.trim() || !longitudeInput.trim() || !Number.isFinite(lat) || !Number.isFinite(lon) || lat < -90 || lat > 90 || lon < -180 || lon > 180) {
      setInputError('Ingresá latitud entre -90 y 90 y longitud entre -180 y 180.')
      return
    }
    const viewer = viewerRef.current
    const ellipsoid = ellipsoidRef.current
    if (!viewer || !ellipsoid) return
    const position = Cesium.Cartesian3.fromDegrees(lon, lat, 0, ellipsoid)
    addMarker(viewer, position)
    const nextCoordinates = { lat, lon }
    coordinatesRef.current = nextCoordinates
    setCoordinates(nextCoordinates)
    setInputError('')
    setCopyStatus('')
    viewer.camera.flyTo({
      destination: Cesium.Cartesian3.fromDegrees(lon, lat, ellipsoid.maximumRadius * .35, ellipsoid),
      orientation: { heading: 0, pitch: -Cesium.Math.PI_OVER_TWO, roll: 0 },
    })
  }

  const copyCoordinates = () => {
    if (!coordinates) return
    void navigator.clipboard.writeText(`${coordinates.lat.toFixed(5)}, ${coordinates.lon.toFixed(5)}`)
      .then(() => setCopyStatus('Copiado'))
      .catch(() => setCopyStatus('No se pudo copiar'))
  }

  const longitude360 = coordinates === null ? null : ((coordinates.lon % 360) + 360) % 360

  return (
    <main className="pv-root">
      <div className="pv-canvas" ref={containerRef} />
      <nav className="pv-tabs pv-glass" aria-label="Elegir astro">
        {(Object.keys(ASTROS) as AstroId[]).map((astro) => (
          <button className={`pv-btn${activeAstro === astro ? ' is-active' : ''}`} key={astro} onClick={() => setActiveAstro(astro)} type="button">
            {ASTROS[astro].name}
          </button>
        ))}
      </nav>
      <a
        className="pv-btn pv-home"
        href="/"
        onClick={(event) => {
          if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return
          event.preventDefault()
          onHome()
        }}
      >
        Volver
      </a>
      <section className="pv-panel pv-glass" aria-live="polite">
        <h1>{ASTROS[activeAstro].name}</h1>
        {coordinates ? (
          <>
            <p>Latitud: {coordinates.lat.toFixed(5)}°</p>
            <p>Longitud: {coordinates.lon.toFixed(5)}°</p>
            {activeAstro === 'mars' && longitude360 !== null && <p>Longitud 0–360: {longitude360.toFixed(5)}°</p>}
            <button className="pv-btn pv-copy" onClick={copyCoordinates} type="button">{copyStatus || 'Copiar'}</button>
          </>
        ) : <p>Hacé clic sobre el astro para marcar un punto.</p>}
        <form className="pv-form" onSubmit={goToCoordinates}>
          <label>Lat<input aria-label="Latitud" max="90" min="-90" onChange={(event) => setLatitudeInput(event.target.value)} placeholder="Latitud" step="any" type="number" value={latitudeInput} /></label>
          <label>Lon<input aria-label="Longitud" max="180" min="-180" onChange={(event) => setLongitudeInput(event.target.value)} placeholder="Longitud" step="any" type="number" value={longitudeInput} /></label>
          <button className="pv-btn pv-go" type="submit">Ir</button>
        </form>
        {inputError && <p className="pv-error">{inputError}</p>}
      </section>
    </main>
  )
}
