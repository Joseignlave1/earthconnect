import { useEffect, useMemo, useRef, useState } from 'react'
import { earthTilePath } from '../shader-test/earth-tiles.ts'
import { SurfaceRenderer } from '../shader-test/renderer.ts'
import type { EarthWorkspaceResources } from '../workspace/earthWorkspaceResources.ts'
import { loadWorkspaceTilePack, scoreManifestUrl, type WorkspaceAssets } from '../workspace/workspaceAssets.ts'
import { buildSiteShader, formatCoord, SITES, type SiteId, type SiteShaderRule } from './editorialData.ts'
import { highlightWgslLine } from './wgslHighlight.tsx'

function TogglePair({
  active,
  firstLabel,
  secondLabel,
  firstValue,
  onChange,
}: {
  active: boolean
  firstLabel: string
  secondLabel: string
  firstValue: boolean
  onChange: (value: boolean) => void
}) {
  return (
    <div className="atlas-toggle-pair">
      <button
        type="button"
        className="atlas-toggle-btn"
        aria-pressed={active === firstValue}
        onClick={() => onChange(firstValue)}
      >
        {firstLabel}
      </button>
      <span className="atlas-sep" aria-hidden="true">
        /
      </span>
      <button
        type="button"
        className="atlas-toggle-btn"
        aria-pressed={active !== firstValue}
        onClick={() => onChange(!firstValue)}
      >
        {secondLabel}
      </button>
    </div>
  )
}

async function dispatchAtlasShader(
  renderer: SurfaceRenderer,
  world: ReturnType<EarthWorkspaceResources['acquireWorld']>,
  shader: SiteShaderRule,
  signal: AbortSignal,
) {
  const earth = await world.loading
  if (signal.aborted) return null
  await world.publish(async () => {
    signal.throwIfAborted()
    await earth.dispatchShader(shader.source, signal)
    signal.throwIfAborted()
    await renderer.setComputedScores(earth, { name: shader.name, description: shader.description }, signal)
  })
  if (signal.aborted) return null
  const best = renderer.bestScoreLocation
  return best ? { latitude: best.latitude, longitude: best.longitude } : null
}

export function AtlasSection({
  scrollerRef,
  assets,
  resources,
  onOpenWorkspace,
}: {
  scrollerRef: React.RefObject<HTMLElement | null>
  assets: WorkspaceAssets
  resources: EarthWorkspaceResources
  onOpenWorkspace: () => void
}) {
  const [siteId, setSiteId] = useState<SiteId>('jezero')
  const [jezeroBasin, setJezeroBasin] = useState(true)
  const [jezeroArid, setJezeroArid] = useState(true)
  const [apolloBare, setApolloBare] = useState(true)
  const [tychoProminence, setTychoProminence] = useState(true)
  const [showCode, setShowCode] = useState(false)
  const [peak, setPeak] = useState<{ key: string; latitude: number; longitude: number } | null>(null)
  const [atlasError, setAtlasError] = useState<string | null>(null)
  const [atlasEnv, setAtlasEnv] = useState<{
    renderer: SurfaceRenderer
    world: ReturnType<EarthWorkspaceResources['acquireWorld']>
  } | null>(null)
  const [activated, setActivated] = useState(() => typeof IntersectionObserver === 'undefined')

  const sectionRef = useRef<HTMLElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const rendererRef = useRef<SurfaceRenderer | null>(null)
  const visibleRef = useRef(typeof IntersectionObserver === 'undefined')

  const activeSite = SITES.find(s => s.id === siteId) ?? SITES[0]
  const activeShader = useMemo(
    () => buildSiteShader(siteId, { jezeroBasin, jezeroArid, apolloBare, tychoProminence }),
    [siteId, jezeroBasin, jezeroArid, apolloBare, tychoProminence],
  )
  const shaderKey = `${siteId}:${jezeroBasin}:${jezeroArid}:${apolloBare}:${tychoProminence}`
  const visiblePeak = peak && peak.key === shaderKey ? peak : null
  const codeLines = activeShader.source.trim().split('\n')

  useEffect(() => {
    const el = sectionRef.current
    if (!el || typeof IntersectionObserver === 'undefined') return
    const io = new IntersectionObserver(
      entries => {
        for (const entry of entries) {
          if (entry.target === el) {
            visibleRef.current = entry.isIntersecting
            if (entry.isIntersecting) setActivated(true)
            rendererRef.current?.setPaused(!entry.isIntersecting)
          }
        }
      },
      { root: scrollerRef.current, rootMargin: '20px 0px', threshold: 0.01 },
    )
    io.observe(el)
    return () => io.disconnect()
  }, [scrollerRef])

  useEffect(() => {
    if (!activated) return
    const canvas = canvasRef.current
    if (!canvas) return
    const controller = new AbortController()
    const { signal } = controller
    let renderer: SurfaceRenderer | null = null
    let world: ReturnType<EarthWorkspaceResources['acquireWorld']> | null = null

    const bindWorldForDevice = (instance: SurfaceRenderer, device: GPUDevice | null) => {
      world?.release()
      world = null
      if (!device || signal.aborted) {
        setAtlasEnv(null)
        return
      }
      world = resources.acquireWorld(device, {
        manifestUrl: scoreManifestUrl,
        readAsset: assets.readAsset,
      })
      setAtlasError(null)
      setAtlasEnv({ renderer: instance, world })
    }

    const onComputeDeviceChange = () => {
      if (signal.aborted || !renderer) return
      bindWorldForDevice(renderer, renderer.computeDevice)
    }
    canvas.addEventListener('computedevicechange', onComputeDeviceChange, { signal })

    void (async () => {
      try {
        const instance = await SurfaceRenderer.create(
          canvas,
          () => {},
          err => {
            if (!signal.aborted) setAtlasError(err.message)
          },
          () => {},
          {
            transparent: true,
            loupeOnly: true,
            initialProjection: 'plane',
            signal,
            acquireDevice: resources.acquireDevice,
            assets: assets.url,
            loadTerrain: resources.acquireTerrain,
            loadEarthTilePack: (packId, packSignal) => loadWorkspaceTilePack(assets, earthTilePath(packId), packSignal),
            initialOptions: {
              contextFloor: 0.08,
              contextOpacity: 0.28,
              contourOpacity: 0.2,
              relief: 18,
              bloom: 0.22,
              emission: 1.45,
              imagery: false,
            },
          },
        )
        if (signal.aborted) {
          instance.destroy()
          return
        }
        renderer = instance
        rendererRef.current = instance
        instance.setPaused(!visibleRef.current)
        bindWorldForDevice(instance, instance.computeDevice)
      } catch (err) {
        if (!signal.aborted) {
          setAtlasError(err instanceof Error ? err.message : String(err))
        }
      }
    })()

    return () => {
      controller.abort()
      rendererRef.current = null
      setAtlasEnv(null)
      world?.release()
      world = null
      renderer?.destroy()
    }
  }, [activated, assets, resources])

  useEffect(() => {
    if (!atlasEnv) return
    const controller = new AbortController()
    const { signal } = controller
    void dispatchAtlasShader(atlasEnv.renderer, atlasEnv.world, activeShader, signal)
      .then(best => {
        if (signal.aborted) return
        setAtlasError(null)
        setPeak(best ? { key: shaderKey, latitude: best.latitude, longitude: best.longitude } : null)
      })
      .catch(err => {
        if (signal.aborted || (err instanceof DOMException && err.name === 'AbortError')) return
        setAtlasError(err instanceof Error ? err.message : String(err))
      })
    return () => controller.abort()
  }, [atlasEnv, activeShader, shaderKey])

  const handleLaunchClick = (event: React.MouseEvent<HTMLAnchorElement>) => {
    if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey || event.button !== 0) return
    event.preventDefault()
    onOpenWorkspace()
  }

  return (
    <section ref={sectionRef} className="atlas-section" aria-label="Live planetary analogue map">
      <header className="atlas-header">
        <div className="atlas-site-nav" role="tablist" aria-label="Planetary comparison sites">
          {SITES.map(site => {
            const selected = site.id === siteId
            return (
              <button
                key={site.id}
                type="button"
                role="tab"
                aria-selected={selected}
                className="atlas-site-btn"
                onClick={() => setSiteId(site.id)}
              >
                {site.label}
              </button>
            )
          })}
        </div>

        <p className="atlas-origin">
          {activeSite.body} · {activeSite.coords}
        </p>
        <h3 className="atlas-question">{activeSite.question}</h3>
        <p className="atlas-subline">{activeSite.subline}</p>
      </header>

      <div className="atlas-stage">
        <canvas
          ref={canvasRef}
          className="atlas-plane-canvas"
          role="img"
          aria-label={`Equal Earth map highlighting terrestrial analogues for ${activeSite.label}. Hover over Earth to magnify regions with the optical loupe.`}
        />
      </div>

      <div className="atlas-controls-bar">
        <div className="atlas-toggles" role="group" aria-label="Comparison criteria">
          {siteId === 'jezero' && (
            <>
              <TogglePair
                active={jezeroBasin}
                firstValue={false}
                firstLabel="Nearby ground · 5 km"
                secondLabel="Include the basin · 21 km"
                onChange={setJezeroBasin}
              />
              <TogglePair
                active={jezeroArid}
                firstValue={true}
                firstLabel="Dry & bare"
                secondLabel="Any climate"
                onChange={setJezeroArid}
              />
            </>
          )}

          {siteId === 'apollo' && (
            <TogglePair
              active={apolloBare}
              firstValue={true}
              firstLabel="Exposed bare ground"
              secondLabel="Any ground cover"
              onChange={setApolloBare}
            />
          )}

          {siteId === 'tycho' && (
            <TogglePair
              active={tychoProminence}
              firstValue={true}
              firstLabel="Full ridge prominence"
              secondLabel="Steepness only"
              onChange={setTychoProminence}
            />
          )}
        </div>

        <div className="atlas-readout">
          {atlasError ? (
            <span className="atlas-peak" role="alert">
              Scoring unavailable: {atlasError}
            </span>
          ) : (
            visiblePeak && (
              <span className="atlas-peak">
                Peak match · {formatCoord(visiblePeak.latitude, visiblePeak.longitude)}
              </span>
            )
          )}
          <span className="atlas-loupe-hint">Hover Earth to magnify</span>
        </div>
      </div>

      <div className="atlas-actions">
        <a className="atlas-launch-link" href="/find" onClick={handleLaunchClick}>
          <span>Customize this comparison in the atlas</span>
          <span className="atlas-launch-arrow" aria-hidden="true">
            ↗
          </span>
        </a>

        <button
          type="button"
          className="atlas-code-toggle"
          aria-expanded={showCode}
          onClick={() => setShowCode(v => !v)}
        >
          {showCode ? 'Hide WGSL rule' : `Inspect ${activeShader.filename}`}
        </button>
      </div>

      {showCode && (
        <pre className="atlas-code-pre" aria-label={`${activeShader.filename} source`}>
          <code>
            {codeLines.map((line, idx) => (
              <span key={`${activeShader.filename}-${idx}`} className="atlas-code-line">
                <span className="atlas-code-ln" aria-hidden="true">
                  {String(idx + 1).padStart(2, '0')}
                </span>
                <span className="atlas-code-src">{highlightWgslLine(line, idx)}</span>
              </span>
            ))}
          </code>
        </pre>
      )}
    </section>
  )
}
