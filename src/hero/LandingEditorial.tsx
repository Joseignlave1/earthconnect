import { type ReactNode, useEffect, useMemo, useRef, useState } from 'react'
import { earthTilePath } from '../shader-test/earth-tiles.ts'
import { SurfaceRenderer } from '../shader-test/renderer.ts'
import type { EarthWorkspaceResources } from '../workspace/earthWorkspaceResources.ts'
import { scoreManifestUrl, type WorkspaceAssets } from '../workspace/workspaceAssets.ts'
import { HERO_BRIDGE_SCORE_SHADER } from './bridgeScene.ts'
import { type AstronautPoseId, CrewAstronautStage } from './crewAstronaut.ts'

type SiteId = 'jezero' | 'apollo' | 'tycho'

type SiteConfig = {
  id: SiteId
  label: string
  body: 'Mars' | 'Moon'
  coords: string
  question: string
  subline: string
}

const SITES: readonly SiteConfig[] = [
  {
    id: 'jezero',
    label: 'Jezero Crater',
    body: 'Mars',
    coords: '18.44°N, 77.45°E',
    question: 'Where on Earth feels like Jezero Crater?',
    subline: 'Gently rolling floor inside a rugged regional basin. Change the scope to watch Earth respond.',
  },
  {
    id: 'apollo',
    label: 'Tranquility Base',
    body: 'Moon',
    coords: '0.67°N, 23.47°E',
    question: 'Which plains resemble Tranquility Base?',
    subline: 'Low-slope mare terrain around the Apollo 11 landing ellipse, compared across 1 km and 5 km.',
  },
  {
    id: 'tycho',
    label: 'Tycho North Rim',
    body: 'Moon',
    coords: '41.90°S, 11.20°W',
    question: "Which ridges rise like Tycho's north rim?",
    subline: 'Steep highland crater wall standing high above the surrounding relief over 21 km.',
  },
]

function buildSiteShader(
  site: SiteId,
  opts: {
    jezeroBasin: boolean
    jezeroArid: boolean
    apolloBare: boolean
    tychoProminence: boolean
  },
): { name: string; description: string; filename: string; source: string } {
  if (site === 'jezero') {
    if (opts.jezeroBasin && opts.jezeroArid) {
      return {
        name: 'Jezero Crater · Arid basin',
        description: 'Gently rolling arid basin floor enclosed by rugged 21 km relief',
        filename: 'jezero-basin.wgsl',
        source: HERO_BRIDGE_SCORE_SHADER,
      }
    }
    if (!opts.jezeroBasin && opts.jezeroArid) {
      return {
        name: 'Jezero Crater · 5 km arid ground',
        description: 'Local 5 km relief and dry bare surface without regional basin constraint',
        filename: 'jezero-local.wgsl',
        source: `// Jezero Crater — nearby 5 km ground only (dry & bare)
fn score(cell: EarthCell) -> f32 {
  if (cell.terrain_valid_fraction < 0.9 || cell.landcover_valid_fraction < 0.9 ||
      !has_data(cell.relief_5km_m) || !has_data(cell.annual_precip_mm) ||
      !has_data(cell.bare_fraction)) {
    return NO_SCORE;
  }
  let local_relief = (cell.relief_5km_m - 95.0) / 110.0;
  let aridity = max(0.0, cell.annual_precip_mm - 180.0) / 260.0;
  let terrain = exp(-0.5 * (local_relief * local_relief + aridity * aridity));
  return terrain * smoothstep(0.15, 0.78, cell.bare_fraction);
}`,
      }
    }
    if (opts.jezeroBasin && !opts.jezeroArid) {
      return {
        name: 'Jezero Crater · Basin shape (any climate)',
        description: '5 km floor inside a 21 km regional basin regardless of rainfall or cover',
        filename: 'jezero-terrain.wgsl',
        source: `// Jezero Crater — 5 km floor & 21 km basin shape (any climate)
fn score(cell: EarthCell) -> f32 {
  if (cell.terrain_valid_fraction < 0.9 || !has_data(cell.relief_5km_m) ||
      !has_data(cell.relief_21km_m) || !has_data(cell.tpi_21km_m)) {
    return NO_SCORE;
  }
  let difference = vec3f(
    (cell.relief_5km_m - 95.0) / 110.0,
    (cell.relief_21km_m - 420.0) / 360.0,
    (cell.tpi_21km_m + 115.0) / 145.0,
  );
  return exp(-0.5 * dot(difference, difference) * 0.65);
}`,
      }
    }
    return {
      name: 'Jezero Crater · 5 km relief (any climate)',
      description: 'Local 5 km rolling terrain anywhere on Earth',
      filename: 'jezero-local-any.wgsl',
      source: `// Jezero Crater — nearby 5 km ground only (any climate)
fn score(cell: EarthCell) -> f32 {
  if (cell.terrain_valid_fraction < 0.9 || !has_data(cell.relief_5km_m)) {
    return NO_SCORE;
  }
  let local_relief = (cell.relief_5km_m - 95.0) / 110.0;
  return exp(-0.5 * local_relief * local_relief);
}`,
    }
  }

  if (site === 'apollo') {
    if (opts.apolloBare) {
      return {
        name: 'Tranquility Base · Exposed plains',
        description: 'Flat 1 km slope and gentle 5 km relief on bare exposed ground',
        filename: 'apollo-11.wgsl',
        source: `// Apollo 11 · Tranquility Base (exposed bare plains)
fn score(cell: EarthCell) -> f32 {
  if (cell.terrain_valid_fraction < 0.9 || cell.landcover_valid_fraction < 0.9 ||
      !has_data(cell.slope_mean_deg) || !has_data(cell.relief_5km_m) ||
      !has_data(cell.bare_fraction)) {
    return NO_SCORE;
  }
  let slope_err = (cell.slope_mean_deg - 1.4) / 1.5;
  let relief_err = (cell.relief_5km_m - 65.0) / 85.0;
  let terrain = exp(-0.5 * (slope_err * slope_err + relief_err * relief_err));
  return terrain * smoothstep(0.2, 0.8, cell.bare_fraction);
}`,
      }
    }
    return {
      name: 'Tranquility Base · Terrain only',
      description: 'Flat 1 km slope and gentle 5 km relief across any land cover',
      filename: 'apollo-11-terrain.wgsl',
      source: `// Apollo 11 · Tranquility Base (any ground cover)
fn score(cell: EarthCell) -> f32 {
  if (cell.terrain_valid_fraction < 0.9 ||
      !has_data(cell.slope_mean_deg) || !has_data(cell.relief_5km_m)) {
    return NO_SCORE;
  }
  let slope_err = (cell.slope_mean_deg - 1.4) / 1.5;
  let relief_err = (cell.relief_5km_m - 65.0) / 85.0;
  return exp(-0.5 * (slope_err * slope_err + relief_err * relief_err));
}`,
    }
  }

  if (opts.tychoProminence) {
    return {
      name: 'Tycho North Rim · High-prominence ridges',
      description: 'Steep slopes rising prominently above surrounding 21 km relief',
      filename: 'tycho.wgsl',
      source: `// Tycho North Rim — steep ridges standing above 21 km surroundings
fn score(cell: EarthCell) -> f32 {
  if (cell.terrain_valid_fraction < 0.9 || !has_data(cell.slope_mean_deg) ||
      !has_data(cell.relief_21km_m) || !has_data(cell.tpi_21km_m)) {
    return NO_SCORE;
  }
  let difference = vec3f(
    (cell.slope_mean_deg - 16.5) / 5.5,
    (cell.relief_21km_m - 2100.0) / 800.0,
    (cell.tpi_21km_m - 480.0) / 240.0,
  );
  return exp(-0.5 * dot(difference, difference) / 3.0);
}`,
    }
  }

  return {
    name: 'Tycho North Rim · Steepness only',
    description: 'High 1 km slope matching Tycho crater wall steepness',
    filename: 'tycho-slope.wgsl',
    source: `// Tycho North Rim — 1 km slope steepness only
fn score(cell: EarthCell) -> f32 {
  if (cell.terrain_valid_fraction < 0.9 || !has_data(cell.slope_mean_deg)) {
    return NO_SCORE;
  }
  let slope_err = (cell.slope_mean_deg - 16.5) / 5.0;
  return exp(-0.5 * slope_err * slope_err);
}`,
  }
}

const RICKROLL_URL = 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'

type TeamMember = {
  id: string
  name: string
  role: string
  bio: string
  lat: number
  lon: number
  pose: AstronautPoseId
  /** Optional photo URL taped onto this astronaut's helmet sticky note. */
  photo?: string
  socials: readonly { label: string; href: string }[]
}

const PLACEHOLDER_SOCIALS = [
  { label: 'GitHub', href: RICKROLL_URL },
  { label: 'LinkedIn', href: RICKROLL_URL },
  { label: 'Website', href: RICKROLL_URL },
] as const

/**
 * Six equal team credits (alphabetical, author last).
 * Swap placeholder roles, bios, hometown/hospital coordinates, photos, and social links when ready.
 */
const TEAM_MEMBERS: readonly TeamMember[] = [
  {
    id: 'diego',
    name: 'Diego',
    role: 'Role placeholder',
    bio: 'Short personal note or what you worked on goes here. Still trying to rescue the laptop before spaghettification.',
    lat: -34.9011,
    lon: -56.1645,
    pose: 'laptop',
    socials: PLACEHOLDER_SOCIALS,
  },
  {
    id: 'joaquin',
    name: 'Joaquín',
    role: 'Role placeholder',
    bio: 'Short personal note or what you worked on goes here. Heroically bracing both space boots against empty vacuum.',
    lat: -34.9011,
    lon: -56.1645,
    pose: 'brace',
    socials: PLACEHOLDER_SOCIALS,
  },
  {
    id: 'micaela',
    name: 'Micaela',
    role: 'Role placeholder',
    bio: 'Short personal note or what you worked on goes here. Hanging on to absolutely nothing by one EVA glove.',
    lat: -34.9011,
    lon: -56.1645,
    pose: 'hang',
    socials: PLACEHOLDER_SOCIALS,
  },
  {
    id: 'nacho',
    name: 'Nacho',
    role: 'Role placeholder',
    bio: 'Short personal note or what you worked on goes here. Tumbling sideways into the event horizon at orbital velocity.',
    lat: -34.9011,
    lon: -56.1645,
    pose: 'tumble',
    socials: PLACEHOLDER_SOCIALS,
  },
  {
    id: 'roy',
    name: 'Roy',
    role: 'Role placeholder',
    bio: 'Short personal note or what you worked on goes here. Has made peace with the curvature of spacetime.',
    lat: -34.9011,
    lon: -56.1645,
    pose: 'accept',
    socials: PLACEHOLDER_SOCIALS,
  },
  {
    id: 'cristian',
    name: 'Cristian',
    role: 'Tech Lead',
    bio: 'Short personal note or what you worked on goes here. Currently losing a fight with a gravitational singularity.',
    lat: -34.9011,
    lon: -56.1645,
    pose: 'cling',
    socials: PLACEHOLDER_SOCIALS,
  },
]

type CopilotExample = {
  id: string
  label: string
  prompt: string
  steps: readonly { call: string; detail: string }[]
  snippet: readonly { added?: boolean; code: string }[]
}

const COPILOT_EXAMPLES: readonly CopilotExample[] = [
  {
    id: 'elevation',
    label: 'Filter by elevation',
    prompt: "Keep Jezero's arid basin shape, but exclude plateaus above 2,500 m.",
    steps: [
      { call: 'edit_shader("jezero-basin.wgsl")', detail: 'Added elevation cap before basin scoring' },
      { call: 'rank_regions(limit: 3)', detail: 'Namib Escarpment · Lut Basin · Mojave Basin' },
    ],
    snippet: [
      { code: 'if (!has_data(cell.elevation_m)) { return NO_SCORE; }' },
      { added: true, code: 'if (cell.elevation_m > 2500.0) { return NO_SCORE; }' },
      { code: 'let local_relief = (cell.relief_5km_m - 95.0) / 110.0;' },
    ],
  },
  {
    id: 'compare',
    label: 'Sample another site',
    prompt: 'Build a comparison for Shackleton Crater rim on the lunar south pole.',
    steps: [
      { call: 'moon_point(-89.54, 0.0)', detail: 'Fetched LRO 1 km slope & 21 km rim prominence' },
      { call: 'create_shader("shackleton.wgsl")', detail: 'Compiled live · 31.8M Earth cells scored' },
    ],
    snippet: [
      { added: true, code: 'let rim = moon_point(-89.54, 0.0);' },
      { added: true, code: 'let slope_err = (cell.slope_mean_deg - rim.terrain.slope_1km_deg) / 4.5;' },
      { code: 'return exp(-0.5 * slope_err * slope_err);' },
    ],
  },
  {
    id: 'explain',
    label: 'Inspect top match',
    prompt: 'Why did southern Namibia outscore the Atacama on this rule?',
    steps: [
      { call: 'inspect_cell(-26.28, 15.55)', detail: '5 km relief 98 m · 21 km basin 435 m · 62 mm/yr' },
      { call: 'compare_region("Atacama")', detail: 'Higher regional relief (890 m) penalized basin match' },
    ],
    snippet: [
      { code: '// Namib Escarpment: score 0.99 · TPI -118 m · bare 0.94' },
      { code: '// Atacama Foothills: score 0.81 · TPI -240 m · bare 0.98' },
    ],
  },
]

const WGSL_TOKEN_RE =
  /(\/\/[^\n]*|\b(?:fn|let|var|if|else|return)\b|\b(?:EarthCell|f32|vec2f|vec3f|vec4f|u32|i32|bool)\b|\b(?:mars_point|moon_point|has_data|exp|dot|smoothstep|clamp|mix|sqrt|abs|max|min)\b|\bNO_SCORE\b|(?<![A-Za-z0-9_])-?\d+(?:\.\d+)?(?![A-Za-z0-9_]))/g

function highlightWgslLine(line: string, lineIdx: number): ReactNode[] {
  const parts: ReactNode[] = []
  let last = 0
  WGSL_TOKEN_RE.lastIndex = 0
  let match = WGSL_TOKEN_RE.exec(line)
  let tokIdx = 0
  while (match !== null) {
    if (match.index > last) parts.push(line.slice(last, match.index))
    const token = match[0]
    let cls: string
    if (token.startsWith('//')) cls = 'wgsl-tok-comment'
    else if (/^(fn|let|var|if|else|return)$/.test(token)) cls = 'wgsl-tok-kw'
    else if (/^(EarthCell|f32|vec2f|vec3f|vec4f|u32|i32|bool)$/.test(token)) cls = 'wgsl-tok-type'
    else if (/^(mars_point|moon_point|has_data|exp|dot|smoothstep|clamp|mix|sqrt|abs|max|min)$/.test(token))
      cls = 'wgsl-tok-fn'
    else if (token === 'NO_SCORE') cls = 'wgsl-tok-const'
    else cls = 'wgsl-tok-num'
    parts.push(
      <span key={`${lineIdx}-${tokIdx++}`} className={cls}>
        {token}
      </span>,
    )
    last = match.index + token.length
    match = WGSL_TOKEN_RE.exec(line)
  }
  if (last < line.length) parts.push(line.slice(last))
  return parts
}

function formatCoord(lat: number, lon: number, decimals = 2) {
  const ns = lat >= 0 ? 'N' : 'S'
  const ew = lon >= 0 ? 'E' : 'W'
  return `${Math.abs(lat).toFixed(decimals)}°${ns}, ${Math.abs(lon).toFixed(decimals)}°${ew}`
}

export function LandingEditorial({
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
  const [copilotId, setCopilotId] = useState<string>(COPILOT_EXAMPLES[0].id)
  const [crewIndex, setCrewIndex] = useState<number>(() => Math.floor(Math.random() * TEAM_MEMBERS.length))
  const [peak, setPeak] = useState<{ latitude: number; longitude: number } | null>(null)
  const [mapReady, setMapReady] = useState(false)
  const [atlasActivated, setAtlasActivated] = useState(() => typeof IntersectionObserver === 'undefined')
  const [crewActivated, setCrewActivated] = useState(() => typeof IntersectionObserver === 'undefined')

  const atlasSectionRef = useRef<HTMLElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const rendererRef = useRef<SurfaceRenderer | null>(null)
  const worldRef = useRef<ReturnType<EarthWorkspaceResources['acquireWorld']> | null>(null)
  const crewSectionRef = useRef<HTMLElement>(null)
  const crewCanvasRef = useRef<HTMLCanvasElement>(null)
  const crewStageRef = useRef<CrewAstronautStage | null>(null)
  const crewSwipeStartRef = useRef<number | null>(null)

  const activeSite = SITES.find(s => s.id === siteId) ?? SITES[0]
  const activeShader = useMemo(
    () => buildSiteShader(siteId, { jezeroBasin, jezeroArid, apolloBare, tychoProminence }),
    [siteId, jezeroBasin, jezeroArid, apolloBare, tychoProminence],
  )
  const shaderRef = useRef(activeShader)
  useEffect(() => {
    shaderRef.current = activeShader
  }, [activeShader])

  // Defer mounting below-the-horizon WebGPU canvases until the user scrolls near them,
  // keeping initial hero RAM/VRAM and cold-load time lean.
  useEffect(() => {
    const atlasEl = atlasSectionRef.current
    const crewEl = crewSectionRef.current
    if (typeof IntersectionObserver === 'undefined') return
    const io = new IntersectionObserver(
      entries => {
        for (const entry of entries) {
          if (entry.target === atlasEl) {
            if (entry.isIntersecting) setAtlasActivated(true)
            rendererRef.current?.setPaused(!entry.isIntersecting)
          } else if (entry.target === crewEl) {
            if (entry.isIntersecting) setCrewActivated(true)
            crewStageRef.current?.setActive(entry.isIntersecting)
          }
        }
      },
      { root: scrollerRef.current, rootMargin: '20px 0px', threshold: 0.01 },
    )
    if (atlasEl) io.observe(atlasEl)
    if (crewEl) io.observe(crewEl)
    return () => io.disconnect()
  }, [scrollerRef])

  // Mount one shared-device, transparent Equal Earth WebGPU preview (no stars, non-draggable, hover loupe)
  // only once the user scrolls near the editorial section.
  useEffect(() => {
    if (!atlasActivated) return
    const canvas = canvasRef.current
    if (!canvas) return
    const controller = new AbortController()
    const { signal } = controller
    let renderer: SurfaceRenderer | null = null
    let world: ReturnType<EarthWorkspaceResources['acquireWorld']> | null = null

    void (async () => {
      try {
        const instance = await SurfaceRenderer.create(
          canvas,
          () => {},
          () => {},
          () => {},
          {
            transparent: true,
            loupeOnly: true,
            initialProjection: 'plane',
            signal,
            acquireDevice: resources.acquireDevice,
            assets: assets.url,
            loadTerrain: resources.acquireTerrain,
            loadEarthTilePack: async (packId, packSignal) => {
              const path = earthTilePath(packId)
              try {
                if (import.meta.env.DEV) {
                  const url = await assets.url(path, packSignal)
                  if (!url.startsWith('blob:')) return url
                }
                return await assets.blob(path, packSignal)
              } catch {
                return null
              }
            },
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
        const device = instance.computeDevice
        if (!device) return
        world = resources.acquireWorld(device, {
          manifestUrl: scoreManifestUrl,
          readAsset: assets.readAsset,
        })
        worldRef.current = world

        const earth = await world.loading
        if (signal.aborted) return
        const initial = shaderRef.current
        await world.publish(async () => {
          signal.throwIfAborted()
          await earth.dispatchShader(initial.source, signal)
          signal.throwIfAborted()
          await instance.setComputedScores(earth, { name: initial.name, description: initial.description }, signal)
        })
        if (signal.aborted) return
        const best = instance.bestScoreLocation
        if (best) setPeak({ latitude: best.latitude, longitude: best.longitude })
        setMapReady(true)
      } catch {
        // Keep page clean if aborted during navigation to /find
      }
    })()

    return () => {
      controller.abort()
      rendererRef.current = null
      worldRef.current = null
      world?.release()
      renderer?.destroy()
    }
  }, [atlasActivated, assets, resources])

  // Re-dispatch live WGSL whenever the user switches sites or toggles a criterion.
  useEffect(() => {
    const renderer = rendererRef.current
    const world = worldRef.current
    if (!mapReady || !renderer || !world) return
    const controller = new AbortController()
    const { signal } = controller
    void (async () => {
      try {
        const earth = await world.loading
        if (signal.aborted) return
        await world.publish(async () => {
          signal.throwIfAborted()
          await earth.dispatchShader(activeShader.source, signal)
          signal.throwIfAborted()
          await renderer.setComputedScores(
            earth,
            { name: activeShader.name, description: activeShader.description },
            signal,
          )
        })
        if (signal.aborted) return
        const best = renderer.bestScoreLocation
        setPeak(best ? { latitude: best.latitude, longitude: best.longitude } : null)
      } catch {
        // Superseded by a newer toggle click
      }
    })()
    return () => controller.abort()
  }, [activeShader, mapReady])

  const codeLines = activeShader.source.trim().split('\n')
  const activeCopilot = COPILOT_EXAMPLES.find(item => item.id === copilotId) ?? COPILOT_EXAMPLES[0]
  const activeMember = TEAM_MEMBERS[crewIndex] ?? TEAM_MEMBERS[0]
  const activeMemberIdRef = useRef(activeMember.id)
  useEffect(() => {
    activeMemberIdRef.current = activeMember.id
  }, [activeMember.id])

  // Mount the shared-device 3D spaghettified crew astronaut canvas lazily once scrolled near the footer.
  useEffect(() => {
    if (!crewActivated) return
    const canvas = crewCanvasRef.current
    if (!canvas) return
    const controller = new AbortController()
    let stage: CrewAstronautStage | null = null

    void (async () => {
      try {
        const created = await CrewAstronautStage.create(
          canvas,
          resources.acquireDevice,
          TEAM_MEMBERS,
          activeMemberIdRef.current,
          controller.signal,
        )
        if (controller.signal.aborted) {
          created.destroy()
          return
        }
        stage = created
        crewStageRef.current = created
        created.setActive(true)
      } catch (err) {
        if (!controller.signal.aborted) console.error('Could not initialize crew astronaut stage:', err)
      }
    })()

    return () => {
      controller.abort()
      crewStageRef.current = null
      stage?.destroy()
    }
  }, [crewActivated, resources])

  useEffect(() => {
    crewStageRef.current?.setMember(activeMember.id)
  }, [activeMember.id])

  const stepCrew = (delta: number) => {
    setCrewIndex(prev => (prev + delta + TEAM_MEMBERS.length) % TEAM_MEMBERS.length)
  }

  const handleLaunchClick = (event: React.MouseEvent<HTMLAnchorElement>) => {
    if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey || event.button !== 0) return
    event.preventDefault()
    onOpenWorkspace()
  }

  return (
    <div className="editorial-root">
      {/* Opening · What Earth Connect is & why planetary-scale search matters */}
      <section className="intro-section" aria-label="About Earth Connect">
        <h2 className="intro-headline">Find where Earth matches the Moon and Mars.</h2>
        <p className="intro-body">
          Training crews, testing rovers, and reading orbital geology all rely on terrestrial analogues—places on Earth
          shaped by comparable slopes, basins, and aridity. Searching the whole planet at 4 km resolution means
          evaluating <span className="intro-emphasis">31.8 million cells</span> across elevation, regional relief,
          rainfall, and land cover. Earth Connect runs those comparisons live on your GPU, turning a global GIS search
          into an interactive atlas in the browser.
        </p>
      </section>

      {/* Live Equal Earth Comparison */}
      <section ref={atlasSectionRef} className="atlas-section" aria-label="Live planetary analogue map">
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
                  data-active={selected ? 'true' : undefined}
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
                <div className="atlas-toggle-pair">
                  <button
                    type="button"
                    className="atlas-toggle-btn"
                    data-active={!jezeroBasin ? 'true' : undefined}
                    onClick={() => setJezeroBasin(false)}
                  >
                    Nearby ground · 5 km
                  </button>
                  <span className="atlas-sep" aria-hidden="true">
                    /
                  </span>
                  <button
                    type="button"
                    className="atlas-toggle-btn"
                    data-active={jezeroBasin ? 'true' : undefined}
                    onClick={() => setJezeroBasin(true)}
                  >
                    Include the basin · 21 km
                  </button>
                </div>

                <div className="atlas-toggle-pair">
                  <button
                    type="button"
                    className="atlas-toggle-btn"
                    data-active={jezeroArid ? 'true' : undefined}
                    onClick={() => setJezeroArid(true)}
                  >
                    Dry &amp; bare
                  </button>
                  <span className="atlas-sep" aria-hidden="true">
                    /
                  </span>
                  <button
                    type="button"
                    className="atlas-toggle-btn"
                    data-active={!jezeroArid ? 'true' : undefined}
                    onClick={() => setJezeroArid(false)}
                  >
                    Any climate
                  </button>
                </div>
              </>
            )}

            {siteId === 'apollo' && (
              <div className="atlas-toggle-pair">
                <button
                  type="button"
                  className="atlas-toggle-btn"
                  data-active={apolloBare ? 'true' : undefined}
                  onClick={() => setApolloBare(true)}
                >
                  Exposed bare ground
                </button>
                <span className="atlas-sep" aria-hidden="true">
                  /
                </span>
                <button
                  type="button"
                  className="atlas-toggle-btn"
                  data-active={!apolloBare ? 'true' : undefined}
                  onClick={() => setApolloBare(false)}
                >
                  Any ground cover
                </button>
              </div>
            )}

            {siteId === 'tycho' && (
              <div className="atlas-toggle-pair">
                <button
                  type="button"
                  className="atlas-toggle-btn"
                  data-active={tychoProminence ? 'true' : undefined}
                  onClick={() => setTychoProminence(true)}
                >
                  Full ridge prominence
                </button>
                <span className="atlas-sep" aria-hidden="true">
                  /
                </span>
                <button
                  type="button"
                  className="atlas-toggle-btn"
                  data-active={!tychoProminence ? 'true' : undefined}
                  onClick={() => setTychoProminence(false)}
                >
                  Steepness only
                </button>
              </div>
            )}
          </div>

          <div className="atlas-readout">
            {peak && <span className="atlas-peak">Peak match · {formatCoord(peak.latitude, peak.longitude)}</span>}
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

      {/* Workspace Assistant · Plain-language queries & tool-driven exploration */}
      <section className="copilot-section" aria-label="Workspace AI assistant">
        <div className="copilot-grid">
          <div className="copilot-copy">
            <p className="copilot-status">Workspace assistant · In development</p>
            <h2 className="copilot-heading">Ask in plain language. Edit the shader when you want.</h2>
            <p className="copilot-body">
              You don&apos;t have to write WGSL by hand to explore another world. Describe the terrain or mission
              constraints you care about—an integrated assistant translates your question into a live GPU scoring rule,
              inspects matching regions with tools, and leaves the shader open to refine.
            </p>
            <div className="copilot-switches" role="tablist" aria-label="Assistant example prompts">
              {COPILOT_EXAMPLES.map(example => {
                const selected = example.id === activeCopilot.id
                return (
                  <button
                    key={example.id}
                    type="button"
                    role="tab"
                    aria-selected={selected}
                    className="copilot-switch-btn"
                    data-active={selected ? 'true' : undefined}
                    onClick={() => setCopilotId(example.id)}
                  >
                    {example.label}
                  </button>
                )
              })}
            </div>
          </div>

          <div className="copilot-trace" role="tabpanel" aria-label={activeCopilot.label}>
            <p className="copilot-prompt">
              <span className="copilot-chevron" aria-hidden="true">
                &gt;
              </span>
              <span>{activeCopilot.prompt}</span>
            </p>

            <ul className="copilot-steps">
              {activeCopilot.steps.map(step => (
                <li key={step.call} className="copilot-step">
                  <span className="copilot-step-call">{step.call}</span>
                  <span className="copilot-step-detail">{step.detail}</span>
                </li>
              ))}
            </ul>

            <pre className="copilot-snippet">
              <code>
                {activeCopilot.snippet.map((line, idx) => (
                  <span
                    key={`${activeCopilot.id}-${idx}`}
                    className="copilot-snippet-line"
                    data-added={line.added ? 'true' : undefined}
                  >
                    <span className="copilot-snippet-mark" aria-hidden="true">
                      {line.added ? '+' : ' '}
                    </span>
                    <span className="atlas-code-src">{highlightWgslLine(line.code, idx)}</span>
                  </span>
                ))}
              </code>
            </pre>
          </div>
        </div>
      </section>

      {/* Equal Six-Person Crew Carousel & Quiet Colophon */}
      <footer
        ref={crewSectionRef}
        className="credits-section"
        aria-label="Team and data credits"
        onKeyDown={event => {
          if (event.key === 'ArrowLeft') stepCrew(-1)
          else if (event.key === 'ArrowRight') stepCrew(1)
        }}
      >
        <div className="crew-top-bar">
          <h2 className="credits-heading">Made by</h2>

          <div className="crew-name-strip" role="tablist" aria-label="Team members">
            {TEAM_MEMBERS.map((member, idx) => {
              const selected = idx === crewIndex
              return (
                <button
                  key={member.id}
                  type="button"
                  role="tab"
                  aria-selected={selected}
                  className="crew-name-btn"
                  data-active={selected ? 'true' : undefined}
                  onClick={() => setCrewIndex(idx)}
                >
                  {member.name}
                </button>
              )
            })}
          </div>

          <div className="crew-arrows" aria-label="Carousel navigation">
            <button
              type="button"
              className="crew-arrow-btn"
              aria-label="Previous team member"
              onClick={() => stepCrew(-1)}
            >
              ←
            </button>
            <button type="button" className="crew-arrow-btn" aria-label="Next team member" onClick={() => stepCrew(1)}>
              →
            </button>
          </div>
        </div>

        <div
          className="crew-carousel-stage"
          role="tabpanel"
          aria-label={activeMember.name}
          onTouchStart={event => {
            crewSwipeStartRef.current = event.touches[0]?.clientX ?? null
          }}
          onTouchEnd={event => {
            const startX = crewSwipeStartRef.current
            crewSwipeStartRef.current = null
            const endX = event.changedTouches[0]?.clientX
            if (startX === null || endX === undefined) return
            const dx = endX - startX
            if (Math.abs(dx) > 42) stepCrew(dx < 0 ? 1 : -1)
          }}
        >
          <div className="crew-visual-col">
            <canvas
              ref={crewCanvasRef}
              className="crew-astronaut-canvas"
              role="img"
              aria-label={`3D stylized astronaut representing ${activeMember.name} being stretched toward the black hole`}
              onPointerMove={event => {
                const rect = event.currentTarget.getBoundingClientRect()
                const nx = ((event.clientX - rect.left) / Math.max(1, rect.width)) * 2 - 1
                const ny = ((event.clientY - rect.top) / Math.max(1, rect.height)) * 2 - 1
                crewStageRef.current?.setPointer(nx, ny, true)
              }}
              onPointerLeave={() => crewStageRef.current?.setPointer(0, 0, false)}
            />
          </div>

          <div className="crew-bio-col">
            <span className="crew-counter">
              {String(crewIndex + 1).padStart(2, '0')} / {String(TEAM_MEMBERS.length).padStart(2, '0')}
            </span>
            <h3 className="crew-person-name">{activeMember.name}</h3>
            <p className="crew-person-role">{activeMember.role}</p>
            <p className="crew-person-bio">{activeMember.bio}</p>
            <p className="crew-person-coord">Born at: {formatCoord(activeMember.lat, activeMember.lon, 4)}</p>

            <nav className="crew-socials" aria-label={`${activeMember.name} links`}>
              {activeMember.socials.map(social => (
                <a
                  key={social.label}
                  className="crew-social-link"
                  href={social.href}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <span>{social.label}</span>
                  <span className="crew-social-arrow" aria-hidden="true">
                    ↗
                  </span>
                </a>
              ))}
            </nav>
          </div>
        </div>

        <div className="credits-colophon">
          <p className="credits-origin">
            Created for NASA Space Apps Challenge 2026 · Special Mention · Open-source Apache-2.0
          </p>
          <p className="credits-sources">
            Terrain &amp; sky data from NASA MOLA / CTX, Lunar Reconnaissance Orbiter, GEBCO 2026, MODIS 4 km grid, NASA
            SVS Deep Star Map, and Earth Index.
          </p>
        </div>
      </footer>
    </div>
  )
}
