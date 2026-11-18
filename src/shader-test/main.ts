import './style.css'
import { SurfaceRenderer } from './renderer.ts'
import { cellAddress } from './world-grid.ts'
import { magmaColor } from './color.ts'
import { HIGHLIGHT_START, scoreAtPalette, type ColorMode } from './score-style.ts'
import { createLiveQueryControls } from './live-query.ts'
import { earthImageUrl, flatSnapshotUrl, scoreSnapshotUrl, terrainDataUrl } from './labAssets.ts'

const surfaceAssets: Readonly<Record<string, string>> = {
  'textures/earth-blue-marble.jpg': earthImageUrl,
  'textures/earth-terrain.bin': terrainDataUrl,
}

const element = <T extends HTMLElement>(id: string) => document.getElementById(id) as T
const output = (id: string, text: string) => { element(id).textContent = text }
const error = (value: Error) => { const panel = element('error'); panel.hidden = false; panel.textContent = value.message }
let renderer: SurfaceRenderer | undefined
let liveQuery: ReturnType<typeof createLiveQueryControls> | undefined
let animation = 0
let noiseTimer: ReturnType<typeof setTimeout> | undefined
let disposed = false
let seed = 42
let refreshLegend = () => {}

async function start() {
  try {
    renderer = await SurfaceRenderer.create(element<HTMLCanvasElement>('surface'), (selection, score) => {
      const cell = selection ? cellAddress(selection.longitude, selection.latitude) : null
      output('inspection', selection && cell ? `${selection.latitude.toFixed(2)}° latitude · ${selection.longitude.toFixed(2)}° longitude · h${String(cell.h).padStart(2, '0')}v${String(cell.v).padStart(2, '0')} [${cell.x}, ${cell.y}] · score ${score == null ? 'missing' : score.toFixed(4)}${selection.elevation == null ? '' : ` · elevation ${Math.round(selection.elevation)} m`}` : 'Click the surface to inspect')
    }, error, (status) => { output('texture-info', status); refreshLegend(); liveQuery?.refreshMap() }, {
      assets: async (path, signal) => {
        signal.throwIfAborted()
        const url = surfaceAssets[path]
        if (!url) throw new Error(`Missing surface asset ${path}.`)
        return url
      },
    })
    if (disposed) { renderer.destroy(); return }
    const canvas = element<HTMLCanvasElement>('surface')
    const refreshRenderInfo = () => {
      const text = canvas.dataset.renderSize ? `${canvas.dataset.renderSize} · native 4× MSAA` : 'Preparing WebGPU…'
      if (element('render-info').textContent !== text) output('render-info',text)
      if (canvas.dataset.sunTime) {
        const latitude = Number(canvas.dataset.sunLatitude),longitude = Number(canvas.dataset.sunLongitude)
        const sun = `Sun at load · ${canvas.dataset.sunTime.slice(11,19)} UTC · ${Math.abs(latitude).toFixed(1)}°${latitude < 0 ? 'S' : 'N'}, ${Math.abs(longitude).toFixed(1)}°${longitude < 0 ? 'W' : 'E'}`
        if (element('sun-info').textContent !== sun) output('sun-info',sun)
        element('sun-info').title = `Subsolar point at ${canvas.dataset.sunTime}. Fixed until reload; ambient and fill light keep the night side readable.`
      }
    }
    canvas.addEventListener('renderstatus',refreshRenderInfo)
    refreshRenderInfo()
    element('benchmark').addEventListener('click',async () => {
      cancelAnimationFrame(animation)
      const controls = document.querySelector<HTMLElement>('.controls')!
      controls.inert = true
      try {
        const results = await renderer!.benchmark(message => output('benchmark-info',message))
        output('benchmark-info',results.map(result => `${result.activity}: ${result.median.toFixed(2)} ms`).join(' · ') + ` · ${canvas.width} × ${canvas.height}, 60 samples each. Median GPU execution; excludes CPU and presentation wait.`)
        output('benchmark-stages',Object.entries(results[0].stages).filter(([name]) => name !== 'total').map(([name,time]) => `${name}: ${time.toFixed(2)} ms`).join(' · '))
        canvas.dataset.benchmark = JSON.stringify(results)
      } catch (value) {
        output('benchmark-info',value instanceof Error ? value.message : String(value))
      } finally { controls.inert = false }
    })
    let legendKey = ''
    refreshLegend = () => {
      const canvas = element('surface'), style = renderer!.scoreStyle, options = renderer!.options
      const ready = Boolean(canvas.dataset.bestScore), locked = options.colorMode === 'locked', auto = style.kind === 'auto'
      element<HTMLButtonElement>('best-match').disabled = !ready || !renderer!.matchingScoreCount
      element<HTMLSelectElement>('color-mode').options[2].disabled = !ready
      element<HTMLInputElement>('base').disabled = locked
      for (const name of ['topRange','emphasis','focus']) {
        const input = element<HTMLInputElement>(name)
        input.hidden = name === 'focus' ? !auto : auto
        element(`${name}-label`).hidden = input.hidden
        input.disabled = locked
      }
      if (canvas.dataset.scoreDescription) output('query-description',canvas.dataset.scoreDescription)
      const key = JSON.stringify([style,options.colorMode,options.emission,options.hdr,options.minimumScore,renderer!.matchingScoreCount,ready])
      if (key === legendKey) return
      legendKey = key
      const colors = Array.from({ length: 129 },(_,i) => {
        const color = magmaColor(HIGHLIGHT_START + (1 - HIGHLIGHT_START) * i / 128)
        return `rgb(${color.map(v => Math.round(v * 255)).join(' ')}) ${i / 128 * 100}%`
      })
      element('score-ramp').style.background = `linear-gradient(to right,${colors.join(',')})`
      for (const [i,id] of ['score-low','score-quarter','score-middle','score-three-quarter','score-high'].entries()) {
        const score = scoreAtPalette(HIGHLIGHT_START + (1 - HIGHLIGHT_START) * i / 4,style)
        output(id,score.toFixed(3)); element(id).title = `Score ${score.toFixed(6)}`
      }
      const empty = style.best <= style.cutoff || (auto && !style.eligible)
      output('score-focus',!ready ? 'Calibrating colors…' : empty ? 'No bright matches above baseline · context still available' : auto
        ? `Most colors: ${style.anchors[0].toFixed(4)}–${style.best.toFixed(4)} · top ${Math.round(style.focus * 100)}% focus`
        : `Most colors: ${(style.best - (style.best - style.cutoff) * style.topRange).toFixed(4)}–${style.best.toFixed(4)}`)
      output('score-policy',locked ? 'Highlight scale and baseline locked across queries. Context and filter remain adjustable.' : auto
        ? `Relative colors · ${style.eligible.toLocaleString()} candidates above baseline. Context uses raw scores. Lock to compare queries.`
        : 'Manual highlight range; dim context uses raw scores. Lock to compare queries.')
      output('score-filter',options.minimumScore === 0 ? 'Filter off · all valid scores available.' : !ready ? 'Loading scores…' : `${renderer!.matchingScoreCount.toLocaleString()} cells ≥ ${options.minimumScore.toFixed(3)} · color scale unchanged.`)
      output('score-glow',options.hdr && auto && ready && !empty ? `Auto peak emission: ${(options.emission * style.emissionScale).toFixed(2)}×` : '')
    }
    refreshLegend()
    element<HTMLSelectElement>('color-mode').addEventListener('change',event => {
      renderer!.update({ colorMode: (event.currentTarget as HTMLSelectElement).value as ColorMode })
      refreshLegend()
    })
    const hdr = element<HTMLInputElement>('hdr')
    const hdrAvailable = element('surface').dataset.hdr === 'float16'
    const updateHdr = () => {
      output('render-mode', hdr.checked ? 'WebGPU · HDR' : 'WebGPU · SDR')
      for (const id of ['exposure','emission','bloom']) element<HTMLInputElement>(id).disabled = !hdr.checked
    }
    if (!hdrAvailable) { hdr.checked = false; hdr.disabled = true; hdr.title = 'HDR rendering is unavailable on this device.' }
    hdr.addEventListener('change',updateHdr)
    updateHdr()
    const morph = element<HTMLInputElement>('morph')
    const updateMorph = (value: number) => {
      morph.value = String(value)
      output('morph-value', `${Math.round(value * 100)}%`)
      renderer!.update({ morph: value })
    }
    morph.addEventListener('input', () => { cancelAnimationFrame(animation); updateMorph(morph.valueAsNumber) })
    element('best-match').addEventListener('click', () => { cancelAnimationFrame(animation); updateMorph(1); renderer!.focusBest() })
    for (const name of ['contextFloor', 'contextOpacity', 'minimumScore', 'focus', 'base', 'emphasis', 'topRange', 'emission', 'opacity', 'exposure', 'bloom', 'relief', 'contourInterval', 'depthInterval', 'contourOpacity'] as const) {
      const input = element<HTMLInputElement>(name)
      input.addEventListener('input', () => {
        const value = input.valueAsNumber
        output(`${name}-value`, name === 'minimumScore' && value === 0 ? 'Off' : name === 'contextOpacity' || name === 'opacity' || name === 'contourOpacity' || name === 'bloom' || name === 'topRange' || name === 'focus' ? `${Math.round(value * 100)}%` : name === 'contourInterval' || name === 'depthInterval' ? `${value} m` : name === 'relief' ? `${value}×` : name === 'exposure' || name === 'emission' ? `${value.toFixed(1)}×` : value.toFixed(name === 'minimumScore' ? 3 : name === 'base' || name === 'contextFloor' ? 2 : 1))
        renderer!.update({ [name]: input.valueAsNumber })
        refreshLegend()
      })
    }
    for (const name of ['hdr', 'grid', 'peaks', 'imagery', 'dataBoundary'] as const) {
      const input = element<HTMLInputElement>(name)
      input.addEventListener('change', () => { renderer!.update({ [name]: input.checked }); refreshLegend() })
    }
    const frequency = element<HTMLInputElement>('frequency')
    const source = element<HTMLSelectElement>('data-source')
    const snapshotLink = element<HTMLAnchorElement>('download-snapshot')
    snapshotLink.addEventListener('click', event => {
      if (snapshotLink.origin === location.origin || !/^https?:$/.test(snapshotLink.protocol)) return
      event.preventDefault()
      const filename = snapshotLink.download
      void fetch(snapshotLink.href).then(async response => {
        if (!response.ok) throw new Error(`Could not download the score snapshot (HTTP ${response.status}).`)
        const url = URL.createObjectURL(await response.blob())
        const link = document.createElement('a')
        link.href = url
        link.download = filename
        link.click()
        setTimeout(() => URL.revokeObjectURL(url), 1000)
      }).catch(error)
    })
    liveQuery = createLiveQueryControls(canvas, renderer)
    if (source.value === 'live' && element<HTMLOptionElement>('live-source').disabled) source.value = 'snapshot'
    const regenerate = () => {
      if (source.value !== 'noise') return
      try { renderer!.regenerate(frequency.valueAsNumber, seed) }
      catch (value) { error(value instanceof Error ? value : new Error(String(value))) }
    }
    const loadData = () => {
      clearTimeout(noiseTimer)
      const noise = source.value === 'noise'
      const live = source.value === 'live'
      frequency.disabled = !noise
      element<HTMLButtonElement>('regenerate').disabled = !noise
      element('error').hidden = true
      snapshotLink.hidden = noise || live
      element('fixed-query').hidden = live
      liveQuery!.setActive(live)
      if (live) return
      const flat = source.value === 'flat'
      snapshotLink.href = flat ? flatSnapshotUrl : scoreSnapshotUrl
      snapshotLink.download = flat ? 'surface-scores.bin' : 'surface-basins.bin'
      if (noise) regenerate()
      else renderer!.loadSnapshot(flat ? flatSnapshotUrl : scoreSnapshotUrl)
    }
    source.addEventListener('change',loadData)
    frequency.addEventListener('input', () => {
      output('frequency-value', frequency.valueAsNumber.toFixed(1))
      clearTimeout(noiseTimer)
      noiseTimer = setTimeout(regenerate, 180)
    })
    element('regenerate').addEventListener('click', () => { clearTimeout(noiseTimer); seed++; regenerate() })
    element('reset').addEventListener('click', () => renderer!.resetCamera())
    const transition = (target: number) => {
      cancelAnimationFrame(animation)
      renderer!.resetCamera()
      const from = morph.valueAsNumber, start = performance.now()
      const tick = (now: number) => {
        const progress = Math.min(1, (now - start) / 900)
        updateMorph(from + (target - from) * progress)
        if (progress < 1) animation = requestAnimationFrame(tick)
      }
      animation = requestAnimationFrame(tick)
    }
    element('plane').addEventListener('click', () => transition(0))
    element('globe').addEventListener('click', () => transition(1))
    loadData()
  } catch (value) { if (!disposed) error(value instanceof Error ? value : new Error(String(value))) }
}
void start()

if (import.meta.hot) import.meta.hot.dispose(() => {
  disposed = true
  liveQuery?.dispose()
  clearTimeout(noiseTimer); cancelAnimationFrame(animation); renderer?.destroy()
})
