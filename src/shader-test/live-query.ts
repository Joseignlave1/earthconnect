import type { EarthScore } from '../earth-score/index.ts'
import type { BodyId, Criterion } from '../explorerData.ts'
import type { SurfaceRenderer } from './renderer.ts'
import { scoreManifestUrl } from './labAssets.ts'

type Point = { body: BodyId; latitude: number; longitude: number }
type EffectiveCriterion = Criterion & { target: number }
type Result = { name: string; criteria: EffectiveCriterion[]; partial: boolean }
type World = { device: GPUDevice; controller: AbortController; earth?: EarthScore }

const standalone = import.meta.env.VITE_SURFACE_LAB_STANDALONE === 'true'
const format = new Intl.NumberFormat(undefined, { maximumSignificantDigits: 6 })
const unitLabels: Record<string, string> = { '1': '', degree: '°', degree_Celsius: '°C' }
const kindLabels = { measured: 'Measured', approximate: 'Approximate', preference: 'Earth preference', manual: 'Manual' }
const errorMessage = (error: unknown) => error instanceof Error ? error.message : String(error)

function byId<T extends HTMLElement = HTMLElement>(id: string) {
  return document.getElementById(id) as T
}

function pointName({ body, latitude, longitude }: Point) {
  return `${body === 'moon' ? 'Moon' : 'Mars'} · ${latitude}°, ${longitude}°`
}

export function createLiveQueryControls(canvas: HTMLCanvasElement, renderer: SurfaceRenderer) {
  if (standalone || location.protocol === 'file:') {
    byId<HTMLOptionElement>('live-source').disabled = true
    byId('live-unavailable').hidden = false
    return { setActive: () => {}, refreshMap: () => {}, dispose: () => {} }
  }

  const panel = byId('live-query')
  const form = byId<HTMLFormElement>('live-form')
  const body = byId<HTMLSelectElement>('live-body')
  const latitude = byId<HTMLInputElement>('live-latitude')
  const longitude = byId<HTMLInputElement>('live-longitude')
  const run = byId<HTMLButtonElement>('run-live')
  const worldStatus = byId('earth-data-status')
  const retry = byId<HTMLButtonElement>('retry-earth-data')
  const status = byId('live-status')
  const details = byId<HTMLDetailsElement>('live-criteria')
  let active = false
  let request: AbortController | undefined
  let baseCriteria: Criterion[] | undefined
  let world: World | undefined
  let applied: Result | undefined
  let renderedCriteria: Result | undefined
  let mapName: string | undefined
  let phase = ''
  let failure = ''
  let publication = Promise.resolve()

  function updateRun() {
    run.disabled = !world?.earth || Boolean(request)
  }

  function setWorldStatus(state: 'waiting' | 'loading' | 'ready' | 'error', message: string) {
    worldStatus.dataset.state = state
    worldStatus.textContent = message
    retry.hidden = state !== 'error'
    updateRun()
  }

  function setStatus(state: 'idle' | 'loading' | 'ready' | 'stale' | 'error', message: string) {
    status.dataset.state = state
    status.textContent = message
  }

  function displayedResult() {
    return canvas.dataset.scoreName === applied?.name ? applied : undefined
  }

  function renderCriteria(result: Result | undefined) {
    if (renderedCriteria === result) return
    renderedCriteria = result
    details.hidden = !result
    if (!result) return
    details.querySelector('summary')!.textContent = `Effective criteria · ${result.criteria.length}`
    byId('live-criteria-source').textContent = `${result.name}${result.partial ? ' · partial sample' : ''}`
    const rows = result.criteria.flatMap(criterion => {
      const name = document.createElement('dt')
      name.textContent = criterion.label
      name.title = criterion.origin ?? ''
      const term = document.createElement('dd')
      const unit = unitLabels[criterion.unit] ?? criterion.unit
      const kind = criterion.kind ? ` · ${kindLabels[criterion.kind]}` : ''
      term.textContent = `${format.format(criterion.target)} ± ${format.format(criterion.tolerance)}${unit ? ` ${unit}` : ''} · weight ${criterion.weight} · ${criterion.group}${kind}`
      return [name, term]
    })
    byId('live-criteria-list').replaceChildren(...rows)
    const coverage = [...new Set(result.criteria.flatMap(criterion => criterion.qualityKey ? [criterion.qualityKey] : []))]
    const coverageLabels = coverage.map(key => key.replace(/_valid_fraction$/, '').replaceAll('_', ' '))
    byId('live-criteria-coverage').textContent = coverage.length ? `Minimum Earth coverage of 90% for ${coverageLabels.join(', ')}.` : ''
  }

  function pointFromForm(): Point {
    return { body: body.value as BodyId, latitude: latitude.valueAsNumber, longitude: longitude.valueAsNumber }
  }

  function resetStatus() {
    const result = displayedResult()
    mapName = canvas.dataset.scoreName
    renderCriteria(result)
    if (!result) setStatus('idle', `${mapName ? `The map shows ${mapName}. ` : ''}Run a query to show Earth analogues.`)
    else if (result.name === pointName(pointFromForm())) setStatus('ready', `Showing ${result.name}.${result.partial ? ' Sample coverage is partial.' : ''}`)
    else setStatus('stale', `Inputs changed. The map still shows ${result.name}.`)
  }

  function cancel() {
    request?.abort()
    request = undefined
    updateRun()
  }

  function releaseWorld() {
    const previous = world
    world = undefined
    if (!previous) return
    previous.controller.abort()
    previous.earth?.destroy()
  }

  async function loadWorld() {
    const device = renderer.computeDevice
    if (world && world.device !== device) {
      releaseWorld()
      if (request) {
        cancel()
        showError('The WebGPU device was reset. Run the query again when the Earth dataset is ready.')
      }
    }
    if (!device) {
      setWorldStatus('waiting', 'Waiting for WebGPU to load Earth data…')
      return
    }
    if (world) return
    const controller = new AbortController()
    const entry: World = { device, controller }
    world = entry
    setWorldStatus('loading', 'Loading the 183 MB Earth dataset into WebGPU…')
    try {
      const { loadEarthScore } = await import('../earth-score/index.ts')
      const earth = await loadEarthScore(device, { manifestUrl: scoreManifestUrl, signal: controller.signal })
      if (world !== entry) {
        earth.destroy()
        return
      }
      entry.earth = earth
      setWorldStatus('ready', 'Earth dataset ready on WebGPU.')
    } catch (error) {
      if (world !== entry) return
      releaseWorld()
      setWorldStatus('error', `Could not load the Earth dataset. ${errorMessage(error)}`)
    }
  }

  function showError(message: string) {
    failure = message
    const name = canvas.dataset.scoreName
    setStatus('error', `${message}${name ? ` The map still shows ${name}.` : ''}`)
  }

  function report(message: string) {
    phase = message
    const name = canvas.dataset.scoreName
    setStatus('loading', `${message}${name ? ` Map shows ${name}.` : ''}`)
  }

  async function submit() {
    const earth = world?.earth
    if (!active || request || !earth) return
    const invalid = [latitude, longitude].filter(input => !input.validity.valid)
    if (invalid.length) {
      for (const input of invalid) input.setAttribute('aria-invalid', 'true')
      showError('Enter latitude from −90 to 90 and longitude from −180 to 180.')
      invalid[0].focus()
      return
    }
    const point = pointFromForm()
    const name = pointName(point)
    const controller = new AbortController()
    const { signal } = controller
    request = controller
    updateRun()
    let describeError = errorMessage
    report(`Sampling ${name}…`)
    try {
      const [api, compute] = await Promise.all([import('../earthIndexApi.ts'), import('../earth-score/index.ts')])
      signal.throwIfAborted()
      describeError = api.friendlyApiError
      const [sample, criteria] = await Promise.all([
        api.samplePoint(point.body, point.latitude, point.longitude, signal),
        baseCriteria ?? api.loadDatasource(signal).then(api.criteriaFor),
      ])
      signal.throwIfAborted()
      baseCriteria = criteria
      describeError = errorMessage
      const sampled = api.applySample(criteria, point.body, sample)
      const effective = sampled.filter((criterion): criterion is EffectiveCriterion => criterion.enabled && criterion.target != null)
      if (!effective.length) throw new Error('This point has no measurements usable for Earth scoring. Try another coordinate.')
      const query = compute.buildEarthQuery(sampled)

      // The compute calls share output buffers with the renderer. Finish or cancel
      // their publication before a newer query can overwrite those buffers.
      const publish = publication.then(async () => {
        signal.throwIfAborted()
        report('Scoring the 4 km Earth grid on WebGPU…')
        await earth.dispatchQuery(query)
        signal.throwIfAborted()
        report('Preparing the Earth score map…')
        const description = `Live Earth analogue for ${name} · ${effective.length} effective criteria${sample.partial ? ' · partial sample' : ''}`
        await renderer.setComputedScores(earth, { name, description }, signal)
        signal.throwIfAborted()
        applied = { name, criteria: effective, partial: sample.partial }
      })
      publication = publish.catch(() => {})
      await publish
      if (!signal.aborted) resetStatus()
    } catch (error) {
      if (!signal.aborted) showError(describeError(error))
    } finally {
      if (request === controller) cancel()
    }
  }

  function onInput() {
    for (const input of [latitude, longitude]) input.removeAttribute('aria-invalid')
    cancel()
    resetStatus()
  }
  function onSubmit(event: SubmitEvent) { event.preventDefault(); void submit() }

  form.addEventListener('input', onInput)
  form.addEventListener('submit', onSubmit)
  canvas.addEventListener('computedevicechange', loadWorld)
  retry.addEventListener('click', loadWorld)
  worldStatus.hidden = false
  void loadWorld()

  return {
    refreshMap() {
      if (!active || mapName === canvas.dataset.scoreName) return
      mapName = canvas.dataset.scoreName
      renderCriteria(displayedResult())
      if (request) report(phase)
      else if (status.dataset.state === 'error') showError(failure)
      else resetStatus()
    },
    setActive(enabled: boolean) {
      if (active === enabled) return
      active = enabled
      panel.hidden = !enabled
      if (enabled) resetStatus()
      else cancel()
    },
    dispose() {
      canvas.removeEventListener('computedevicechange', loadWorld)
      retry.removeEventListener('click', loadWorld)
      form.removeEventListener('input', onInput)
      form.removeEventListener('submit', onSubmit)
      releaseWorld()
      cancel()
    },
  }
}
