import type { SampleResult } from '../earthIndexApi.ts'
import type { BodyId } from '../explorerData.ts'

export type PlanetaryCoordinates = { body: BodyId; latitude: number; longitude: number }
type LoadPoint = (body: BodyId, latitude: number, longitude: number, signal: AbortSignal) => Promise<SampleResult>

export function coordinatesValid(latitude: number, longitude: number) {
  return Number.isFinite(latitude) && Math.abs(latitude) <= 90 && Number.isFinite(longitude) && Math.abs(longitude) <= 180
}

export function canonicalPoint({ body, latitude, longitude }: PlanetaryCoordinates): PlanetaryCoordinates {
  latitude = Math.fround(latitude) || 0
  longitude = Math.fround(longitude) || 0
  return { body, latitude, longitude: Math.abs(latitude) === 90 ? 0 : longitude === 180 ? -180 : longitude }
}

export function pointKey({ body, latitude, longitude }: PlanetaryCoordinates) {
  return `${body}/${latitude}/${longitude}`
}

export function formatPointCall({ body, latitude, longitude }: PlanetaryCoordinates) {
  const literal = (value: number) => {
    const text = value.toString()
    return /[.e]/.test(text) ? text : `${text}.0`
  }
  return `${body}_point(${literal(latitude)}, ${literal(longitude)})`
}

type Pending = {
  request: Promise<SampleResult>
  controller: AbortController
  consumers: number
  abortTimer?: ReturnType<typeof setTimeout>
}

const maxCached = 128
const partialLifetime = 8000
// Retain a pending point briefly across the editor's debounce. Cancel its
// consumer immediately and cancel the request itself if no new run needs it.
const reuseWindow = 800

export function createPlanetaryPointCache(load: LoadPoint, lifetime: AbortSignal) {
  const completed = new Map<string, { sample: SampleResult; expires: number }>()
  const pending = new Map<string, Pending>()

  return async (reference: PlanetaryCoordinates, signal: AbortSignal): Promise<SampleResult> => {
    signal.throwIfAborted()
    lifetime.throwIfAborted()
    if (!coordinatesValid(reference.latitude, reference.longitude)) {
      throw new Error('Enter latitude from −90° to 90° and longitude from −180° to 180°.')
    }
    const point = canonicalPoint(reference)
    const key = pointKey(point)
    const cached = completed.get(key)
    if (cached) {
      completed.delete(key)
      if (cached.expires > Date.now()) {
        completed.set(key, cached)
        return cached.sample
      }
    }

    let entry = pending.get(key)
    if (!entry) {
      const controller = new AbortController()
      const requestSignal = AbortSignal.any([controller.signal, lifetime])
      const request = load(point.body, point.latitude, point.longitude, requestSignal).then(sample => {
        requestSignal.throwIfAborted()
        const refresh = sample.partial || !sample.measurements.some(measurement => typeof measurement.decoded === 'number')
        completed.set(key, { sample, expires: refresh ? Date.now() + partialLifetime : Infinity })
        if (completed.size > maxCached) completed.delete(completed.keys().next().value!)
        return sample
      }).finally(() => {
        const active = pending.get(key)
        if (active?.controller === controller) {
          clearTimeout(active.abortTimer)
          pending.delete(key)
        }
      })
      entry = { request, controller, consumers: 0 }
      pending.set(key, entry)
    }

    const active = entry
    clearTimeout(active.abortTimer)
    active.consumers++
    let abort: () => void = () => {}
    try {
      return await new Promise<SampleResult>((resolve, reject) => {
        abort = () => reject(signal.reason)
        signal.addEventListener('abort', abort, { once: true })
        if (signal.aborted) abort()
        void active.request.then(resolve, reject)
      })
    } finally {
      signal.removeEventListener('abort', abort)
      if (--active.consumers === 0 && pending.get(key) === active) {
        active.abortTimer = setTimeout(() => {
          pending.delete(key)
          active.controller.abort()
        }, reuseWindow)
      }
    }
  }
}
