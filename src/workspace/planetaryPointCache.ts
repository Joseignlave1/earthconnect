import type { SampleMeasurement, SampleResult } from '../earthIndexApi.ts'
import type { BodyId } from '../explorerData.ts'

export type PlanetaryCoordinates = { body: BodyId; latitude: number; longitude: number }
type LoadPoint = (body: BodyId, latitude: number, longitude: number, signal: AbortSignal) => Promise<SampleResult>

export type PersistedPointRecord = {
  key: string
  sample: SampleResult
  bytes: number
  score: number
  lastUsed: number
}

export type PlanetaryPointStorage = {
  loadAll(): Promise<PersistedPointRecord[]>
  put(record: PersistedPointRecord): Promise<void>
  delete(key: string): Promise<void>
}

export type PlanetaryPointCacheOptions = {
  maxBytes?: number
  storage?: PlanetaryPointStorage | null
}

const DEFAULT_MAX_BYTES = 1_048_576 // 1 MiB
const HALF_LIFE_MS = 7 * 24 * 60 * 60 * 1000 // 7 days
const DB_NAME = 'earth-connect-planetary-points'
const STORE_NAME = 'points'
const partialLifetime = 8000
// Retain a pending point briefly across the editor's debounce. Cancel its
// consumer immediately and cancel the request itself if no new run needs it.
const reuseWindow = 800

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

function compactSample(sample: SampleResult): SampleResult {
  return {
    partial: sample.partial,
    cached: true,
    measurements: sample.measurements.map(item => ({
      variableId: item.variableId,
      unit: item.unit,
      status: item.status,
      qualityFlags: [...item.qualityFlags],
      decoded: item.decoded,
    }) as SampleMeasurement),
  }
}

function estimateBytes(key: string, sample: SampleResult): number {
  try {
    return new TextEncoder().encode(JSON.stringify({ key, sample })).byteLength
  } catch {
    return key.length * 2 + sample.measurements.length * 160
  }
}

function decayedScore(score: number, lastUsed: number, now: number) {
  const age = Math.max(0, now - lastUsed)
  return age === 0 ? score : score * Math.pow(0.5, age / HALF_LIFE_MS)
}

function createIndexedDbStorage(): PlanetaryPointStorage | null {
  if (typeof indexedDB === 'undefined') return null
  let dbPromise: Promise<IDBDatabase | undefined> | undefined
  function openDb() {
    if (!dbPromise) {
      dbPromise = new Promise<IDBDatabase | undefined>(resolve => {
        try {
          let blocked = false
          const request = indexedDB.open(DB_NAME, 1)
          request.onupgradeneeded = () => {
            if (!request.result.objectStoreNames.contains(STORE_NAME)) {
              request.result.createObjectStore(STORE_NAME, { keyPath: 'key' })
            }
          }
          request.onsuccess = () => {
            const db = request.result
            db.onversionchange = () => { db.close(); dbPromise = undefined }
            if (blocked) db.close()
            else resolve(db)
          }
          request.onerror = () => resolve(undefined)
          request.onblocked = () => { blocked = true; resolve(undefined) }
        } catch {
          resolve(undefined)
        }
      })
    }
    return dbPromise
  }
  return {
    async loadAll() {
      const db = await openDb()
      if (!db) return []
      return new Promise<PersistedPointRecord[]>(resolve => {
        try {
          const tx = db.transaction(STORE_NAME, 'readonly')
          const req = tx.objectStore(STORE_NAME).getAll()
          tx.oncomplete = () => resolve(Array.isArray(req.result) ? req.result : [])
          tx.onerror = tx.onabort = () => resolve([])
        } catch {
          resolve([])
        }
      })
    },
    async put(record) {
      const db = await openDb()
      if (!db) return
      await new Promise<void>(resolve => {
        try {
          const tx = db.transaction(STORE_NAME, 'readwrite')
          tx.objectStore(STORE_NAME).put(record)
          tx.oncomplete = tx.onerror = tx.onabort = () => resolve()
        } catch {
          resolve()
        }
      })
    },
    async delete(key) {
      const db = await openDb()
      if (!db) return
      await new Promise<void>(resolve => {
        try {
          const tx = db.transaction(STORE_NAME, 'readwrite')
          tx.objectStore(STORE_NAME).delete(key)
          tx.oncomplete = tx.onerror = tx.onabort = () => resolve()
        } catch {
          resolve()
        }
      })
    },
  }
}

type Pending = {
  request: Promise<SampleResult>
  controller: AbortController
  consumers: number
  abortTimer?: ReturnType<typeof setTimeout>
}

type CacheEntry = {
  key: string
  sample: SampleResult
  expires: number
  bytes: number
  score: number
  lastUsed: number
  persistent: boolean
}

export function createPlanetaryPointCache(
  load: LoadPoint,
  lifetime: AbortSignal,
  options: PlanetaryPointCacheOptions = {},
) {
  const maxBytes = options.maxBytes ?? DEFAULT_MAX_BYTES
  const storage = options.storage === undefined ? createIndexedDbStorage() : options.storage
  const completed = new Map<string, CacheEntry>()
  const pending = new Map<string, Pending>()
  let totalBytes = 0

  function evictIfNeeded(now: number, keepKey?: string) {
    while (totalBytes > maxBytes && completed.size > 1) {
      let victim: CacheEntry | undefined
      let lowest = Infinity
      for (const entry of completed.values()) {
        if (entry.key === keepKey) continue
        const value = decayedScore(entry.score, entry.lastUsed, now)
        if (value < lowest || (value === lowest && victim && entry.lastUsed < victim.lastUsed)) {
          lowest = value
          victim = entry
        }
      }
      if (!victim) break
      completed.delete(victim.key)
      totalBytes -= victim.bytes
      if (victim.persistent && storage) void storage.delete(victim.key).catch(() => {})
    }
  }

  function insertEntry(entry: CacheEntry, now: number, persist: boolean) {
    const existing = completed.get(entry.key)
    if (existing) totalBytes -= existing.bytes
    completed.set(entry.key, entry)
    totalBytes += entry.bytes
    evictIfNeeded(now, entry.key)
    if (persist && entry.persistent && storage) {
      void storage.put({
        key: entry.key,
        sample: entry.sample,
        bytes: entry.bytes,
        score: entry.score,
        lastUsed: entry.lastUsed,
      }).catch(() => {})
    }
  }

  let hydration: Promise<void> | undefined = storage
    ? storage.loadAll().then(records => {
      const now = Date.now()
      for (const record of records) {
        if (!record || typeof record.key !== 'string' || !record.sample || !Array.isArray(record.sample.measurements)) continue
        if (completed.has(record.key)) continue
        const bytes = typeof record.bytes === 'number' && record.bytes > 0
          ? record.bytes
          : estimateBytes(record.key, record.sample)
        const score = typeof record.score === 'number' && Number.isFinite(record.score) && record.score > 0 ? record.score : 1
        const lastUsed = typeof record.lastUsed === 'number' && Number.isFinite(record.lastUsed) ? record.lastUsed : now
        insertEntry({
          key: record.key,
          sample: record.sample,
          expires: Infinity,
          bytes,
          score,
          lastUsed,
          persistent: true,
        }, now, false)
      }
    }).catch(() => {}).finally(() => { hydration = undefined })
    : undefined

  return async (reference: PlanetaryCoordinates, signal: AbortSignal): Promise<SampleResult> => {
    signal.throwIfAborted()
    lifetime.throwIfAborted()
    if (!coordinatesValid(reference.latitude, reference.longitude)) {
      throw new Error('Enter latitude from −90° to 90° and longitude from −180° to 180°.')
    }
    if (hydration) {
      await hydration
      signal.throwIfAborted()
      lifetime.throwIfAborted()
    }
    const point = canonicalPoint(reference)
    const key = pointKey(point)
    const now = Date.now()
    const cached = completed.get(key)
    if (cached) {
      if (cached.expires > now) {
        cached.score = decayedScore(cached.score, cached.lastUsed, now) + 1
        cached.lastUsed = now
        if (cached.persistent && storage) {
          void storage.put({
            key: cached.key,
            sample: cached.sample,
            bytes: cached.bytes,
            score: cached.score,
            lastUsed: cached.lastUsed,
          }).catch(() => {})
        }
        return cached.sample
      }
      completed.delete(key)
      totalBytes -= cached.bytes
    }

    let entry = pending.get(key)
    if (!entry) {
      const controller = new AbortController()
      const requestSignal = AbortSignal.any([controller.signal, lifetime])
      const request = load(point.body, point.latitude, point.longitude, requestSignal).then(sample => {
        requestSignal.throwIfAborted()
        const savedAt = Date.now()
        const refresh = sample.partial || !sample.measurements.some(measurement => typeof measurement.decoded === 'number')
        const storedSample = refresh ? sample : compactSample(sample)
        const bytes = estimateBytes(key, storedSample)
        insertEntry({
          key,
          sample: storedSample,
          expires: refresh ? savedAt + partialLifetime : Infinity,
          bytes,
          score: 1,
          lastUsed: savedAt,
          persistent: !refresh,
        }, savedAt, !refresh)
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
