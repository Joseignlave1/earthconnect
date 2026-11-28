import { strict as assert } from 'node:assert'
import test from 'node:test'
import {
  createPlanetaryPointCache,
  formatPointCall,
  type CachedPlanetarySample,
  type PersistedPointRecord,
  type PlanetaryPointStorage,
} from '../src/workspace/planetaryPointCache.ts'

const complete: CachedPlanetarySample = {
  measurements: [{ variableId: 'moon.slope_1km', unit: 'degree', status: 1, qualityFlags: [], decoded: 12.5 }],
  partial: false,
  cached: false,
}
const moon = { body: 'moon', latitude: 2, longitude: 180 } as const

function memoryStorage(
  initial: PersistedPointRecord[] = [],
): PlanetaryPointStorage & { map: Map<string, PersistedPointRecord> } {
  const map = new Map(initial.map(item => [item.key, structuredClone(item)]))
  return {
    map,
    async loadAll() {
      return [...map.values()].map(item => structuredClone(item))
    },
    async put(record) {
      map.set(record.key, structuredClone(record))
    },
    async delete(key) {
      map.delete(key)
    },
  }
}

test('samples are shared between consumers and keyed by body and canonical coordinates', async () => {
  const calls: { body: string; latitude: number; longitude: number }[] = []
  const read = createPlanetaryPointCache(async (body, latitude, longitude) => {
    calls.push({ body, latitude, longitude })
    return complete
  }, new AbortController().signal)
  const signal = new AbortController().signal
  await Promise.all([read(moon, signal), read({ ...moon, longitude: -180 }, signal)])
  await read(moon, signal)
  await read({ ...moon, body: 'mars' }, signal)
  assert.deepEqual(calls, [
    { body: 'moon', latitude: 2, longitude: -180 },
    { body: 'mars', latitude: 2, longitude: -180 },
  ])
  assert.equal(formatPointCall({ body: 'moon', latitude: 2, longitude: -14.5 }), 'moon_point(2.0, -14.5)')
})

test('cancelled revisions do not consume responses, while the next revision can reuse the request', async () => {
  const response = Promise.withResolvers<CachedPlanetarySample>()
  let calls = 0
  const read = createPlanetaryPointCache(() => {
    calls++
    return response.promise
  }, new AbortController().signal)
  const previous = new AbortController()
  const obsolete = read(moon, previous.signal)
  previous.abort()
  await assert.rejects(obsolete, { name: 'AbortError' })
  const current = read(moon, new AbortController().signal)
  response.resolve(complete)
  assert.equal(await current, complete)
  assert.equal(calls, 1)
})

test('partial and failed requests can be refreshed', async t => {
  t.mock.timers.enable({ apis: ['Date'] })
  let calls = 0
  const partial = { ...complete, partial: true }
  const read = createPlanetaryPointCache(async () => {
    calls++
    if (calls === 1) throw new Error('Unavailable')
    return calls === 2 ? partial : complete
  }, new AbortController().signal)
  const signal = new AbortController().signal
  await assert.rejects(read(moon, signal), /Unavailable/)
  assert.equal(await read(moon, signal), partial)
  assert.equal(await read(moon, signal), partial)
  t.mock.timers.tick(9000)
  assert.deepEqual(await read(moon, signal), complete)
  assert.equal(calls, 3)
})

test('complete samples persist across sessions and evict lowest-frecency entries above maxBytes', async t => {
  t.mock.timers.enable({ apis: ['Date'] })
  const storage = memoryStorage()
  const signal = new AbortController().signal
  let calls = 0
  const makeSample = (value: number): CachedPlanetarySample => ({
    measurements: [{ variableId: 'moon.elevation', unit: 'm', status: 1, qualityFlags: [], decoded: value }],
    partial: false,
    cached: false,
  })
  const p1 = { body: 'moon', latitude: 1, longitude: 1 } as const
  const p2 = { body: 'moon', latitude: 2, longitude: 2 } as const
  const p3 = { body: 'moon', latitude: 3, longitude: 3 } as const

  // Allow ~2 records in the budget so inserting a 3rd evicts the lowest-frecency entry.
  const first = createPlanetaryPointCache(
    async (_b, lat) => {
      calls++
      return makeSample(lat * 10)
    },
    signal,
    { maxBytes: 420, storage },
  )

  await first(p1, signal)
  await first(p1, signal) // bump p1 frecency
  await first(p1, signal) // bump p1 frecency again
  t.mock.timers.tick(60_000)
  await first(p2, signal) // p2 has lower frequency (1 hit) than p1 (3 hits)
  t.mock.timers.tick(60_000)
  await first(p3, signal) // triggers eviction: p2 should be evicted, p1 kept!
  assert.equal(calls, 3)

  // New cache instance hydrates from persisted storage and avoids API calls for p1 and p3.
  const second = createPlanetaryPointCache(
    async (_b, lat) => {
      calls++
      return makeSample(lat * 100)
    },
    signal,
    { maxBytes: 420, storage },
  )

  const cached1 = await second(p1, signal)
  const cached3 = await second(p3, signal)
  assert.equal(cached1.measurements[0].decoded, 10)
  assert.equal(cached3.measurements[0].decoded, 30)
  assert.equal(calls, 3)

  // p2 was evicted so it re-fetches from the API.
  const fetched2 = await second(p2, signal)
  assert.equal(fetched2.measurements[0].decoded, 200)
  assert.equal(calls, 4)
})

test('hydration evicts lowest-frecency entries regardless of storage enumeration order', async t => {
  t.mock.timers.enable({ apis: ['Date'] })
  const now = Date.now()
  const records = [
    {
      key: 'moon/1/1',
      sample: {
        measurements: [
          { variableId: 'moon.elevation', unit: 'm', status: 1, qualityFlags: [], decoded: 100 } as SampleMeasurement,
        ],
        partial: false,
        cached: true,
      },
      bytes: 180,
      score: 100,
      lastUsed: now,
    },
    {
      key: 'moon/2/2',
      sample: {
        measurements: [
          { variableId: 'moon.elevation', unit: 'm', status: 1, qualityFlags: [], decoded: 1 } as SampleMeasurement,
        ],
        partial: false,
        cached: true,
      },
      bytes: 180,
      score: 1,
      lastUsed: now,
    },
  ]
  const deleted: string[] = []
  const storage: PlanetaryPointStorage = {
    async loadAll() {
      return records
    },
    async put() {},
    async delete(key) {
      deleted.push(key)
    },
  }
  const signal = new AbortController().signal
  let fetches = 0
  const cache = createPlanetaryPointCache(
    async () => {
      fetches++
      return { measurements: [], partial: false, cached: false }
    },
    signal,
    { maxBytes: 200, storage },
  )
  const hit = await cache({ body: 'moon', latitude: 1, longitude: 1 }, signal)
  assert.equal(hit.measurements[0].decoded, 100)
  assert.equal(fetches, 0)
  assert.deepEqual(deleted, ['moon/2/2'])
})
