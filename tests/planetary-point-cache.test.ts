import { strict as assert } from 'node:assert'
import test from 'node:test'
import type { SampleResult } from '../src/earthIndexApi.ts'
import { createPlanetaryPointCache, formatPointCall } from '../src/workspace/planetaryPointCache.ts'

const complete: SampleResult = { measurements: [], partial: false, cached: false }
const moon = { body: 'moon', latitude: 2, longitude: 180 } as const

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
  const response = Promise.withResolvers<SampleResult>()
  let calls = 0
  const read = createPlanetaryPointCache(() => { calls++; return response.promise }, new AbortController().signal)
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
  assert.equal(await read(moon, signal), complete)
  assert.equal(calls, 3)
})
