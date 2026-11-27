import assert from 'node:assert/strict'
import { test } from 'node:test'
import { createBootGraph } from '../src/workspace/bootGraph.ts'

test('BootGraph runs independent critical tasks in parallel and passes dependency values', async () => {
  const boot = createBootGraph()
  const events: string[] = []

  let resolveA!: (v: number) => void
  let resolveB!: (v: number) => void

  boot.task('a', {
    recordTiming: false,
    run: () =>
      new Promise<number>(resolve => {
        events.push('start:a')
        resolveA = resolve
      }),
  })
  boot.task('b', {
    recordTiming: false,
    run: () =>
      new Promise<number>(resolve => {
        events.push('start:b')
        resolveB = resolve
      }),
  })
  boot.task('c', {
    after: ['a', 'b'],
    recordTiming: false,
    run: ({ get }) => {
      events.push('start:c')
      return get<number>('a') + get<number>('b')
    },
  })

  // Flush microtasks so both independent tasks start concurrently before either resolves.
  await Promise.resolve()
  assert.deepEqual(events, ['start:a', 'start:b'])

  resolveA(10)
  await Promise.resolve()
  assert.deepEqual(events, ['start:a', 'start:b'])

  resolveB(32)
  await boot.ready('critical')
  assert.deepEqual(events, ['start:a', 'start:b', 'start:c'])
  assert.equal(boot.get<number>('c'), 42)
})

test('BootGraph parks background tasks until startBackground() without blocking critical ready()', async () => {
  const boot = createBootGraph()
  let backgroundRan = false

  boot.task('core', {
    recordTiming: false,
    run: () => 'ok',
  })
  boot.task('warm-cache', {
    after: ['core'],
    tier: 'background',
    run: ({ get }) => {
      backgroundRan = true
      return `${get<string>('core')}-warm`
    },
  })

  await boot.ready('critical')
  assert.equal(backgroundRan, false)

  boot.startBackground()
  await boot.ready('background')
  assert.equal(backgroundRan, true)
  assert.equal(boot.get<string>('warm-cache'), 'ok-warm')
})

test('BootGraph rejects on missing dependency, duplicate name, and abort', async () => {
  const controller = new AbortController()
  const boot = createBootGraph(controller.signal)

  assert.throws(() => boot.task('orphan', { after: ['missing'], recordTiming: false, run: () => 1 }), /unregistered/)
  boot.task('first', { recordTiming: false, run: () => 1 })
  assert.throws(() => boot.task('first', { recordTiming: false, run: () => 2 }), /Duplicate/)

  let dependentRan = false
  boot.task('slow', {
    recordTiming: false,
    run: ({ signal }) =>
      new Promise<void>((_resolve, reject) => {
        signal.addEventListener('abort', () => reject(signal.reason), { once: true })
      }),
  })
  boot.task('after-slow', {
    after: ['slow'],
    recordTiming: false,
    run: () => {
      dependentRan = true
    },
  })

  controller.abort(new Error('cancelled'))
  await assert.rejects(() => boot.ready('critical'), /cancelled/)
  assert.equal(dependentRan, false)
})
