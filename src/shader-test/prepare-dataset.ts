import { decodeSnapshotTiles } from './score-snapshot.ts'
import { generateWorldTiles, type WorldNoise } from './world-noise.ts'

export type DataRequest = { kind: 'noise'; frequency: number; seed: number } | { kind: 'snapshot'; source: string | Blob }

export type DatasetMessage =
  | { kind: 'progress'; done: number; total: number }
  | { kind: 'complete'; dataset: WorldNoise }
  | { kind: 'error'; message: string }

async function snapshotBuffer(source: string | Blob, signal?: AbortSignal) {
  let stream
  if (typeof source === 'string') {
    const response = await fetch(source,{ signal })
    if (!response.ok || !response.body) throw new Error(`Could not load the real score snapshot (${response.status}).`)
    stream = response.body
  } else {
    stream = source.stream()
  }
  return new Response(stream.pipeThrough(new DecompressionStream('gzip'),{ signal })).arrayBuffer()
}

export async function prepareDataset(request: DataRequest, progress: (done: number,total: number) => void, signal?: AbortSignal): Promise<WorldNoise> {
  const tiles = request.kind === 'snapshot'
    ? decodeSnapshotTiles(await snapshotBuffer(request.source,signal))
    : generateWorldTiles(request.frequency,request.seed)
  signal?.throwIfAborted()
  let deadline = performance.now() + 8
  let tile = tiles.next()
  while (!tile.done) {
    progress(...tile.value)
    // The fallback shares the UI thread and must yield before starting more tiles.
    if (signal && performance.now() >= deadline) {
      await new Promise(resolve => setTimeout(resolve,0))
      signal.throwIfAborted()
      deadline = performance.now() + 8
    }
    tile = tiles.next()
  }
  return tile.value
}
