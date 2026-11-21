import { generateWorldTiles, type WorldNoise } from './world-noise.ts'

export type DataRequest = { kind: 'noise'; frequency: number; seed: number }

export type DatasetMessage =
  | { kind: 'progress'; done: number; total: number }
  | { kind: 'complete'; dataset: WorldNoise }
  | { kind: 'error'; message: string }

export async function prepareDataset(request: DataRequest, progress: (done: number,total: number) => void, signal?: AbortSignal): Promise<WorldNoise> {
  const tiles = generateWorldTiles(request.frequency,request.seed)
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
