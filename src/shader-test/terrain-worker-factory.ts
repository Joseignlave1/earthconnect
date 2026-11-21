export function createTerrainWorker() {
  return new Worker(new URL('./terrain-worker.ts', import.meta.url), { type: 'module' })
}
