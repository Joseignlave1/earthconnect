import TerrainWorker from './terrain-worker.ts?worker&inline'

export function createTerrainWorker() {
  return new TerrainWorker()
}
