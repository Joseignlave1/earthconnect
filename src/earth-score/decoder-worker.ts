export function createDecoderWorker() {
  return new Worker(new URL('./decoder-entry.worker.ts', import.meta.url), { type: 'module' })
}
